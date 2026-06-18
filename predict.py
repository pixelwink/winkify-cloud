import os
import cv2
import numpy as np
import torch
import subprocess
from PIL import Image
from cog import BasePredictor, Input, Path
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


class Predictor(BasePredictor):
    def setup(self):
        """Load model once into GPU memory"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.processor = AutoImageProcessor.from_pretrained(
            "depth-anything/Depth-Anything-V2-Large-hf"
        )

        self.model = AutoModelForDepthEstimation.from_pretrained(
            "depth-anything/Depth-Anything-V2-Large-hf"
        ).to(self.device)

        self.model.eval()
        if self.device == "cuda":
            self.model = self.model.half()

    def compute_depth(self, img_pil):
        """Fast depth inference (optimized)"""
        with torch.no_grad():
            inputs = self.processor(images=img_pil, return_tensors="pt").to(self.device)

            if self.device == "cuda":
                inputs = {k: v.half() if v.is_floating_point() else v for k, v in inputs.items()}

            outputs = self.model(**inputs)
            depth = outputs.predicted_depth

            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(1),
                size=img_pil.size[::-1],
                mode="bicubic",
                align_corners=False,
            )

        depth = depth.squeeze().float().cpu().numpy()

        dmin, dmax = depth.min(), depth.max()
        if dmax - dmin > 0:
            depth = (depth - dmin) / (dmax - dmin)

        return depth

    def process_frame(self, img_bgr, max_shift):
        """Stereo warp only (fast path)"""

        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        # 🔥 resize for speed (critical cost reduction)
        small_pil = img_pil.resize((640, int(640 * h / w)))

        depth = self.compute_depth(small_pil)
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)

        x, y = np.meshgrid(np.arange(w), np.arange(h))
        shift = (depth * max_shift).astype(np.float32)

        map_x_l = (x - shift).astype(np.float32)
        map_x_r = (x + shift).astype(np.float32)
        map_y = y.astype(np.float32)

        left = cv2.remap(
            img_bgr, map_x_l, map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )

        right = cv2.remap(
            img_bgr, map_x_r, map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )

        return np.hstack((left, right))

    def predict(
        self,
        file: Path = Input(description="Upload image or video"),
        max_shift: int = Input(default=15, ge=5, le=40),
        frame_skip: int = Input(default=2, ge=1, le=5)
    ) -> Path:

        file_path = str(file)
        ext = os.path.splitext(file_path)[1].lower()

        # -------------------------
        # IMAGE PIPELINE (fast)
        # -------------------------
        if ext in [".jpg", ".jpeg", ".png", ".webp"]:
            img = cv2.imread(file_path)
            result = self.process_frame(img, max_shift)

            out_path = Path("/tmp/wink.jpg")
            cv2.imwrite(str(out_path), result)
            return out_path

        # -------------------------
        # VIDEO PIPELINE (optimized)
        # -------------------------
        elif ext in [".mp4", ".mov", ".avi", ".mkv"]:
            cap = cv2.VideoCapture(file_path)

            fps = cap.get(cv2.CAP_PROP_FPS)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            raw_out = "/tmp/wink_raw.mp4"
            final_out = "/tmp/wink_final.mp4"

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(raw_out, fourcc, fps, (w * 2, h))

            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # 🔥 frame skipping (major cost saver)
                if frame_count % frame_skip != 0:
                    frame_count += 1
                    continue

                processed = self.process_frame(frame, max_shift)
                writer.write(processed)

                frame_count += 1

            cap.release()
            writer.release()

            # -------------------------
            # FAST SAFE FFMPEG ENCODE
            # -------------------------
            subprocess.run([
                "ffmpeg", "-y",
                "-i", raw_out,
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-preset", "veryfast",
                final_out
            ], check=True)

            return Path(final_out)

        else:
            raise ValueError("Unsupported file type")
