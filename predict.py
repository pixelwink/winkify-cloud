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
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.processor = AutoImageProcessor.from_pretrained(
            "depth-anything/Depth-Anything-V2-Large-hf"
        )

        self.model = AutoModelForDepthEstimation.from_pretrained(
            "depth-anything/Depth-Anything-V2-Large-hf"
        ).to(self.device)

        self.model.eval()

        # speed boost
        if self.device == "cuda":
            self.model = self.model.half()

    # -----------------------------
    # DEPTH MODEL (FAST + CACHED USAGE)
    # -----------------------------
    def compute_depth(self, img_pil):
        with torch.no_grad():
            inputs = self.processor(images=img_pil, return_tensors="pt").to(self.device)

            if self.device == "cuda":
                inputs = {
                    k: v.half() if v.is_floating_point() else v
                    for k, v in inputs.items()
                }

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

    # -----------------------------
    # STEREO WARP
    # -----------------------------
    def warp(self, img_bgr, depth, max_shift):
        h, w = img_bgr.shape[:2]

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

    # -----------------------------
    # FRAME PROCESSOR
    # -----------------------------
    def process_frame(self, img_bgr, depth, max_shift):
        return self.warp(img_bgr, depth, max_shift)

    # -----------------------------
    # MAIN PIPELINE
    # -----------------------------
    def predict(
        self,
        file: Path = Input(description="Image or video"),
        max_shift: int = Input(default=15, ge=5, le=40),
        keyframe_interval: int = Input(default=12, ge=2, le=30),
        video_inpaint: bool = Input(default=False)
    ) -> Path:

        file_path = str(file)
        ext = os.path.splitext(file_path)[1].lower()

        # -------------------------
        # IMAGE MODE (FULL QUALITY)
        # -------------------------
        if ext in [".jpg", ".jpeg", ".png", ".webp"]:
            img = cv2.imread(file_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(img_rgb)

            # high quality full-res depth
            depth = self.compute_depth(pil)
            depth = cv2.resize(depth, (img.shape[1], img.shape[0]))

            result = self.process_frame(img, depth, max_shift)

            out = Path("/tmp/wink.jpg")
            cv2.imwrite(str(out), result)
            return out

        # -------------------------
        # VIDEO MODE (KEYFRAME DEPTH)
        # -------------------------
        elif ext in [".mp4", ".mov", ".avi", ".mkv"]:
            cap = cv2.VideoCapture(file_path)

            fps = cap.get(cv2.CAP_PROP_FPS)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            raw_out = "/tmp/wink_raw.mp4"
            final_out = "/tmp/wink_final.mp4"

            writer = cv2.VideoWriter(
                raw_out,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (w * 2, h)
            )

            frame_idx = 0
            depth_cache = None

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # -------------------------
                # KEYFRAME DEPTH LOGIC
                # -------------------------
                if frame_idx % keyframe_interval == 0:
                    small = cv2.resize(frame, (640, int(640 * h / w)))
                    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                    pil = Image.fromarray(rgb)

                    depth_cache = self.compute_depth(pil)
                    depth_cache = cv2.resize(depth_cache, (w, h))

                # reuse cached depth (THIS IS THE COST FIX)
                depth = depth_cache

                # optional: light smoothing for stability
                depth = cv2.GaussianBlur(depth, (5, 5), 0)

                result = self.process_frame(frame, depth, max_shift)

                # video mode: skip expensive inpaint by default
                writer.write(result)

                frame_idx += 1

            cap.release()
            writer.release()

            # -------------------------
            # SAFE ENCODING
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
            raise ValueError("Unsupported file format")
