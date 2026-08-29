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
            "DepthAnything/Depth-Anything-V2-Large-hf"
        )

        self.model = AutoModelForDepthEstimation.from_pretrained(
            "DepthAnything/Depth-Anything-V2-Large-hf"
        ).to(self.device)

        self.model.eval()

        # Speed boost
        if self.device == "cuda":
            self.model = self.model.half()

    # -----------------------------
    # DEPTH MODEL
    # -----------------------------
    def compute_depth(self, img_pil):
        with torch.no_grad():
            inputs = self.processor(
                images=img_pil,
                return_tensors="pt"
            ).to(self.device)

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

        x, y = np.meshgrid(
            np.arange(w),
            np.arange(h)
        )

        shift = (depth * max_shift).astype(np.float32)

        map_x_l = (x - shift).astype(np.float32)
        map_x_r = (x + shift).astype(np.float32)
        map_y = y.astype(np.float32)

        left = cv2.remap(
            img_bgr,
            map_x_l,
            map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )

        right = cv2.remap(
            img_bgr,
            map_x_r,
            map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )

        return np.hstack((left, right))

    # -----------------------------
    # FRAME PROCESSOR
    # -----------------------------
    def process_frame(self, img_bgr, depth, max_shift):
        return self.warp(
            img_bgr,
            depth,
            max_shift
        )

    # -----------------------------
    # IMAGE PIPELINE
    # -----------------------------
    def process_image(self, file_path, max_shift):
        img = cv2.imread(file_path)

        if img is None:
            raise ValueError("Could not decode input as an image.")

        img_rgb = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        pil = Image.fromarray(img_rgb)

        # High-quality full-resolution depth
        depth = self.compute_depth(pil)

        depth = cv2.resize(
            depth,
            (img.shape[1], img.shape[0])
        )

        result = self.process_frame(
            img,
            depth,
            max_shift
        )

        out = Path("/tmp/wink.jpg")

        cv2.imwrite(
            str(out),
            result,
            [cv2.IMWRITE_JPEG_QUALITY, 95]
        )

        return out

    # -----------------------------
    # VIDEO PIPELINE
    # -----------------------------
    def process_video(
        self,
        file_path,
        max_shift,
        keyframe_interval
    ):
        cap = cv2.VideoCapture(file_path)

        if not cap.isOpened():
            raise ValueError(
                "Could not decode input as a video."
            )

        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps <= 0:
            fps = 30.0

        w = int(
            cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        h = int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        if w <= 0 or h <= 0:
            cap.release()

            raise ValueError(
                "Could not determine video dimensions."
            )

        raw_out = "/tmp/wink_raw.mp4"
        final_out = "/tmp/wink_final.mp4"

        writer = cv2.VideoWriter(
            raw_out,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w * 2, h)
        )

        if not writer.isOpened():
            cap.release()

            raise ValueError(
                "Could not create output video."
            )

        frame_idx = 0
        depth_cache = None

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            # -------------------------
            # KEYFRAME DEPTH
            # -------------------------
            if (
                depth_cache is None
                or frame_idx % keyframe_interval == 0
            ):
                small_height = max(
                    1,
                    int(640 * h / w)
                )

                small = cv2.resize(
                    frame,
                    (640, small_height)
                )

                rgb = cv2.cvtColor(
                    small,
                    cv2.COLOR_BGR2RGB
                )

                pil = Image.fromarray(rgb)

                depth_cache = self.compute_depth(
                    pil
                )

                depth_cache = cv2.resize(
                    depth_cache,
                    (w, h)
                )

            depth = depth_cache

            # Light smoothing for temporal stability
            depth = cv2.GaussianBlur(
                depth,
                (5, 5),
                0
            )

            result = self.process_frame(
                frame,
                depth,
                max_shift
            )

            writer.write(result)

            frame_idx += 1

        cap.release()
        writer.release()

        if frame_idx == 0:
            raise ValueError(
                "Video contained no readable frames."
            )

        # -----------------------------
        # FINAL H.264 ENCODING
        # -----------------------------
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                raw_out,
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-preset",
                "veryfast",
                final_out
            ],
            check=True
        )

        return Path(final_out)

    # -----------------------------
    # MAIN PIPELINE
    # -----------------------------
    def predict(
        self,
        file: Path = Input(
            description="Image or video"
        ),
        max_shift: int = Input(
            default=15,
            ge=5,
            le=40
        ),
        keyframe_interval: int = Input(
            default=12,
            ge=2,
            le=30
        ),
        video_inpaint: bool = Input(
            default=False
        )
    ) -> Path:

        file_path = str(file)

        # -----------------------------
        # TRY IMAGE FIRST
        # -----------------------------
        try:
            img = cv2.imread(file_path)

            if img is not None:
                return self.process_image(
                    file_path,
                    max_shift
                )

        except Exception:
            pass

        # -----------------------------
        # IF NOT IMAGE, TRY VIDEO
        # -----------------------------
        try:
            cap = cv2.VideoCapture(file_path)

            if cap.isOpened():
                cap.release()

                return self.process_video(
                    file_path,
                    max_shift,
                    keyframe_interval
                )

            cap.release()

        except Exception:
            pass

        # -----------------------------
        # NOTHING WORKED
        # -----------------------------
        raise ValueError(
            "Unsupported file format. "
            "Could not decode input as an image or video."
        )
