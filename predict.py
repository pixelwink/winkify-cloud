import os
import cv2
import numpy as np
import torch
import shutil
from PIL import Image
from cog import BasePredictor, Input, Path
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into GPU memory once"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Large-hf")
        self.model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Large-hf")
        self.model.to(self.device)

    def process_single_frame(self, img_pil, max_shift):
        """Core math logic: Calculates depth map and applies Navier-Stokes inpainted SBS shift"""
        original_img = np.array(img_pil)
        h, w, c = original_img.shape
        
        # 1. Get depth map
        inputs = self.processor(images=img_pil, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth
            
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=img_pil.size[::-1],
            mode="bicubic",
            align_corners=False,
        )
        
        depth_output = prediction.squeeze().cpu().numpy()
        depth_min, depth_max = depth_output.min(), depth_output.max()
        if depth_max - depth_min > 0:
            depth_normalized = (depth_output - depth_min) / (depth_max - depth_min)
        else:
            depth_normalized = depth_output
        depth_map = (depth_normalized * 255).astype(np.uint8)
        
        # 2. Warp perspectives
        depth_float = depth_map.astype(np.float32) / 255.0
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        shift_map = (depth_float * max_shift).astype(np.float32)
        
        map_x_l = (x - shift_map).astype(np.float32)
        map_x_r = (x + shift_map).astype(np.float32)
        map_y = y.astype(np.float32)
        
        left_raw = cv2.remap(original_img, map_x_l, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
        right_raw = cv2.remap(original_img, map_x_r, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
        
        # 3. Clean up edge tearing via Navier-Stokes inpainting
        mask_l = cv2.inRange(left_raw, np.array([0,0,0]), np.array([2,2,2]))
        mask_r = cv2.inRange(right_raw, np.array([0,0,0]), np.array([2,2,2]))
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask_l = cv2.dilate(mask_l, kernel, iterations=1)
        mask_r = cv2.dilate(mask_r, kernel, iterations=1)
        
        left_eye = cv2.inpaint(left_raw, mask_l, inpaintRadius=5, flags=cv2.INPAINT_NS)
        right_eye = cv2.inpaint(right_raw, mask_r, inpaintRadius=5, flags=cv2.INPAINT_NS)
        
        # Stack Side-by-Side configuration
        return np.hstack((left_eye, right_eye))

    def predict(
        self,
        file: Path = Input(description="Upload a 2D Photo (.jpg/.png) OR a Video (.mp4/.mov)"),
        max_shift: int = Input(description="Depth intensity split level", default=15, ge=5, le=40)
    ) -> Path:
        """Dynamically handle inputs and output matching stereoscopic format"""
        file_path = str(file)
        extension = os.path.splitext(file_path)[1].lower()
        
        # --- IMAGE PIPELINE ---
        if extension in ['.jpg', '.jpeg', '.png', '.webp']:
            print("📸 Processing flat image input...")
            img_pil = Image.open(file_path).convert("RGB")
            sbs_result = self.process_single_frame(img_pil, max_shift)
            
            out_path = Path("/tmp/wink_sbs_output.jpg")
            cv2.imwrite(str(out_path), cv2.cvtColor(sbs_result, cv2.COLOR_RGB2BGR))
            return out_path
            
        # --- VIDEO PIPELINE ---
        elif extension in ['.mp4', '.mov', '.avi', '.mkv']:
            print("🎬 Processing video clip input...")
            cap = cv2.VideoCapture(file_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Temporary storage path for processed frames
            tmp_dir = "/tmp/wink_frames"
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)
            os.makedirs(tmp_dir)
            
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert OpenCV BGR to PIL RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.open(Path(file_path)) # dummy instantiation structure
                img_pil = Image.fromarray(frame_rgb)
                
                # Process individual frame through 3D pipeline
                sbs_frame = self.process_single_frame(img_pil, max_shift)
                
                # Save out sequentially 
                cv2.imwrite(f"{tmp_dir}/frame_{frame_idx:05d}.jpg", cv2.cvtColor(sbs_frame, cv2.COLOR_RGB2BGR))
                frame_idx += 1
                
            cap.release()
            
            # Stitch processed frame sequence back into an MP4 container using FFMPEG
            video_out_path = "/tmp/wink_sbs_output.mp4"
            if os.path.exists(video_out_path):
                os.remove(video_out_path)
                
            # Compile command matching frame width * 2 (because it's horizontal SBS)
            cmd = f"ffmpeg -y -framerate {fps} -i {tmp_dir}/frame_%05d.jpg -c:v libx264 -pix_fmt yuv420p {video_out_path}"
            os.system(cmd)
            
            # Clean up frames directory to save server space
            shutil.rmtree(tmp_dir)
            
            return Path(video_out_path)
            
        else:
            raise ValueError("Unsupported file format! Please upload an image or video file.")
