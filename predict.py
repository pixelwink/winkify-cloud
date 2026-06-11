import os
import cv2
import numpy as np
import torch
from PIL import Image
from cog import BasePredictor, Input, Path
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into GPU memory once when the server boots up"""
        print("🚀 Loading Depth-Anything-V2-Large onto cloud GPU...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Upgrading to the 'Large' model since we have cloud GPU power now
        self.processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Large-hf")
        self.model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Large-hf")
        self.model.to(self.device)

    def predict(
        self,
        image: Path = Input(description="Input 2D photo to winkify"),
        max_shift: int = Input(description="Depth separation intensity (parallax)", default=20, ge=5, le=50)
    ) -> Path:
        """Run the 2D-to-SBS 3D pipeline on the cloud"""
        
        # 1. Load and process incoming image
        img_pil = Image.open(str(image)).convert("RGB")
        inputs = self.processor(images=img_pil, return_tensors="pt").to(self.device)
        
        # 2. Extract Depth Map
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth
            
        # 3. Resize depth map to match original dimensions
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=img_pil.size[::-1],
            mode="bicubic",
            align_corners=False,
        )
        
        # 4. Normalize depth map to 0-255
        depth_output = prediction.squeeze().cpu().numpy()
        depth_min, depth_max = depth_output.min(), depth_output.max()
        depth_normalized = (depth_output - depth_min) / (depth_max - depth_min)
        depth_map = (depth_normalized * 255).astype(np.uint8)
        
        # 5. Render Stereoscopic SBS Views (DIBR)
        original_img = np.array(img_pil)
        h, w, c = original_img.shape
        depth_float = depth_map.astype(np.float32) / 255.0
        
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        shift_map = (depth_float * max_shift).astype(np.float32)
        
        map_x_l = (x - shift_map).astype(np.float32)
        map_x_r = (x + shift_map).astype(np.float32)
        map_y = y.astype(np.float32)
        
        left_eye = cv2.remap(original_img, map_x_l, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        right_eye = cv2.remap(original_img, map_x_r, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        
        # 6. Stitch Side-by-Side configuration
        sbs_image = np.hstack((left_eye, right_eye))
        
        # 7. Save output file and return the cloud path
        out_path = Path("/tmp/wink_sbs_output.jpg")
        cv2.imwrite(str(out_path), cv2.cvtColor(sbs_image, cv2.COLOR_RGB2BGR))
        
        return out_path