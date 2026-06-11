import os
import cv2
import numpy as np
import torch
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

    def predict(
        self,
        image: Path = Input(description="Input 2D photo to winkify"),
        max_shift: int = Input(description="Depth separation intensity (parallax)", default=20, ge=5, le=50)
    ) -> Path:
        """Run the 2D-to-SBS 3D pipeline with background inpainting"""
        
        # 1. Load and process incoming image
        img_pil = Image.open(str(image)).convert("RGB")
        original_img = np.array(img_pil)
        h, w, c = original_img.shape
        
        # 2. Extract Depth Map via Depth-Anything-V2-Large
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
        depth_normalized = (depth_output - depth_min) / (depth_max - depth_min)
        depth_map = (depth_normalized * 255).astype(np.uint8)
        
        # 3. Setup Perspective Warping Mapping
        depth_float = depth_map.astype(np.float32) / 255.0
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        shift_map = (depth_float * max_shift).astype(np.float32)
        
        map_x_l = (x - shift_map).astype(np.float32)
        map_x_r = (x + shift_map).astype(np.float32)
        map_y = y.astype(np.float32)
        
        # 4. Generate Raw Perspectives (this creates the empty tearing zones)
        # We use a unique border value (0,0,0) to easily isolate the stretched "tears"
        left_raw = cv2.remap(original_img, map_x_l, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
        right_raw = cv2.remap(original_img, map_x_r, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
        
        # 5. Detect and Inpaint Occlusions (The Tearing Fix)
        # Create a black-and-white mask where the pixels became pure black due to stretching
        mask_l = cv2.inRange(left_raw, np.array([0,0,0]), np.array([2,2,2]))
        mask_r = cv2.inRange(right_raw, np.array([0,0,0]), np.array([2,2,2]))
        
        # Dilate the masks slightly to ensure we catch the blurry boundaries
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask_l = cv2.dilate(mask_l, kernel, iterations=1)
        mask_r = cv2.dilate(mask_r, kernel, iterations=1)
        
        # Execute Navier-Stokes Cloud Inpainting over the missing background gaps
        print("🎨 Inpainting background occlusion layers...")
        left_eye = cv2.inpaint(left_raw, mask_l, inpaintRadius=5, flags=cv2.INPAINT_NS)
        right_eye = cv2.inpaint(right_raw, mask_r, inpaintRadius=5, flags=cv2.INPAINT_NS)
        
        # 6. Stitch Side-by-Side configuration
        sbs_image = np.hstack((left_eye, right_eye))
        
        # 7. Save output file and return cloud path
        out_path = Path("/tmp/wink_sbs_output.jpg")
        cv2.imwrite(str(out_path), cv2.cvtColor(sbs_image, cv2.COLOR_RGB2BGR))
        
        return out_path
