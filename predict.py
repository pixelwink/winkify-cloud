# --- VIDEO PIPELINE (OPTIMIZED) ---
        elif extension in ['.mp4', '.mov', '.avi', '.mkv']:
            print("🎬 Processing video clip input with memory-stream optimization...")
            cap = cv2.VideoCapture(file_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Setup a direct video writer straight to the temp MP4 container
            # The output width is doubled because it's side-by-side (w * 2)
            video_out_path = "/tmp/wink_sbs_output.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
            out = cv2.VideoWriter(video_out_path, fourcc, fps, (width * 2, height))
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert OpenCV BGR to PIL RGB seamlessly
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(frame_rgb)
                
                # Process frame entirely in GPU VRAM
                sbs_frame = self.process_single_frame(img_pil, max_shift)
                
                # Write the memory array directly into the video stream container
                out.write(cv2.cvtColor(sbs_frame, cv2.COLOR_RGB2BGR))
                
            cap.release()
            out.release()
            
            # Quick H.264 compression pass so mobile devices can stream it instantly
            final_compressed_path = "/tmp/wink_final.mp4"
            cmd = f"ffmpeg -y -i {video_out_path} -c:v libx264 -pix_fmt yuv420p {final_compressed_path}"
            os.system(cmd)
            
            return Path(final_compressed_path)
