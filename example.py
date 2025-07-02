import cv2
import numpy as np
from player_reid import PlayerReID

def main():
    # Initialize the player re-identification system
    reid_system = PlayerReID()
    
    # You can use your own video file or use the camera
    video_source = 0  # 0 for webcam, or provide video file path
    
    # Or use a sample video (uncomment and provide your video path)
    # video_source = "path_to_your_video.mp4"
    
    # Process the video
    reid_system.process_video(
        video_path=video_source,
        output_path="output_video.mp4",
        show_video=True
    )

if __name__ == "__main__":
    main()
