import os
import cv2
from pathlib import Path
from player_reid import PlayerReID
from tqdm import tqdm

def process_video_folder(input_dir='video_resources', output_dir='processed_videos'):
    """
    Process all video files in the input directory and save processed videos to output directory
    
    Args:
        input_dir (str): Directory containing input videos
        output_dir (str): Directory to save processed videos
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize player re-identification system
    print("Initializing Player Re-Identification System...")
    reid_system = PlayerReID()
    
    # Supported video formats
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    
    # Get all video files in the input directory
    video_files = []
    for ext in video_extensions:
        video_files.extend(Path(input_dir).glob(f'*{ext}'))
    
    if not video_files:
        print(f"No video files found in {input_dir}")
        return
    
    print(f"Found {len(video_files)} video(s) to process")
    
    # Process each video
    for video_path in video_files:
        print(f"\nProcessing: {video_path.name}")
        output_path = Path(output_dir) / f"processed_{video_path.name}"
        
        try:
            # Process the video
            reid_system.process_video(
                video_path=str(video_path),
                output_path=str(output_path),
                show_video=False  # Set to True to display the video during processing
            )
            print(f"Successfully processed and saved to: {output_path}")
            
        except Exception as e:
            print(f"Error processing {video_path.name}: {str(e)}")
            continue

def main():
    # Process all videos in the video_resources folder
    process_video_folder()
    print("\nAll videos have been processed!")

if __name__ == "__main__":
    main()
