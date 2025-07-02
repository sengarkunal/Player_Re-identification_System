# Player Re-Identification System

This project implements a player re-identification system for sports videos. It can detect players in video frames and track their identities across frames using deep learning-based feature extraction.

## Features

- Player detection using Faster R-CNN
- Feature extraction using ResNet50
- Player re-identification using cosine similarity
- Real-time video processing
- Visualization of player tracking

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd Player-Re-Identification
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your input video file in the project directory (or provide the full path).

2. Run the player re-identification script:
```bash
python player_reid.py
```

3. The script will:
   - Process the input video
   - Detect and track players
   - Display the output in real-time
   - Save the processed video as `output_video.mp4`

## Configuration

You can modify the following parameters in the `main()` function of `player_reid.py`:

- `video_path`: Path to the input video file
- `output_path`: Path to save the output video
- `confidence_threshold`: Detection confidence threshold (default: 0.8)
- `similarity_threshold`: Feature similarity threshold for re-identification (default: 0.7)

## Requirements

- Python 3.7+
- PyTorch
- OpenCV
- NumPy
- scikit-learn
- tqdm

## Notes

- The system works best with clear views of players
- Performance may vary depending on video quality and camera angle
- For better results, you may need to fine-tune the model on your specific dataset

## License

This project is open source and available under the MIT License.
