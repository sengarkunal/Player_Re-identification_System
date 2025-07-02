import os
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')  # Suppress deprecation warnings

class PlayerReID:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.detection_model = self._load_detection_model()
        self.feature_extractor = self._load_feature_extractor()
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.known_players = {}
        self.next_player_id = 1

    def _load_detection_model(self):
        """Load the player detection model (Faster R-CNN)"""
        # Load model with pretrained weights
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn(weights=weights)
        model.eval()
        model.to(self.device)
        return model

    def _load_feature_extractor(self):
        """Load feature extraction model (ResNet50 without the final layer)"""
        # Load model with pretrained weights
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
        model = torch.nn.Sequential(*(list(model.children())[:-1]))  # Remove the last layer
        model.eval()
        model.to(self.device)
        return model

    def detect_players(self, frame, confidence_threshold=0.8):
        """Detect players in a frame using the detection model"""
        # Convert frame to tensor
        img_tensor = torch.from_numpy(frame).permute(2, 0, 1).float().to(self.device)
        img_tensor = img_tensor.unsqueeze(0) / 255.0

        with torch.no_grad():
            predictions = self.detection_model(img_tensor)
        
        # Filter detections (class 1 is 'person' in COCO dataset)
        boxes = predictions[0]['boxes'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()
        
        # Filter by confidence and class (person)
        mask = (scores >= confidence_threshold) & (labels == 1)
        boxes = boxes[mask].astype(int)
        
        return boxes

    def extract_features(self, frame, bbox):
        """Extract features for a player using the feature extractor"""
        x1, y1, x2, y2 = bbox
        player_img = frame[y1:y2, x1:x2]
        if player_img.size == 0:
            return None
            
        # Preprocess the image
        img_tensor = self.transform(player_img).unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.feature_extractor(img_tensor)
        
        return features.squeeze().cpu().numpy()

    def match_player(self, features, threshold=0.7):
        """Match player features with known players"""
        if not self.known_players:
            return None
            
        # Calculate similarity with known players
        similarities = []
        for player_id, player_features in self.known_players.items():
            sim = cosine_similarity(
                features.reshape(1, -1),
                player_features.reshape(1, -1)
            )[0][0]
            similarities.append((player_id, sim))
        
        # Return best match if similarity is above threshold
        if similarities:
            best_match = max(similarities, key=lambda x: x[1])
            if best_match[1] >= threshold:
                return best_match[0]
        
        return None

    def process_video(self, video_path, output_path=None, show_video=True):
        """Process a video to detect and re-identify players"""
        # Handle webcam input
        if isinstance(video_path, int):
            cap = cv2.VideoCapture(video_path)
        else:
            cap = cv2.VideoCapture(str(video_path))
            
        if not cap.isOpened():
            print(f"Error: Could not open video source: {video_path}")
            return

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Initialize video writer if output path is provided
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Process each frame
        for _ in tqdm(range(total_frames), desc="Processing video"):
            ret, frame = cap.read()
            if not ret:
                break

            # Detect players in the frame
            boxes = self.detect_players(frame)
            
            # Process each detected player
            for box in boxes:
                x1, y1, x2, y2 = box
                
                # Extract features
                features = self.extract_features(frame, box)
                if features is None:
                    continue
                
                # Try to match with known players
                player_id = self.match_player(features)
                
                # If no match, assign new ID
                if player_id is None:
                    player_id = self.next_player_id
                    self.known_players[player_id] = features
                    self.next_player_id += 1
                else:
                    # Update features with running average
                    self.known_players[player_id] = 0.8 * self.known_players[player_id] + 0.2 * features
                
                # Draw bounding box and ID
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Player {player_id}", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Write frame to output video
            if output_path:
                out.write(frame)
            
            # Show the frame
            if show_video:
                cv2.imshow('Player Re-Identification', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        # Clean up
        cap.release()
        if output_path:
            out.release()
        if show_video:
            cv2.destroyAllWindows()

def main():
    # Example usage
    video_path = "input_video.mp4"  # Replace with your video path
    output_path = "output_video.mp4"
    
    # Initialize player re-identification system
    reid_system = PlayerReID()
    
    # Process video
    reid_system.process_video(video_path, output_path)

if __name__ == "__main__":
    main()
