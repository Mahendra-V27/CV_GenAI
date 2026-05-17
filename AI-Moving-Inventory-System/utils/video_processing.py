"""
Video Processing Module
Handles video loading and frame extraction using OpenCV
"""


import cv2
import os
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image




class VideoProcessor:
   """
   Extracts frames from video files for object detection.
   Supports MP4, MOV, AVI formats.
   """
  
   SUPPORTED_FORMATS = ['.mp4', '.mov', '.avi', '.mkv', '.webm']
  
   def __init__(self, output_dir: str = "extracted_frames"):
       """
       Initialize VideoProcessor.
      
       Args:
           output_dir: Directory to save extracted frames
       """
       self.output_dir = Path(output_dir)
       self.output_dir.mkdir(parents=True, exist_ok=True)
      
   def validate_video(self, video_path: str) -> Tuple[bool, str]:
       """
       Validate if the video file is supported and readable.
      
       Args:
           video_path: Path to video file
          
       Returns:
           Tuple of (is_valid, message)
       """
       path = Path(video_path)
      
       if not path.exists():
           return False, f"Video file not found: {video_path}"
          
       if path.suffix.lower() not in self.SUPPORTED_FORMATS:
           return False, f"Unsupported format. Supported: {self.SUPPORTED_FORMATS}"
          
       # Try to open with OpenCV
       cap = cv2.VideoCapture(str(video_path))
       if not cap.isOpened():
           return False, "Cannot open video file"
          
       ret, _ = cap.read()
       cap.release()
      
       if not ret:
           return False, "Cannot read video frames"
          
       return True, "Video is valid"
  
   def get_video_info(self, video_path: str) -> dict:
       """
       Get video metadata.
      
       Args:
           video_path: Path to video file
          
       Returns:
           Dictionary with video information
       """
       cap = cv2.VideoCapture(str(video_path))
      
       info = {
           "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
           "fps": cap.get(cv2.CAP_PROP_FPS),
           "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
           "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
           "duration_seconds": 0
       }
      
       if info["fps"] > 0:
           info["duration_seconds"] = info["total_frames"] / info["fps"]
          
       cap.release()
       return info
  
   def extract_frames(
       self,
       video_path: str,
       extraction_rate: str = "1_per_second",
       max_frames: int = 100,
       session_id: Optional[str] = None
   ) -> Tuple[List[np.ndarray], List[str]]:
       """
       Extract frames from video at specified rate.
      
       Args:
           video_path: Path to video file
           extraction_rate: One of "1_per_second", "2_per_second", "every_10_frames", "every_30_frames"
           max_frames: Maximum number of frames to extract
           session_id: Unique session ID for organizing frames
          
       Returns:
           Tuple of (list of frame arrays, list of saved frame paths)
       """
       cap = cv2.VideoCapture(str(video_path))
      
       if not cap.isOpened():
           raise ValueError(f"Cannot open video: {video_path}")
      
       fps = cap.get(cv2.CAP_PROP_FPS)
       total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
      
       # Calculate frame interval based on extraction rate
       if extraction_rate == "1_per_second":
           frame_interval = max(1, int(fps))
       elif extraction_rate == "2_per_second":
           frame_interval = max(1, int(fps / 2))
       elif extraction_rate == "every_10_frames":
           frame_interval = 10
       elif extraction_rate == "every_30_frames":
           frame_interval = 30
       else:
           frame_interval = max(1, int(fps))  # Default: 1 per second
          
       # Create session directory
       if session_id:
           session_dir = self.output_dir / session_id
       else:
           session_dir = self.output_dir / "default"
       session_dir.mkdir(parents=True, exist_ok=True)
      
       frames = []
       frame_paths = []
       frame_count = 0
       extracted_count = 0
      
       while True:
           ret, frame = cap.read()
           if not ret:
               break
              
           # Extract frame at specified interval
           if frame_count % frame_interval == 0:
               # Convert BGR to RGB
               frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
               frames.append(frame_rgb)
              
               # Save frame
               frame_path = session_dir / f"frame_{extracted_count:04d}.jpg"
               cv2.imwrite(str(frame_path), frame)
               frame_paths.append(str(frame_path))
              
               extracted_count += 1
              
               if extracted_count >= max_frames:
                   break
                  
           frame_count += 1
          
       cap.release()
      
       return frames, frame_paths
  
   def extract_frames_as_pil(
       self,
       video_path: str,
       extraction_rate: str = "1_per_second",
       max_frames: int = 100
   ) -> List[Image.Image]:
       """
       Extract frames and return as PIL Images.
      
       Args:
           video_path: Path to video file
           extraction_rate: Frame extraction rate
           max_frames: Maximum frames to extract
          
       Returns:
           List of PIL Image objects
       """
       frames_np, _ = self.extract_frames(video_path, extraction_rate, max_frames)
      
       pil_frames = []
       for frame in frames_np:
           pil_frames.append(Image.fromarray(frame))
          
       return pil_frames
  
   def cleanup_frames(self, session_id: Optional[str] = None):
       """
       Clean up extracted frames.
      
       Args:
           session_id: Session ID to clean up, or None for all
       """
       if session_id:
           session_dir = self.output_dir / session_id
           if session_dir.exists():
               for f in session_dir.glob("*.jpg"):
                   f.unlink()
               session_dir.rmdir()
       else:
           # Clean all
           for session_dir in self.output_dir.iterdir():
               if session_dir.is_dir():
                   for f in session_dir.glob("*.jpg"):
                       f.unlink()
                   session_dir.rmdir()




def create_sample_video(output_path: str, duration: int = 5):
   """
   Create a simple test video with shapes (for testing when no video available).
  
   Args:
       output_path: Path to save the video
       duration: Duration in seconds
   """
   fps = 30
   width, height = 640, 480
  
   fourcc = cv2.VideoWriter_fourcc(*'mp4v')
   out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
  
   total_frames = duration * fps
  
   for i in range(total_frames):
       # Create frame with colored background
       frame = np.zeros((height, width, 3), dtype=np.uint8)
      
       # Add some rectangles to simulate furniture
       # Simulate a "sofa"
       cv2.rectangle(frame, (50, 300), (250, 400), (139, 69, 19), -1)
      
       # Simulate a "TV"
       cv2.rectangle(frame, (400, 150), (600, 300), (50, 50, 50), -1)
      
       # Simulate "chairs"
       cv2.rectangle(frame, (300, 350), (350, 420), (165, 42, 42), -1)
       cv2.rectangle(frame, (360, 350), (410, 420), (165, 42, 42), -1)
      
       # Add frame number text
       cv2.putText(frame, f"Frame: {i}", (10, 30),
                  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
      
       out.write(frame)
  
   out.release()
   print(f"Sample video created: {output_path}")




if __name__ == "__main__":
   # Test the video processor
   processor = VideoProcessor()
  
   # Create a sample video for testing
   test_video = "test_sample.mp4"
   create_sample_video(test_video)
  
   # Validate and extract
   is_valid, msg = processor.validate_video(test_video)
   print(f"Validation: {is_valid}, {msg}")
  
   if is_valid:
       info = processor.get_video_info(test_video)
       print(f"Video info: {info}")
      
       frames, paths = processor.extract_frames(test_video, "1_per_second", max_frames=10)
       print(f"Extracted {len(frames)} frames")