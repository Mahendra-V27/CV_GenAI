"""
Object Detection Module
YOLOv8-based household item detection
"""


import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from PIL import Image
import cv2


try:
   from ultralytics import YOLO
except ImportError:
   YOLO = None




# Household items from COCO dataset that are relevant for moving inventory
HOUSEHOLD_CLASSES = {
   # Furniture
   56: "chair",
   57: "couch",  # sofa
   58: "potted plant",
   59: "bed",
   60: "dining table",
  
   # Electronics
   62: "tv",
   63: "laptop",
   64: "mouse",
   65: "remote",
   66: "keyboard",
   67: "cell phone",
  
   # Kitchen Appliances
   68: "microwave",
   69: "oven",
   70: "toaster",
   71: "sink",
   72: "refrigerator",
  
   # Books and decor
   73: "book",
   74: "clock",
   75: "vase",
   76: "scissors",
   77: "teddy bear",
   78: "hair drier",
   79: "toothbrush",
  
   # Storage
   24: "backpack",
   25: "umbrella",
   26: "handbag",
   28: "suitcase",
  
   # Additional useful items
   39: "bottle",
   41: "cup",
   42: "fork",
   43: "knife",
   44: "spoon",
   45: "bowl",
}


# Category mapping for better organization
ITEM_CATEGORIES = {
   "chair": "Furniture",
   "couch": "Furniture",
   "sofa": "Furniture",
   "bed": "Furniture",
   "dining table": "Furniture",
   "table": "Furniture",
   "desk": "Furniture",
  
   "tv": "Electronics",
   "television": "Electronics",
   "laptop": "Electronics",
   "keyboard": "Electronics",
   "mouse": "Electronics",
   "remote": "Electronics",
   "cell phone": "Electronics",
  
   "microwave": "Kitchen Appliances",
   "oven": "Kitchen Appliances",
   "toaster": "Kitchen Appliances",
   "sink": "Kitchen Appliances",
   "refrigerator": "Kitchen Appliances",
  
   "potted plant": "Decor",
   "clock": "Decor",
   "vase": "Decor",
   "book": "Decor",
   "teddy bear": "Decor",
  
   "backpack": "Storage",
   "suitcase": "Storage",
   "handbag": "Storage",
   "umbrella": "Storage",
  
   "bottle": "Kitchen Items",
   "cup": "Kitchen Items",
   "bowl": "Kitchen Items",
   "fork": "Kitchen Items",
   "knife": "Kitchen Items",
   "spoon": "Kitchen Items",
}




class DetectedObject:
   """Represents a detected object with its properties."""
  
   def __init__(
       self,
       class_name: str,
       confidence: float,
       bbox: Tuple[int, int, int, int],  # x1, y1, x2, y2
       frame_index: int,
       center: Optional[Tuple[int, int]] = None
   ):
       self.class_name = class_name
       self.confidence = confidence
       self.bbox = bbox
       self.frame_index = frame_index
       self.center = center or self._calculate_center()
       self.category = ITEM_CATEGORIES.get(class_name.lower(), "Other")
      
   def _calculate_center(self) -> Tuple[int, int]:
       x1, y1, x2, y2 = self.bbox
       return ((x1 + x2) // 2, (y1 + y2) // 2)
  
   @property
   def area(self) -> int:
       x1, y1, x2, y2 = self.bbox
       return (x2 - x1) * (y2 - y1)
  
   def to_dict(self) -> dict:
       return {
           "class_name": self.class_name,
           "confidence": self.confidence,
           "bbox": self.bbox,
           "center": self.center,
           "frame_index": self.frame_index,
           "category": self.category,
           "area": self.area
       }




class ObjectDetector:
   """
   YOLOv8-based object detector for household items.
   """
  
   def __init__(
       self,
       model_path: str = "yolov8n.pt",
       confidence_threshold: float = 0.4,
       filter_household_only: bool = True
   ):
       """
       Initialize the object detector.
      
       Args:
           model_path: Path to YOLO model (will download if not exists)
           confidence_threshold: Minimum confidence for detections
           filter_household_only: Only detect household items
       """
       if YOLO is None:
           raise ImportError("ultralytics package not installed. Run: pip install ultralytics")
          
       self.model = YOLO(model_path)
       self.confidence_threshold = confidence_threshold
       self.filter_household_only = filter_household_only
       self.household_class_ids = set(HOUSEHOLD_CLASSES.keys())
      
   def detect_objects(
       self,
       image: np.ndarray,
       frame_index: int = 0
   ) -> List[DetectedObject]:
       """
       Detect objects in a single image/frame.
      
       Args:
           image: Image as numpy array (RGB or BGR)
           frame_index: Index of the frame in the video
          
       Returns:
           List of DetectedObject instances
       """
       # Run detection
       results = self.model(image, verbose=False)[0]
      
       detected_objects = []
      
       for box in results.boxes:
           class_id = int(box.cls[0])
           confidence = float(box.conf[0])
          
           # Filter by confidence
           if confidence < self.confidence_threshold:
               continue
              
           # Filter household items only if enabled
           if self.filter_household_only and class_id not in self.household_class_ids:
               continue
              
           # Get bounding box
           x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
          
           # Get class name
           class_name = self.model.names[class_id]
          
           # Normalize some class names
           if class_name == "couch":
               class_name = "sofa"
           elif class_name == "dining table":
               class_name = "table"
           elif class_name == "tv" or class_name == "tvmonitor":
               class_name = "tv"
              
           detected_objects.append(DetectedObject(
               class_name=class_name,
               confidence=confidence,
               bbox=(x1, y1, x2, y2),
               frame_index=frame_index
           ))
          
       return detected_objects
  
   def detect_in_frames(
       self,
       frames: List[np.ndarray],
       progress_callback=None
   ) -> List[List[DetectedObject]]:
       """
       Detect objects in multiple frames.
      
       Args:
           frames: List of frames as numpy arrays
           progress_callback: Optional callback for progress updates
          
       Returns:
           List of detection lists, one per frame
       """
       all_detections = []
      
       for idx, frame in enumerate(frames):
           detections = self.detect_objects(frame, frame_index=idx)
           all_detections.append(detections)
          
           if progress_callback:
               progress_callback(idx + 1, len(frames))
              
       return all_detections
  
   def detect_from_path(self, image_path: str, frame_index: int = 0) -> List[DetectedObject]:
       """
       Detect objects from an image file path.
      
       Args:
           image_path: Path to image file
           frame_index: Index for tracking
          
       Returns:
           List of DetectedObject instances
       """
       image = cv2.imread(image_path)
       if image is None:
           raise ValueError(f"Cannot read image: {image_path}")
       return self.detect_objects(image, frame_index)
  
   def get_annotated_frame(
       self,
       image: np.ndarray,
       detections: List[DetectedObject]
   ) -> np.ndarray:
       """
       Draw bounding boxes and labels on the image.
      
       Args:
           image: Original image
           detections: List of detected objects
          
       Returns:
           Annotated image
       """
       annotated = image.copy()
      
       for det in detections:
           x1, y1, x2, y2 = det.bbox
          
           # Color based on category
           color_map = {
               "Furniture": (0, 255, 0),      # Green
               "Electronics": (255, 0, 0),    # Blue
               "Kitchen Appliances": (0, 165, 255),  # Orange
               "Decor": (255, 255, 0),        # Cyan
               "Storage": (255, 0, 255),      # Magenta
               "Other": (128, 128, 128)       # Gray
           }
           color = color_map.get(det.category, (128, 128, 128))
          
           # Draw bounding box
           cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
          
           # Draw label with background
           label = f"{det.class_name}: {det.confidence:.2f}"
           (label_w, label_h), _ = cv2.getTextSize(
               label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
           )
          
           cv2.rectangle(
               annotated,
               (x1, y1 - label_h - 10),
               (x1 + label_w, y1),
               color,
               -1
           )
          
           cv2.putText(
               annotated,
               label,
               (x1, y1 - 5),
               cv2.FONT_HERSHEY_SIMPLEX,
               0.5,
               (255, 255, 255),
               1
           )
          
       return annotated
  
   def get_all_class_names(self) -> List[str]:
       """Get list of all detectable class names."""
       return list(self.model.names.values())
  
   def get_household_class_names(self) -> List[str]:
       """Get list of household item class names."""
       return list(HOUSEHOLD_CLASSES.values())




def run_detection_test():
   """Test the detector with a sample image."""
   detector = ObjectDetector(
       model_path="yolov8n.pt",
       confidence_threshold=0.3,
       filter_household_only=False  # Show all for testing
   )
  
   # Create a test image
   test_image = np.zeros((480, 640, 3), dtype=np.uint8)
   test_image[:] = (200, 200, 200)  # Gray background
  
   # Detect (will be empty on blank image)
   detections = detector.detect_objects(test_image)
   print(f"Detections on test image: {len(detections)}")
  
   # List available classes
   print(f"\nHousehold classes: {detector.get_household_class_names()}")




if __name__ == "__main__":
   run_detection_test()