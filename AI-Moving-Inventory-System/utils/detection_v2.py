"""
Object Detection Module V2
Enhanced YOLOv8 detection with interface support and ONNX/TensorRT export
"""


import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import cv2
import logging
import os


from .interfaces import (
   IDetector, Detection, BoundingBox, ModelType
)
from .model_registry import ModelRegistry
from .detection import HOUSEHOLD_CLASSES, ITEM_CATEGORIES


logger = logging.getLogger(__name__)


# Try imports
try:
   from ultralytics import YOLO
   YOLO_AVAILABLE = True
except ImportError:
   YOLO_AVAILABLE = False
   YOLO = None


try:
   import onnxruntime as ort
   ONNX_AVAILABLE = True
except ImportError:
   ONNX_AVAILABLE = False
   ort = None




class YOLOv8Detector(IDetector):
   """
   YOLOv8-based object detector implementing IDetector interface.
   Supports multiple model sizes and ONNX export.
   """
  
   MODEL_PATHS = {
       ModelType.YOLOV8_NANO: "yolov8n.pt",
       ModelType.YOLOV8_SMALL: "yolov8s.pt",
       ModelType.YOLOV8_MEDIUM: "yolov8m.pt",
       ModelType.YOLOV8_LARGE: "yolov8l.pt",
       ModelType.YOLOV8_XLARGE: "yolov8x.pt",
   }
  
   def __init__(
       self,
       model_type: ModelType = ModelType.YOLOV8_NANO,
       confidence_threshold: float = 0.4,
       filter_household_only: bool = True,
       device: str = None
   ):
       """
       Initialize YOLOv8 detector.
      
       Args:
           model_type: YOLOv8 model variant
           confidence_threshold: Detection confidence threshold
           filter_household_only: Only detect household items
           device: Device to run on (cpu, cuda, mps). None for auto-detect.
       """
       if not YOLO_AVAILABLE:
           raise ImportError("ultralytics not installed. Run: pip install ultralytics")
      
       self.model_type = model_type
       self.confidence_threshold = confidence_threshold
       self.filter_household_only = filter_household_only
      
       # Auto-detect device
       if device is None:
           try:
               import torch
               if torch.cuda.is_available():
                   self.device = "cuda"
               elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                   self.device = "mps"
               else:
                   self.device = "cpu"
           except:
               self.device = "cpu"
       else:
           self.device = device
      
       # Load model
       model_path = self.MODEL_PATHS.get(model_type, "yolov8n.pt")
       self.model = YOLO(model_path)
      
       self.household_class_ids = set(HOUSEHOLD_CLASSES.keys())
      
       logger.info(f"Loaded YOLOv8 model: {model_path}")
  
   def detect(self, frame: np.ndarray, frame_index: int = 0) -> List[Detection]:
       """Detect objects in frame."""
       results = self.model(frame, verbose=False, device=self.device)[0]
      
       detections = []
      
       for box in results.boxes:
           class_id = int(box.cls[0])
           confidence = float(box.conf[0])
          
           if confidence < self.confidence_threshold:
               continue
          
           if self.filter_household_only and class_id not in self.household_class_ids:
               continue
          
           x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
          
           class_name = self.model.names[class_id]
          
           # Normalize names
           if class_name == "couch":
               class_name = "sofa"
           elif class_name == "dining table":
               class_name = "table"
           elif class_name in ["tv", "tvmonitor"]:
               class_name = "tv"
          
           detections.append(Detection(
               class_name=class_name,
               class_id=class_id,
               confidence=confidence,
               bbox=BoundingBox(x1, y1, x2, y2),
               frame_index=frame_index
           ))
      
       return detections
  
   def get_model_info(self) -> Dict[str, Any]:
       """Get model information."""
       return {
           "model_type": self.model_type.value,
           "framework": "ultralytics",
           "input_size": 640,
           "classes": len(self.model.names),
           "household_classes": len(self.household_class_ids),
           "device": self.device
       }
  
   def export_to_onnx(self, output_path: str) -> bool:
       """Export model to ONNX format."""
       try:
           self.model.export(format="onnx", imgsz=640)
          
           # YOLO exports to same directory with .onnx extension
           default_path = Path(self.MODEL_PATHS[self.model_type]).with_suffix(".onnx")
          
           if default_path.exists() and str(default_path) != output_path:
               import shutil
               shutil.move(str(default_path), output_path)
          
           logger.info(f"Exported ONNX model to: {output_path}")
           return True
          
       except Exception as e:
           logger.error(f"ONNX export failed: {e}")
           return False
  
   def export_to_tensorrt(self, output_path: str) -> bool:
       """Export model to TensorRT format."""
       try:
           self.model.export(format="engine", imgsz=640)
           logger.info(f"Exported TensorRT engine to: {output_path}")
           return True
       except Exception as e:
           logger.error(f"TensorRT export failed: {e}")
           return False
  
   def get_annotated_frame(
       self,
       frame: np.ndarray,
       detections: List[Detection],
       show_depth: bool = False,
       show_room: bool = False
   ) -> np.ndarray:
       """Draw detections on frame."""
       annotated = frame.copy()
      
       color_map = {
           "Furniture": (0, 255, 0),
           "Electronics": (255, 0, 0),
           "Kitchen Appliances": (0, 165, 255),
           "Decor": (255, 255, 0),
           "Storage": (255, 0, 255),
           "Other": (128, 128, 128)
       }
      
       for det in detections:
           bbox = det.bbox
           category = ITEM_CATEGORIES.get(det.class_name.lower(), "Other")
           color = color_map.get(category, (128, 128, 128))
          
           # Draw box
           cv2.rectangle(annotated, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), color, 2)
          
           # Build label
           label_parts = [f"{det.class_name}: {det.confidence:.2f}"]
          
           if det.track_id is not None:
               label_parts.append(f"ID:{det.track_id}")
          
           if show_depth and det.depth is not None:
               label_parts.append(f"D:{det.depth:.1f}m")
          
           if show_room and det.room:
               label_parts.append(det.room)
          
           label = " | ".join(label_parts)
          
           # Draw label background
           (label_w, label_h), _ = cv2.getTextSize(
               label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
           )
          
           cv2.rectangle(
               annotated,
               (bbox.x1, bbox.y1 - label_h - 10),
               (bbox.x1 + label_w + 5, bbox.y1),
               color,
               -1
           )
          
           cv2.putText(
               annotated,
               label,
               (bbox.x1 + 2, bbox.y1 - 5),
               cv2.FONT_HERSHEY_SIMPLEX,
               0.5,
               (255, 255, 255),
               1
           )
      
       return annotated




class ONNXDetector(IDetector):
   """
   ONNX Runtime-based detector for optimized inference.
   """
  
   def __init__(
       self,
       model_path: str,
       confidence_threshold: float = 0.4,
       filter_household_only: bool = True
   ):
       """
       Initialize ONNX detector.
      
       Args:
           model_path: Path to ONNX model file
           confidence_threshold: Detection confidence threshold
           filter_household_only: Only detect household items
       """
       if not ONNX_AVAILABLE:
           raise ImportError("onnxruntime not installed. Run: pip install onnxruntime")
      
       self.model_path = model_path
       self.confidence_threshold = confidence_threshold
       self.filter_household_only = filter_household_only
      
       # Create session
       providers = ['CPUExecutionProvider']
       if 'CUDAExecutionProvider' in ort.get_available_providers():
           providers.insert(0, 'CUDAExecutionProvider')
      
       self.session = ort.InferenceSession(model_path, providers=providers)
      
       # Get input/output names
       self.input_name = self.session.get_inputs()[0].name
       self.output_names = [o.name for o in self.session.get_outputs()]
      
       # COCO class names
       self.class_names = {i: name for i, name in enumerate(
           ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
            "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
            "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
            "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
       )}
      
       self.household_class_ids = set(HOUSEHOLD_CLASSES.keys())
      
       logger.info(f"Loaded ONNX model: {model_path}")
  
   def _preprocess(self, frame: np.ndarray) -> np.ndarray:
       """Preprocess frame for ONNX model."""
       # Resize to 640x640
       input_size = 640
       h, w = frame.shape[:2]
      
       # Calculate scale
       scale = min(input_size / h, input_size / w)
       new_h, new_w = int(h * scale), int(w * scale)
      
       # Resize
       resized = cv2.resize(frame, (new_w, new_h))
      
       # Pad to square
       padded = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
       padded[:new_h, :new_w] = resized
      
       # Convert to NCHW format
       blob = padded.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32) / 255.0
      
       return blob, scale, (h, w)
  
   def detect(self, frame: np.ndarray, frame_index: int = 0) -> List[Detection]:
       """Detect objects using ONNX model."""
       blob, scale, original_size = self._preprocess(frame)
      
       # Run inference
       outputs = self.session.run(self.output_names, {self.input_name: blob})
      
       # Parse outputs (assuming YOLOv8 output format)
       predictions = outputs[0][0]  # [num_detections, 84] for COCO
      
       detections = []
      
       for pred in predictions.T:  # Transpose to iterate over detections
           # Get class scores
           class_scores = pred[4:]
           class_id = np.argmax(class_scores)
           confidence = float(class_scores[class_id])
          
           if confidence < self.confidence_threshold:
               continue
          
           if self.filter_household_only and class_id not in self.household_class_ids:
               continue
          
           # Get bounding box (center format)
           cx, cy, w, h = pred[:4]
          
           # Scale back to original image
           x1 = int((cx - w / 2) / scale)
           y1 = int((cy - h / 2) / scale)
           x2 = int((cx + w / 2) / scale)
           y2 = int((cy + h / 2) / scale)
          
           # Clamp to image bounds
           x1 = max(0, min(x1, original_size[1]))
           y1 = max(0, min(y1, original_size[0]))
           x2 = max(0, min(x2, original_size[1]))
           y2 = max(0, min(y2, original_size[0]))
          
           class_name = self.class_names.get(class_id, f"class_{class_id}")
          
           # Normalize names
           if class_name == "couch":
               class_name = "sofa"
           elif class_name == "dining table":
               class_name = "table"
          
           detections.append(Detection(
               class_name=class_name,
               class_id=class_id,
               confidence=confidence,
               bbox=BoundingBox(x1, y1, x2, y2),
               frame_index=frame_index
           ))
      
       return detections
  
   def get_model_info(self) -> Dict[str, Any]:
       """Get ONNX model information."""
       return {
           "model_type": "onnx",
           "model_path": self.model_path,
           "providers": self.session.get_providers(),
           "input_name": self.input_name,
           "output_names": self.output_names
       }
  
   def export_to_onnx(self, output_path: str) -> bool:
       """Already ONNX, copy file."""
       import shutil
       try:
           shutil.copy(self.model_path, output_path)
           return True
       except Exception as e:
           logger.error(f"Copy failed: {e}")
           return False




def check_yolo_available() -> bool:
   """Check if YOLO is available."""
   return YOLO_AVAILABLE




def check_onnx_available() -> bool:
   """Check if ONNX runtime is available."""
   return ONNX_AVAILABLE




# Register detectors
for model_type in [ModelType.YOLOV8_NANO, ModelType.YOLOV8_SMALL,
                  ModelType.YOLOV8_MEDIUM, ModelType.YOLOV8_LARGE, ModelType.YOLOV8_XLARGE]:
   ModelRegistry.register_detector(
       model_type,
       f"YOLOv8 {model_type.value}",
       f"YOLOv8 {model_type.value.replace('yolov8', '').upper()} variant",
       YOLOv8Detector,
       {"model_type": model_type, "confidence_threshold": 0.4},
       check_yolo_available
   )


ModelRegistry.register_detector(
   ModelType.ONNX,
   "ONNX Runtime",
   "Optimized ONNX inference",
   ONNXDetector,
   {"model_path": "yolov8n.onnx", "confidence_threshold": 0.4},
   check_onnx_available
)