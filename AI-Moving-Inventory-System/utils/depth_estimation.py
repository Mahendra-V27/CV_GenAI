"""
Depth Estimation and 3D Volume Calculation
Supports MiDaS, Depth Anything, and rule-based estimation
"""


import numpy as np
from typing import Dict, Optional, Tuple
import logging
import cv2


from .interfaces import (
   IDepthEstimator, Detection, BoundingBox, DepthEstimatorType
)
from .model_registry import ModelRegistry


logger = logging.getLogger(__name__)


# Try to import torch for MiDaS
try:
   import torch
   TORCH_AVAILABLE = True
except ImportError:
   TORCH_AVAILABLE = False




# Standard furniture dimensions (L x W x H in feet)
OBJECT_DIMENSIONS = {
   # Furniture
   "sofa": (7.0, 3.0, 3.0),
   "couch": (7.0, 3.0, 3.0),
   "chair": (2.0, 2.0, 3.5),
   "bed": (6.5, 5.0, 2.5),
   "dining table": (6.0, 3.5, 2.5),
   "table": (4.0, 2.5, 2.5),
   "desk": (5.0, 2.5, 2.5),
  
   # Electronics
   "tv": (4.0, 0.5, 2.5),
   "television": (4.0, 0.5, 2.5),
   "laptop": (1.0, 0.8, 0.1),
   "monitor": (2.0, 0.5, 1.5),
   "refrigerator": (3.0, 2.5, 6.0),
   "microwave": (1.5, 1.5, 1.0),
   "oven": (2.5, 2.5, 3.0),
  
   # Decor
   "potted plant": (1.5, 1.5, 3.0),
   "vase": (0.5, 0.5, 1.5),
   "clock": (1.0, 0.2, 1.0),
   "book": (0.8, 0.5, 0.2),
  
   # Storage
   "suitcase": (2.0, 1.5, 0.8),
   "backpack": (1.0, 0.8, 1.5),
}




class RuleBasedDepthEstimator(IDepthEstimator):
   """
   Rule-based depth estimation using object size and position.
   Uses known object dimensions and perspective geometry.
   """
  
   def __init__(
       self,
       camera_fov: float = 60.0,
       reference_height_pixels: int = 720
   ):
       """
       Initialize estimator.
      
       Args:
           camera_fov: Estimated camera field of view in degrees
           reference_height_pixels: Reference image height for calibration
       """
       self.camera_fov = camera_fov
       self.reference_height = reference_height_pixels
      
       # Calibrated focal length (approximate)
       self.focal_length = reference_height_pixels / (2 * np.tan(np.radians(camera_fov / 2)))
  
   def estimate_depth(self, frame: np.ndarray) -> np.ndarray:
       """
       Estimate depth map using simple heuristics.
       Returns relative depth based on vertical position.
       """
       height, width = frame.shape[:2]
      
       # Create gradient-based depth map (closer at bottom, farther at top)
       y_coords = np.arange(height).reshape(-1, 1)
       depth_map = (height - y_coords) / height * 10  # 0-10 meters range
       depth_map = np.broadcast_to(depth_map, (height, width)).astype(np.float32)
      
       return depth_map
  
   def get_object_depth(self, depth_map: np.ndarray, bbox: BoundingBox) -> float:
       """Get depth at object center from depth map."""
       cx, cy = bbox.center
      
       # Clamp to image bounds
       cy = min(max(0, cy), depth_map.shape[0] - 1)
       cx = min(max(0, cx), depth_map.shape[1] - 1)
      
       return float(depth_map[cy, cx])
  
   def estimate_3d_volume(self, detection: Detection, depth_map: np.ndarray) -> float:
       """
       Estimate 3D volume of detected object in cubic feet.
       Uses known object dimensions scaled by apparent size.
       """
       class_name = detection.class_name.lower()
      
       # Get standard dimensions
       if class_name in OBJECT_DIMENSIONS:
           length, width, height = OBJECT_DIMENSIONS[class_name]
       else:
           # Default small item
           length, width, height = 2.0, 1.5, 1.5
      
       # Calculate base volume
       base_volume = length * width * height
      
       # Adjust based on detected size relative to expected
       bbox = detection.bbox
       detected_area = bbox.area
      
       # Expected area at standard distance (approximate)
       expected_area = (length * 30) * (height * 30)  # Pixels per foot approximation
      
       # Scale factor
       scale = np.sqrt(detected_area / expected_area) if expected_area > 0 else 1.0
       scale = np.clip(scale, 0.5, 2.0)  # Reasonable bounds
      
       # Adjusted volume
       adjusted_volume = base_volume * (scale ** 2)  # Scale affects 2D projection
      
       return round(adjusted_volume, 1)




class MiDaSDepthEstimator(IDepthEstimator):
   """
   MiDaS-based monocular depth estimation.
   Uses Intel's MiDaS model for accurate depth prediction.
   """
  
   def __init__(self, model_type: str = "MiDaS_small"):
       """
       Initialize MiDaS estimator.
      
       Args:
           model_type: MiDaS model variant (MiDaS_small, DPT_Large, etc.)
       """
       self.model_type = model_type
       self.model = None
       self.transform = None
       self.device = None
       self._load_model()
  
   def _load_model(self):
       """Load MiDaS model."""
       if not TORCH_AVAILABLE:
           logger.warning("PyTorch not available, MiDaS will not work")
           return
      
       try:
           self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
          
           # Load MiDaS model from torch hub
           self.model = torch.hub.load("intel-isl/MiDaS", self.model_type)
           self.model.to(self.device)
           self.model.eval()
          
           # Load transforms
           midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
           if self.model_type in ["DPT_Large", "DPT_Hybrid"]:
               self.transform = midas_transforms.dpt_transform
           else:
               self.transform = midas_transforms.small_transform
          
           logger.info(f"MiDaS model loaded: {self.model_type}")
          
       except Exception as e:
           logger.error(f"Failed to load MiDaS: {e}")
           self.model = None
  
   def estimate_depth(self, frame: np.ndarray) -> np.ndarray:
       """Estimate depth map using MiDaS."""
       if self.model is None:
           # Fallback to simple gradient
           return RuleBasedDepthEstimator().estimate_depth(frame)
      
       try:
           # Convert BGR to RGB
           if len(frame.shape) == 3 and frame.shape[2] == 3:
               frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
           else:
               frame_rgb = frame
          
           # Transform and predict
           input_batch = self.transform(frame_rgb).to(self.device)
          
           with torch.no_grad():
               prediction = self.model(input_batch)
               prediction = torch.nn.functional.interpolate(
                   prediction.unsqueeze(1),
                   size=frame.shape[:2],
                   mode="bicubic",
                   align_corners=False
               ).squeeze()
          
           depth_map = prediction.cpu().numpy()
          
           # Normalize to meters (approximate)
           depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-6)
           depth_map = depth_map * 10  # 0-10 meters
          
           return depth_map.astype(np.float32)
          
       except Exception as e:
           logger.error(f"MiDaS prediction error: {e}")
           return RuleBasedDepthEstimator().estimate_depth(frame)
  
   def get_object_depth(self, depth_map: np.ndarray, bbox: BoundingBox) -> float:
       """Get median depth within bounding box."""
       x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2
      
       # Clamp to bounds
       y1 = max(0, y1)
       y2 = min(depth_map.shape[0], y2)
       x1 = max(0, x1)
       x2 = min(depth_map.shape[1], x2)
      
       if y2 <= y1 or x2 <= x1:
           return 5.0  # Default middle distance
      
       roi_depth = depth_map[y1:y2, x1:x2]
       return float(np.median(roi_depth))
  
   def estimate_3d_volume(self, detection: Detection, depth_map: np.ndarray) -> float:
       """Estimate 3D volume using depth information."""
       class_name = detection.class_name.lower()
       bbox = detection.bbox
      
       # Get depth
       depth = self.get_object_depth(depth_map, bbox)
      
       # Get standard dimensions
       if class_name in OBJECT_DIMENSIONS:
           length, width, height = OBJECT_DIMENSIONS[class_name]
       else:
           length, width, height = 2.0, 1.5, 1.5
      
       # Calculate scale based on depth
       # Objects closer appear larger, farther appear smaller
       reference_depth = 3.0  # meters
       scale = reference_depth / (depth + 0.5)
       scale = np.clip(scale, 0.5, 2.0)
      
       # Calculate volume with perspective correction
       pixel_height = bbox.height
       pixel_width = bbox.width
      
       # Estimate real-world dimensions
       real_height = height * scale
       real_width = (pixel_width / pixel_height) * real_height if pixel_height > 0 else width
       real_length = length * scale
      
       volume = real_length * real_width * real_height
       return round(volume, 1)




class DepthAnythingEstimator(IDepthEstimator):
   """
   Depth Anything V2 based depth estimation.
   State-of-the-art monocular depth estimation.
   """
  
   def __init__(self, model_size: str = "small"):
       """
       Initialize Depth Anything estimator.
      
       Args:
           model_size: Model size (small, base, large)
       """
       self.model_size = model_size
       self.model = None
       self.processor = None
       self._load_model()
  
   def _load_model(self):
       """Load Depth Anything model."""
       if not TORCH_AVAILABLE:
           logger.warning("PyTorch not available")
           return
      
       try:
           from transformers import pipeline
          
           model_id = f"depth-anything/Depth-Anything-V2-{self.model_size.capitalize()}-hf"
           self.pipe = pipeline(
               task="depth-estimation",
               model=model_id,
               device=0 if torch.cuda.is_available() else -1
           )
           logger.info(f"Depth Anything loaded: {model_id}")
          
       except Exception as e:
           logger.warning(f"Could not load Depth Anything: {e}")
           self.pipe = None
  
   def estimate_depth(self, frame: np.ndarray) -> np.ndarray:
       """Estimate depth using Depth Anything."""
       if self.pipe is None:
           return RuleBasedDepthEstimator().estimate_depth(frame)
      
       try:
           from PIL import Image
          
           # Convert to PIL Image
           if len(frame.shape) == 3:
               frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
           else:
               frame_rgb = frame
          
           image = Image.fromarray(frame_rgb)
          
           # Get depth prediction
           result = self.pipe(image)
           depth_map = np.array(result["depth"])
          
           # Resize to original size
           depth_map = cv2.resize(depth_map, (frame.shape[1], frame.shape[0]))
          
           # Normalize
           depth_map = depth_map.astype(np.float32)
           depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-6)
           depth_map = depth_map * 10
          
           return depth_map
          
       except Exception as e:
           logger.error(f"Depth Anything error: {e}")
           return RuleBasedDepthEstimator().estimate_depth(frame)
  
   def get_object_depth(self, depth_map: np.ndarray, bbox: BoundingBox) -> float:
       """Get object depth from depth map."""
       return MiDaSDepthEstimator.get_object_depth(self, depth_map, bbox)
  
   def estimate_3d_volume(self, detection: Detection, depth_map: np.ndarray) -> float:
       """Estimate 3D volume."""
       return MiDaSDepthEstimator.estimate_3d_volume(self, detection, depth_map)




def check_midas_available() -> bool:
   """Check if MiDaS is available."""
   if not TORCH_AVAILABLE:
       return False
   try:
       import torch
       return True
   except:
       return False




def check_depth_anything_available() -> bool:
   """Check if Depth Anything is available."""
   if not TORCH_AVAILABLE:
       return False
   try:
       from transformers import pipeline
       return True
   except:
       return False




# Register depth estimators
ModelRegistry.register_depth_estimator(
   DepthEstimatorType.ESTIMATED,
   "Rule-Based Estimation",
   "Fast rule-based depth estimation using object dimensions",
   RuleBasedDepthEstimator,
   {"camera_fov": 60.0}
)


ModelRegistry.register_depth_estimator(
   DepthEstimatorType.MIDAS,
   "MiDaS Depth",
   "Intel MiDaS monocular depth estimation",
   MiDaSDepthEstimator,
   {"model_type": "MiDaS_small"},
   check_midas_available
)


ModelRegistry.register_depth_estimator(
   DepthEstimatorType.DEPTH_ANYTHING,
   "Depth Anything V2",
   "State-of-the-art depth estimation",
   DepthAnythingEstimator,
   {"model_size": "small"},
   check_depth_anything_available
)