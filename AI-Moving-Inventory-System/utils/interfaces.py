"""
Abstract Interfaces for AI Moving Inventory System
Provides loosely coupled architecture with dependency injection
"""


from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np




class ModelType(Enum):
   """Supported model types for detection."""
   YOLOV8_NANO = "yolov8n"
   YOLOV8_SMALL = "yolov8s"
   YOLOV8_MEDIUM = "yolov8m"
   YOLOV8_LARGE = "yolov8l"
   YOLOV8_XLARGE = "yolov8x"
   ONNX = "onnx"
   TENSORRT = "tensorrt"




class TrackerType(Enum):
   """Supported tracker types."""
   NONE = "none"
   SIMPLE = "simple"
   DEEPSORT = "deepsort"
   BYTETRACK = "bytetrack"




class DepthEstimatorType(Enum):
   """Supported depth estimation methods."""
   NONE = "none"
   MIDAS = "midas"
   DEPTH_ANYTHING = "depth_anything"
   ESTIMATED = "estimated"  # Rule-based estimation




class RoomClassifierType(Enum):
   """Supported room classification methods."""
   NONE = "none"
   RULE_BASED = "rule_based"
   CNN = "cnn"
   CLIP = "clip"




class SummaryGeneratorType(Enum):
   """Supported summary generation methods."""
   TEMPLATE = "template"
   OLLAMA = "ollama"
   TRANSFORMERS = "transformers"
   OPENAI = "openai"




@dataclass
class BoundingBox:
   """Bounding box representation."""
   x1: int
   y1: int
   x2: int
   y2: int
  
   @property
   def center(self) -> Tuple[int, int]:
       return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)
  
   @property
   def width(self) -> int:
       return self.x2 - self.x1
  
   @property
   def height(self) -> int:
       return self.y2 - self.y1
  
   @property
   def area(self) -> int:
       return self.width * self.height
  
   def to_tuple(self) -> Tuple[int, int, int, int]:
       return (self.x1, self.y1, self.x2, self.y2)




@dataclass
class Detection:
   """Single detection result."""
   class_name: str
   class_id: int
   confidence: float
   bbox: BoundingBox
   frame_index: int
   track_id: Optional[int] = None
   depth: Optional[float] = None  # Estimated depth in meters
   volume_3d: Optional[float] = None  # Estimated 3D volume in cubic feet
   room: Optional[str] = None
  
   def to_dict(self) -> Dict:
       return {
           "class_name": self.class_name,
           "class_id": self.class_id,
           "confidence": self.confidence,
           "bbox": self.bbox.to_tuple(),
           "center": self.bbox.center,
           "frame_index": self.frame_index,
           "track_id": self.track_id,
           "depth": self.depth,
           "volume_3d": self.volume_3d,
           "room": self.room
       }




@dataclass
class TrackedObject:
   """Object tracked across multiple frames."""
   track_id: int
   class_name: str
   detections: List[Detection] = field(default_factory=list)
   is_active: bool = True
  
   @property
   def best_detection(self) -> Optional[Detection]:
       if not self.detections:
           return None
       return max(self.detections, key=lambda d: d.confidence)
  
   @property
   def avg_confidence(self) -> float:
       if not self.detections:
           return 0.0
       return sum(d.confidence for d in self.detections) / len(self.detections)




@dataclass
class FrameResult:
   """Result of processing a single frame."""
   frame_index: int
   detections: List[Detection]
   annotated_frame: Optional[np.ndarray] = None
   depth_map: Optional[np.ndarray] = None
   room_label: Optional[str] = None




@dataclass
class ProcessingConfig:
   """Configuration for the processing pipeline."""
   # Detection settings
   model_type: ModelType = ModelType.YOLOV8_NANO
   confidence_threshold: float = 0.4
   filter_household_only: bool = True
  
   # Tracking settings
   tracker_type: TrackerType = TrackerType.SIMPLE
  
   # Depth estimation
   depth_estimator_type: DepthEstimatorType = DepthEstimatorType.ESTIMATED
  
   # Room classification
   room_classifier_type: RoomClassifierType = RoomClassifierType.RULE_BASED
  
   # Summary generation
   summary_generator_type: SummaryGeneratorType = SummaryGeneratorType.TEMPLATE
  
   # Video processing
   extraction_rate: str = "1_per_second"
   max_frames: int = 100
  
   # Output settings
   save_annotated_frames: bool = True
   preview_frame_count: int = 5




# Abstract Interfaces


class IDetector(ABC):
   """Abstract interface for object detection."""
  
   @abstractmethod
   def detect(self, frame: np.ndarray, frame_index: int = 0) -> List[Detection]:
       """Detect objects in a frame."""
       pass
  
   @abstractmethod
   def get_model_info(self) -> Dict[str, Any]:
       """Get information about the loaded model."""
       pass
  
   @abstractmethod
   def export_to_onnx(self, output_path: str) -> bool:
       """Export model to ONNX format."""
       pass




class ITracker(ABC):
   """Abstract interface for object tracking."""
  
   @abstractmethod
   def update(self, detections: List[Detection], frame: np.ndarray) -> List[Detection]:
       """Update tracker with new detections, returns detections with track IDs."""
       pass
  
   @abstractmethod
   def get_tracks(self) -> List[TrackedObject]:
       """Get all tracked objects."""
       pass
  
   @abstractmethod
   def reset(self):
       """Reset tracker state."""
       pass




class IDepthEstimator(ABC):
   """Abstract interface for depth estimation."""
  
   @abstractmethod
   def estimate_depth(self, frame: np.ndarray) -> np.ndarray:
       """Estimate depth map for a frame."""
       pass
  
   @abstractmethod
   def get_object_depth(self, depth_map: np.ndarray, bbox: BoundingBox) -> float:
       """Get estimated depth for an object."""
       pass
  
   @abstractmethod
   def estimate_3d_volume(self, detection: Detection, depth_map: np.ndarray) -> float:
       """Estimate 3D volume of detected object in cubic feet."""
       pass




class IRoomClassifier(ABC):
   """Abstract interface for room classification."""
  
   @abstractmethod
   def classify_room(self, frame: np.ndarray, detections: List[Detection]) -> str:
       """Classify the room type based on frame and detections."""
       pass
  
   @abstractmethod
   def get_room_confidence(self) -> float:
       """Get confidence of last classification."""
       pass




class ISummaryGenerator(ABC):
   """Abstract interface for summary generation."""
  
   @abstractmethod
   def generate_summary(
       self,
       inventory: Dict,
       volume_estimate: Dict,
       stats: Dict,
       room_breakdown: Optional[Dict] = None
   ) -> str:
       """Generate natural language summary."""
       pass
  
   @abstractmethod
   def is_available(self) -> bool:
       """Check if the generator is available."""
       pass




class IVideoProcessor(ABC):
   """Abstract interface for video processing."""
  
   @abstractmethod
   def extract_frames(
       self,
       video_path: str,
       config: ProcessingConfig
   ) -> Tuple[List[np.ndarray], List[str]]:
       """Extract frames from video."""
       pass
  
   @abstractmethod
   def get_video_info(self, video_path: str) -> Dict:
       """Get video metadata."""
       pass




class IInventoryManager(ABC):
   """Abstract interface for inventory management."""
  
   @abstractmethod
   def process_detections(
       self,
       frame_results: List[FrameResult]
   ) -> Tuple[Dict, Dict]:
       """Process detections into inventory."""
       pass
  
   @abstractmethod
   def estimate_volume(self, inventory: Dict) -> Dict:
       """Estimate moving volume."""
       pass
  
   @abstractmethod
   def get_room_breakdown(self, frame_results: List[FrameResult]) -> Dict:
       """Get inventory breakdown by room."""
       pass

