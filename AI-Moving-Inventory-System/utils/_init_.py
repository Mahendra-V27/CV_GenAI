"""
Utility modules for AI-Powered Moving Inventory Generation System V2
Enhanced with interfaces, advanced tracking, depth estimation, and room classification
"""


# Core modules
from .video_processing import VideoProcessor
from .detection import ObjectDetector
from .inventory import InventoryManager
from .genai_summary import SummaryGenerator


# V2 Interface-based architecture
from .interfaces import (
   IDetector, ITracker, IDepthEstimator, IRoomClassifier, ISummaryGenerator,
   ModelType, TrackerType, DepthEstimatorType, RoomClassifierType,
   Detection, BoundingBox, TrackedObject, FrameResult, ProcessingConfig
)


# Model registry and factory
from .model_registry import ModelRegistry, ComponentFactory, DEFAULT_CONFIG


# Advanced detection with ONNX support
from .detection_v2 import YOLOv8Detector, ONNXDetector


# Advanced trackers
from .trackers import SimpleTracker, DeepSORTTracker, ByteTracker


# Depth estimation
from .depth_estimation import RuleBasedDepthEstimator, MiDaSDepthEstimator


# Room classification
from .room_classifier import RuleBasedRoomClassifier, CLIPRoomClassifier


__all__ = [
   # Core
   "VideoProcessor",
   "ObjectDetector",
   "InventoryManager",
   "SummaryGenerator",
  
   # Interfaces
   "IDetector", "ITracker", "IDepthEstimator", "IRoomClassifier", "ISummaryGenerator",
   "ModelType", "TrackerType", "DepthEstimatorType", "RoomClassifierType",
   "Detection", "BoundingBox", "TrackedObject", "FrameResult", "ProcessingConfig",
  
   # Registry
   "ModelRegistry", "ComponentFactory", "DEFAULT_CONFIG",
  
   # Detectors
   "YOLOv8Detector", "ONNXDetector",
  
   # Trackers
   "SimpleTracker", "DeepSORTTracker", "ByteTracker",
  
   # Depth
   "RuleBasedDepthEstimator", "MiDaSDepthEstimator",
  
   # Room
   "RuleBasedRoomClassifier", "CLIPRoomClassifier",
]
