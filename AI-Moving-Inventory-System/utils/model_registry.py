"""
Model Registry and Factory
Auto-routing and dependency injection for all components
"""


from typing import Dict, Type, Optional, Any, Callable
from dataclasses import dataclass
import logging


from .interfaces import (
   IDetector, ITracker, IDepthEstimator, IRoomClassifier,
   ISummaryGenerator, IVideoProcessor, IInventoryManager,
   ModelType, TrackerType, DepthEstimatorType,
   RoomClassifierType, SummaryGeneratorType, ProcessingConfig
)


logger = logging.getLogger(__name__)




@dataclass
class ModelInfo:
   """Information about a registered model."""
   name: str
   description: str
   model_class: Type
   default_config: Dict[str, Any]
   is_available: Callable[[], bool]




class ModelRegistry:
   """
   Central registry for all model implementations.
   Provides auto-routing based on configuration.
   """
  
   _detectors: Dict[ModelType, ModelInfo] = {}
   _trackers: Dict[TrackerType, ModelInfo] = {}
   _depth_estimators: Dict[DepthEstimatorType, ModelInfo] = {}
   _room_classifiers: Dict[RoomClassifierType, ModelInfo] = {}
   _summary_generators: Dict[SummaryGeneratorType, ModelInfo] = {}
  
   @classmethod
   def register_detector(
       cls,
       model_type: ModelType,
       name: str,
       description: str,
       model_class: Type[IDetector],
       default_config: Dict[str, Any] = None,
       availability_check: Callable[[], bool] = lambda: True
   ):
       """Register a detector implementation."""
       cls._detectors[model_type] = ModelInfo(
           name=name,
           description=description,
           model_class=model_class,
           default_config=default_config or {},
           is_available=availability_check
       )
  
   @classmethod
   def register_tracker(
       cls,
       tracker_type: TrackerType,
       name: str,
       description: str,
       model_class: Type[ITracker],
       default_config: Dict[str, Any] = None,
       availability_check: Callable[[], bool] = lambda: True
   ):
       """Register a tracker implementation."""
       cls._trackers[tracker_type] = ModelInfo(
           name=name,
           description=description,
           model_class=model_class,
           default_config=default_config or {},
           is_available=availability_check
       )
  
   @classmethod
   def register_depth_estimator(
       cls,
       estimator_type: DepthEstimatorType,
       name: str,
       description: str,
       model_class: Type[IDepthEstimator],
       default_config: Dict[str, Any] = None,
       availability_check: Callable[[], bool] = lambda: True
   ):
       """Register a depth estimator implementation."""
       cls._depth_estimators[estimator_type] = ModelInfo(
           name=name,
           description=description,
           model_class=model_class,
           default_config=default_config or {},
           is_available=availability_check
       )
  
   @classmethod
   def register_room_classifier(
       cls,
       classifier_type: RoomClassifierType,
       name: str,
       description: str,
       model_class: Type[IRoomClassifier],
       default_config: Dict[str, Any] = None,
       availability_check: Callable[[], bool] = lambda: True
   ):
       """Register a room classifier implementation."""
       cls._room_classifiers[classifier_type] = ModelInfo(
           name=name,
           description=description,
           model_class=model_class,
           default_config=default_config or {},
           is_available=availability_check
       )
  
   @classmethod
   def register_summary_generator(
       cls,
       generator_type: SummaryGeneratorType,
       name: str,
       description: str,
       model_class: Type[ISummaryGenerator],
       default_config: Dict[str, Any] = None,
       availability_check: Callable[[], bool] = lambda: True
   ):
       """Register a summary generator implementation."""
       cls._summary_generators[generator_type] = ModelInfo(
           name=name,
           description=description,
           model_class=model_class,
           default_config=default_config or {},
           is_available=availability_check
       )
  
   @classmethod
   def get_available_detectors(cls) -> Dict[ModelType, ModelInfo]:
       """Get all available detector implementations."""
       return {k: v for k, v in cls._detectors.items() if v.is_available()}
  
   @classmethod
   def get_available_trackers(cls) -> Dict[TrackerType, ModelInfo]:
       """Get all available tracker implementations."""
       return {k: v for k, v in cls._trackers.items() if v.is_available()}
  
   @classmethod
   def get_available_depth_estimators(cls) -> Dict[DepthEstimatorType, ModelInfo]:
       """Get all available depth estimator implementations."""
       return {k: v for k, v in cls._depth_estimators.items() if v.is_available()}
  
   @classmethod
   def get_available_room_classifiers(cls) -> Dict[RoomClassifierType, ModelInfo]:
       """Get all available room classifier implementations."""
       return {k: v for k, v in cls._room_classifiers.items() if v.is_available()}
  
   @classmethod
   def get_available_summary_generators(cls) -> Dict[SummaryGeneratorType, ModelInfo]:
       """Get all available summary generator implementations."""
       return {k: v for k, v in cls._summary_generators.items() if v.is_available()}




class ComponentFactory:
   """
   Factory for creating component instances based on configuration.
   Implements auto-routing with fallback logic.
   """
  
   def __init__(self, config: ProcessingConfig):
       self.config = config
       self._cache: Dict[str, Any] = {}
  
   def get_detector(self, **kwargs) -> IDetector:
       """Get or create detector instance."""
       cache_key = f"detector_{self.config.model_type.value}"
      
       if cache_key not in self._cache:
           available = ModelRegistry.get_available_detectors()
          
           # Try requested model type
           if self.config.model_type in available:
               model_info = available[self.config.model_type]
               merged_config = {**model_info.default_config, **kwargs}
               self._cache[cache_key] = model_info.model_class(**merged_config)
           else:
               # Fallback to first available
               if available:
                   first_type = list(available.keys())[0]
                   model_info = available[first_type]
                   logger.warning(
                       f"Requested detector {self.config.model_type} not available, "
                       f"falling back to {first_type}"
                   )
                   merged_config = {**model_info.default_config, **kwargs}
                   self._cache[cache_key] = model_info.model_class(**merged_config)
               else:
                   raise RuntimeError("No detector implementations available")
      
       return self._cache[cache_key]
  
   def get_tracker(self, **kwargs) -> Optional[ITracker]:
       """Get or create tracker instance."""
       if self.config.tracker_type == TrackerType.NONE:
           return None
      
       cache_key = f"tracker_{self.config.tracker_type.value}"
      
       if cache_key not in self._cache:
           available = ModelRegistry.get_available_trackers()
          
           if self.config.tracker_type in available:
               model_info = available[self.config.tracker_type]
               merged_config = {**model_info.default_config, **kwargs}
               self._cache[cache_key] = model_info.model_class(**merged_config)
           elif available:
               # Fallback to simple tracker
               if TrackerType.SIMPLE in available:
                   model_info = available[TrackerType.SIMPLE]
                   merged_config = {**model_info.default_config, **kwargs}
                   self._cache[cache_key] = model_info.model_class(**merged_config)
                   logger.warning(
                       f"Requested tracker {self.config.tracker_type} not available, "
                       f"falling back to SIMPLE"
                   )
      
       return self._cache.get(cache_key)
  
   def get_depth_estimator(self, **kwargs) -> Optional[IDepthEstimator]:
       """Get or create depth estimator instance."""
       if self.config.depth_estimator_type == DepthEstimatorType.NONE:
           return None
      
       cache_key = f"depth_{self.config.depth_estimator_type.value}"
      
       if cache_key not in self._cache:
           available = ModelRegistry.get_available_depth_estimators()
          
           if self.config.depth_estimator_type in available:
               model_info = available[self.config.depth_estimator_type]
               merged_config = {**model_info.default_config, **kwargs}
               self._cache[cache_key] = model_info.model_class(**merged_config)
           elif DepthEstimatorType.ESTIMATED in available:
               # Fallback to rule-based
               model_info = available[DepthEstimatorType.ESTIMATED]
               merged_config = {**model_info.default_config, **kwargs}
               self._cache[cache_key] = model_info.model_class(**merged_config)
      
       return self._cache.get(cache_key)
  
   def get_room_classifier(self, **kwargs) -> Optional[IRoomClassifier]:
       """Get or create room classifier instance."""
       if self.config.room_classifier_type == RoomClassifierType.NONE:
           return None
      
       cache_key = f"room_{self.config.room_classifier_type.value}"
      
       if cache_key not in self._cache:
           available = ModelRegistry.get_available_room_classifiers()
          
           if self.config.room_classifier_type in available:
               model_info = available[self.config.room_classifier_type]
               merged_config = {**model_info.default_config, **kwargs}
               self._cache[cache_key] = model_info.model_class(**merged_config)
           elif RoomClassifierType.RULE_BASED in available:
               model_info = available[RoomClassifierType.RULE_BASED]
               merged_config = {**model_info.default_config, **kwargs}
               self._cache[cache_key] = model_info.model_class(**merged_config)
      
       return self._cache.get(cache_key)
  
   def get_summary_generator(self, **kwargs) -> ISummaryGenerator:
       """Get or create summary generator instance."""
       cache_key = f"summary_{self.config.summary_generator_type.value}"
      
       if cache_key not in self._cache:
           available = ModelRegistry.get_available_summary_generators()
          
           # Try preferred, then fallbacks
           for gen_type in [
               self.config.summary_generator_type,
               SummaryGeneratorType.OLLAMA,
               SummaryGeneratorType.TRANSFORMERS,
               SummaryGeneratorType.TEMPLATE
           ]:
               if gen_type in available:
                   model_info = available[gen_type]
                   merged_config = {**model_info.default_config, **kwargs}
                   instance = model_info.model_class(**merged_config)
                   if instance.is_available():
                       self._cache[cache_key] = instance
                       break
          
           if cache_key not in self._cache:
               raise RuntimeError("No summary generator available")
      
       return self._cache[cache_key]
  
   def clear_cache(self):
       """Clear the component cache."""
       self._cache.clear()




# Default configuration
DEFAULT_CONFIG = ProcessingConfig(
   model_type=ModelType.YOLOV8_NANO,
   confidence_threshold=0.4,
   filter_household_only=True,
   tracker_type=TrackerType.SIMPLE,
   depth_estimator_type=DepthEstimatorType.ESTIMATED,
   room_classifier_type=RoomClassifierType.RULE_BASED,
   summary_generator_type=SummaryGeneratorType.TEMPLATE,
   extraction_rate="1_per_second",
   max_frames=100,
   save_annotated_frames=True,
   preview_frame_count=5
)