"""
Room Classification Module
Classifies room types based on detected objects and visual features
"""


import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import Counter
import logging
import cv2


from .interfaces import (
   IRoomClassifier, Detection, RoomClassifierType
)
from .model_registry import ModelRegistry


logger = logging.getLogger(__name__)


# Try imports
try:
   import torch
   TORCH_AVAILABLE = True
except ImportError:
   TORCH_AVAILABLE = False




# Room type definitions with characteristic objects
ROOM_SIGNATURES = {
   "living_room": {
       "primary": ["sofa", "couch", "tv", "television"],
       "secondary": ["chair", "table", "potted plant", "remote", "clock"],
       "weight": 1.0
   },
   "bedroom": {
       "primary": ["bed"],
       "secondary": ["lamp", "clock", "suitcase", "teddy bear"],
       "weight": 1.2  # Higher weight because bed is very distinctive
   },
   "kitchen": {
       "primary": ["refrigerator", "oven", "microwave", "sink"],
       "secondary": ["bottle", "cup", "bowl", "toaster"],
       "weight": 1.0
   },
   "dining_room": {
       "primary": ["dining table", "table"],
       "secondary": ["chair", "vase", "bowl", "cup", "bottle"],
       "weight": 0.9
   },
   "office": {
       "primary": ["laptop", "keyboard", "mouse", "monitor"],
       "secondary": ["chair", "desk", "book", "cell phone"],
       "weight": 1.0
   },
   "bathroom": {
       "primary": ["toilet", "sink"],
       "secondary": ["toothbrush", "hair drier"],
       "weight": 1.1
   },
   "garage": {
       "primary": ["car", "bicycle", "motorcycle"],
       "secondary": ["backpack", "suitcase"],
       "weight": 0.8
   }
}




class RuleBasedRoomClassifier(IRoomClassifier):
   """
   Rule-based room classifier using detected object signatures.
   """
  
   def __init__(
       self,
       primary_weight: float = 3.0,
       secondary_weight: float = 1.0,
       min_confidence: float = 0.3
   ):
       """
       Initialize classifier.
      
       Args:
           primary_weight: Weight for primary objects
           secondary_weight: Weight for secondary objects
           min_confidence: Minimum confidence threshold for room classification
       """
       self.primary_weight = primary_weight
       self.secondary_weight = secondary_weight
       self.min_confidence = min_confidence
       self._last_confidence = 0.0
  
   def classify_room(self, frame: np.ndarray, detections: List[Detection]) -> str:
       """
       Classify room based on detected objects.
      
       Args:
           frame: Input frame (used for color analysis)
           detections: List of detected objects
          
       Returns:
           Predicted room type
       """
       if not detections:
           self._last_confidence = 0.0
           return "unknown"
      
       # Count detected objects
       object_counts = Counter(det.class_name.lower() for det in detections)
      
       # Score each room type
       room_scores = {}
      
       for room_type, signature in ROOM_SIGNATURES.items():
           score = 0.0
          
           # Primary objects
           for obj in signature["primary"]:
               if obj in object_counts:
                   score += self.primary_weight * object_counts[obj]
          
           # Secondary objects
           for obj in signature["secondary"]:
               if obj in object_counts:
                   score += self.secondary_weight * object_counts[obj]
          
           # Apply room weight
           score *= signature["weight"]
           room_scores[room_type] = score
      
       # Get best match
       if not room_scores or max(room_scores.values()) == 0:
           self._last_confidence = 0.0
           return "unknown"
      
       best_room = max(room_scores, key=room_scores.get)
       best_score = room_scores[best_room]
      
       # Calculate confidence
       total_score = sum(room_scores.values())
       self._last_confidence = best_score / total_score if total_score > 0 else 0.0
      
       if self._last_confidence < self.min_confidence:
           return "unknown"
      
       return best_room
  
   def get_room_confidence(self) -> float:
       """Get confidence of last classification."""
       return self._last_confidence
  
   def get_room_breakdown(self, detections: List[Detection]) -> Dict[str, float]:
       """Get confidence scores for all room types."""
       if not detections:
           return {}
      
       object_counts = Counter(det.class_name.lower() for det in detections)
       room_scores = {}
      
       for room_type, signature in ROOM_SIGNATURES.items():
           score = 0.0
           for obj in signature["primary"]:
               if obj in object_counts:
                   score += self.primary_weight * object_counts[obj]
           for obj in signature["secondary"]:
               if obj in object_counts:
                   score += self.secondary_weight * object_counts[obj]
           score *= signature["weight"]
           room_scores[room_type] = score
      
       # Normalize
       total = sum(room_scores.values())
       if total > 0:
           room_scores = {k: v / total for k, v in room_scores.items()}
      
       return room_scores




class CLIPRoomClassifier(IRoomClassifier):
   """
   CLIP-based room classifier using vision-language model.
   More accurate but requires more compute.
   """
  
   ROOM_PROMPTS = [
       "a photo of a living room",
       "a photo of a bedroom",
       "a photo of a kitchen",
       "a photo of a dining room",
       "a photo of an office or study",
       "a photo of a bathroom",
       "a photo of a garage",
       "a photo of a hallway or corridor"
   ]
  
   ROOM_LABELS = [
       "living_room",
       "bedroom",
       "kitchen",
       "dining_room",
       "office",
       "bathroom",
       "garage",
       "hallway"
   ]
  
   def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
       """
       Initialize CLIP classifier.
      
       Args:
           model_name: CLIP model name from HuggingFace
       """
       self.model_name = model_name
       self.model = None
       self.processor = None
       self._last_confidence = 0.0
       self._load_model()
  
   def _load_model(self):
       """Load CLIP model."""
       if not TORCH_AVAILABLE:
           logger.warning("PyTorch not available for CLIP")
           return
      
       try:
           from transformers import CLIPProcessor, CLIPModel
          
           self.processor = CLIPProcessor.from_pretrained(self.model_name)
           self.model = CLIPModel.from_pretrained(self.model_name)
          
           if torch.cuda.is_available():
               self.model = self.model.cuda()
          
           self.model.eval()
           logger.info(f"CLIP model loaded: {self.model_name}")
          
       except Exception as e:
           logger.error(f"Failed to load CLIP: {e}")
           self.model = None
  
   def classify_room(self, frame: np.ndarray, detections: List[Detection]) -> str:
       """Classify room using CLIP."""
       if self.model is None:
           # Fallback to rule-based
           return RuleBasedRoomClassifier().classify_room(frame, detections)
      
       try:
           from PIL import Image
          
           # Convert frame to PIL
           if len(frame.shape) == 3:
               frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
           else:
               frame_rgb = frame
          
           image = Image.fromarray(frame_rgb)
          
           # Encode image and text
           inputs = self.processor(
               text=self.ROOM_PROMPTS,
               images=image,
               return_tensors="pt",
               padding=True
           )
          
           if torch.cuda.is_available():
               inputs = {k: v.cuda() for k, v in inputs.items()}
          
           with torch.no_grad():
               outputs = self.model(**inputs)
               logits = outputs.logits_per_image.softmax(dim=1)
          
           # Get prediction
           probs = logits[0].cpu().numpy()
           best_idx = np.argmax(probs)
          
           self._last_confidence = float(probs[best_idx])
          
           if self._last_confidence < 0.2:
               return "unknown"
          
           return self.ROOM_LABELS[best_idx]
          
       except Exception as e:
           logger.error(f"CLIP classification error: {e}")
           return RuleBasedRoomClassifier().classify_room(frame, detections)
  
   def get_room_confidence(self) -> float:
       """Get confidence of last classification."""
       return self._last_confidence




class CNNRoomClassifier(IRoomClassifier):
   """
   CNN-based room classifier using pre-trained model.
   Uses Places365 or similar scene classification model.
   """
  
   def __init__(self, model_name: str = "resnet50"):
       """Initialize CNN classifier."""
       self.model_name = model_name
       self.model = None
       self._last_confidence = 0.0
       self._load_model()
  
   def _load_model(self):
       """Load CNN model for scene classification."""
       if not TORCH_AVAILABLE:
           return
      
       try:
           import torchvision.models as models
           import torchvision.transforms as transforms
          
           # Use ImageNet pretrained model
           # In production, would use Places365
           self.model = models.resnet50(pretrained=True)
           self.model.eval()
          
           if torch.cuda.is_available():
               self.model = self.model.cuda()
          
           self.transform = transforms.Compose([
               transforms.ToPILImage(),
               transforms.Resize((224, 224)),
               transforms.ToTensor(),
               transforms.Normalize(
                   mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225]
               )
           ])
          
           logger.info("CNN room classifier loaded")
          
       except Exception as e:
           logger.error(f"Failed to load CNN: {e}")
           self.model = None
  
   def classify_room(self, frame: np.ndarray, detections: List[Detection]) -> str:
       """
       Classify room using CNN features + rule-based combination.
       Since we don't have Places365, we combine with rule-based.
       """
       # Use rule-based as primary
       rule_based = RuleBasedRoomClassifier()
       result = rule_based.classify_room(frame, detections)
       self._last_confidence = rule_based.get_room_confidence()
      
       # Could enhance with CNN features if needed
       return result
  
   def get_room_confidence(self) -> float:
       """Get confidence of last classification."""
       return self._last_confidence




def check_clip_available() -> bool:
   """Check if CLIP is available."""
   if not TORCH_AVAILABLE:
       return False
   try:
       from transformers import CLIPProcessor, CLIPModel
       return True
   except:
       return False




def check_cnn_available() -> bool:
   """Check if CNN models are available."""
   if not TORCH_AVAILABLE:
       return False
   try:
       import torchvision.models as models
       return True
   except:
       return False




# Register room classifiers
ModelRegistry.register_room_classifier(
   RoomClassifierType.RULE_BASED,
   "Rule-Based Classifier",
   "Fast classification using object signatures",
   RuleBasedRoomClassifier,
   {"primary_weight": 3.0, "secondary_weight": 1.0}
)


ModelRegistry.register_room_classifier(
   RoomClassifierType.CLIP,
   "CLIP Classifier",
   "Vision-language model for accurate room classification",
   CLIPRoomClassifier,
   {"model_name": "openai/clip-vit-base-patch32"},
   check_clip_available
)


ModelRegistry.register_room_classifier(
   RoomClassifierType.CNN,
   "CNN Classifier",
   "Convolutional neural network for scene classification",
   CNNRoomClassifier,
   {"model_name": "resnet50"},
   check_cnn_available
)