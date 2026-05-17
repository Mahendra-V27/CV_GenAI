"""
Advanced Object Tracking Implementations
DeepSORT, ByteTrack, and Simple tracker implementations
"""


import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import logging


from .interfaces import (
   ITracker, Detection, TrackedObject, BoundingBox, TrackerType
)
from .model_registry import ModelRegistry


logger = logging.getLogger(__name__)




def calculate_iou(box1: BoundingBox, box2: BoundingBox) -> float:
   """Calculate IoU between two bounding boxes."""
   x1 = max(box1.x1, box2.x1)
   y1 = max(box1.y1, box2.y1)
   x2 = min(box1.x2, box2.x2)
   y2 = min(box1.y2, box2.y2)
  
   if x2 <= x1 or y2 <= y1:
       return 0.0
  
   intersection = (x2 - x1) * (y2 - y1)
   area1 = box1.area
   area2 = box2.area
   union = area1 + area2 - intersection
  
   return intersection / union if union > 0 else 0.0




class KalmanFilter2D:
   """Simple 2D Kalman Filter for object tracking."""
  
   def __init__(self, initial_state: np.ndarray):
       """
       Initialize Kalman filter.
       State: [x, y, vx, vy] (position and velocity)
       """
       self.state = np.zeros(4)
       self.state[:2] = initial_state[:2]
      
       # State transition matrix
       self.F = np.array([
           [1, 0, 1, 0],
           [0, 1, 0, 1],
           [0, 0, 1, 0],
           [0, 0, 0, 1]
       ], dtype=np.float32)
      
       # Observation matrix
       self.H = np.array([
           [1, 0, 0, 0],
           [0, 1, 0, 0]
       ], dtype=np.float32)
      
       # Covariance matrices
       self.P = np.eye(4) * 100  # State covariance
       self.Q = np.eye(4) * 0.1  # Process noise
       self.R = np.eye(2) * 10   # Measurement noise
  
   def predict(self) -> np.ndarray:
       """Predict next state."""
       self.state = self.F @ self.state
       self.P = self.F @ self.P @ self.F.T + self.Q
       return self.state[:2]
  
   def update(self, measurement: np.ndarray):
       """Update state with measurement."""
       y = measurement - self.H @ self.state
       S = self.H @ self.P @ self.H.T + self.R
       K = self.P @ self.H.T @ np.linalg.inv(S)
      
       self.state = self.state + K @ y
       self.P = (np.eye(4) - K @ self.H) @ self.P
  
   @property
   def position(self) -> np.ndarray:
       return self.state[:2]




class SimpleTracker(ITracker):
   """
   Simple IoU-based tracker.
   Good for scenarios with limited object movement between frames.
   """
  
   def __init__(
       self,
       iou_threshold: float = 0.3,
       max_age: int = 10,
       min_hits: int = 2
   ):
       self.iou_threshold = iou_threshold
       self.max_age = max_age
       self.min_hits = min_hits
      
       self.tracks: Dict[int, TrackedObject] = {}
       self.next_id = 1
       self.frame_count = 0
       self.track_ages: Dict[int, int] = {}
       self.track_hits: Dict[int, int] = {}
  
   def update(self, detections: List[Detection], frame: np.ndarray) -> List[Detection]:
       """Update tracker with new detections."""
       self.frame_count += 1
      
       if not detections:
           # Age out old tracks
           self._age_tracks()
           return []
      
       if not self.tracks:
           # Initialize tracks
           for det in detections:
               self._create_track(det)
           return detections
      
       # Build cost matrix (1 - IoU)
       track_ids = list(self.tracks.keys())
       cost_matrix = np.zeros((len(detections), len(track_ids)))
      
       for i, det in enumerate(detections):
           for j, track_id in enumerate(track_ids):
               track = self.tracks[track_id]
               if track.detections and track.class_name == det.class_name:
                   last_det = track.detections[-1]
                   iou = calculate_iou(det.bbox, last_det.bbox)
                   cost_matrix[i, j] = 1 - iou
               else:
                   cost_matrix[i, j] = 1.0
      
       # Hungarian algorithm for assignment
       row_indices, col_indices = linear_sum_assignment(cost_matrix)
      
       matched_dets = set()
       matched_tracks = set()
      
       for row, col in zip(row_indices, col_indices):
           if cost_matrix[row, col] < (1 - self.iou_threshold):
               track_id = track_ids[col]
               det = detections[row]
               det.track_id = track_id
              
               self.tracks[track_id].detections.append(det)
               self.track_ages[track_id] = 0
               self.track_hits[track_id] += 1
              
               matched_dets.add(row)
               matched_tracks.add(col)
      
       # Create new tracks for unmatched detections
       for i, det in enumerate(detections):
           if i not in matched_dets:
               self._create_track(det)
      
       # Age unmatched tracks
       for j, track_id in enumerate(track_ids):
           if j not in matched_tracks:
               self.track_ages[track_id] += 1
               if self.track_ages[track_id] > self.max_age:
                   self.tracks[track_id].is_active = False
      
       return detections
  
   def _create_track(self, detection: Detection):
       """Create a new track."""
       track_id = self.next_id
       self.next_id += 1
      
       detection.track_id = track_id
       self.tracks[track_id] = TrackedObject(
           track_id=track_id,
           class_name=detection.class_name,
           detections=[detection]
       )
       self.track_ages[track_id] = 0
       self.track_hits[track_id] = 1
  
   def _age_tracks(self):
       """Age all tracks."""
       for track_id in list(self.tracks.keys()):
           self.track_ages[track_id] += 1
           if self.track_ages[track_id] > self.max_age:
               self.tracks[track_id].is_active = False
  
   def get_tracks(self) -> List[TrackedObject]:
       """Get all confirmed tracks."""
       return [
           track for track in self.tracks.values()
           if track.is_active and self.track_hits.get(track.track_id, 0) >= self.min_hits
       ]
  
   def reset(self):
       """Reset tracker state."""
       self.tracks.clear()
       self.track_ages.clear()
       self.track_hits.clear()
       self.next_id = 1
       self.frame_count = 0




class DeepSORTTracker(ITracker):
   """
   DeepSORT tracker implementation.
   Uses Kalman filtering and appearance features for robust tracking.
   """
  
   def __init__(
       self,
       max_age: int = 30,
       min_hits: int = 3,
       iou_threshold: float = 0.3,
       max_cosine_distance: float = 0.4,
       nn_budget: int = 100
   ):
       self.max_age = max_age
       self.min_hits = min_hits
       self.iou_threshold = iou_threshold
       self.max_cosine_distance = max_cosine_distance
       self.nn_budget = nn_budget
      
       self.tracks: Dict[int, TrackedObject] = {}
       self.kalman_filters: Dict[int, KalmanFilter2D] = {}
       self.appearance_features: Dict[int, List[np.ndarray]] = {}
      
       self.next_id = 1
       self.frame_count = 0
       self.track_ages: Dict[int, int] = {}
       self.track_hits: Dict[int, int] = {}
  
   def _extract_features(self, frame: np.ndarray, bbox: BoundingBox) -> np.ndarray:
       """Extract appearance features from detection (simple histogram-based)."""
       x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2
       x1, y1 = max(0, x1), max(0, y1)
       x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
      
       if x2 <= x1 or y2 <= y1:
           return np.zeros(256)
      
       crop = frame[y1:y2, x1:x2]
      
       # Simple color histogram as feature
       hist_features = []
       for i in range(3):
           hist = np.histogram(crop[:, :, i], bins=64, range=(0, 256))[0]
           hist = hist.astype(np.float32) / (hist.sum() + 1e-6)
           hist_features.append(hist)
      
       feature = np.concatenate(hist_features)
       feature = feature / (np.linalg.norm(feature) + 1e-6)
       return feature
  
   def _cosine_distance(self, feat1: np.ndarray, features: List[np.ndarray]) -> float:
       """Calculate minimum cosine distance to feature gallery."""
       if not features:
           return 1.0
      
       gallery = np.array(features[-self.nn_budget:])
       distances = 1 - np.dot(gallery, feat1)
       return np.min(distances)
  
   def update(self, detections: List[Detection], frame: np.ndarray) -> List[Detection]:
       """Update tracker with DeepSORT algorithm."""
       self.frame_count += 1
      
       # Predict new positions
       for track_id, kf in self.kalman_filters.items():
           kf.predict()
      
       if not detections:
           self._age_tracks()
           return []
      
       # Extract features for detections
       det_features = [self._extract_features(frame, det.bbox) for det in detections]
      
       if not self.tracks:
           for det, feat in zip(detections, det_features):
               self._create_track(det, feat)
           return detections
      
       track_ids = list(self.tracks.keys())
      
       # Build cost matrix combining IoU and appearance
       cost_matrix = np.zeros((len(detections), len(track_ids)))
      
       for i, (det, feat) in enumerate(zip(detections, det_features)):
           for j, track_id in enumerate(track_ids):
               track = self.tracks[track_id]
              
               # IoU cost
               if track.detections and track.class_name == det.class_name:
                   last_det = track.detections[-1]
                   iou = calculate_iou(det.bbox, last_det.bbox)
                   iou_cost = 1 - iou
               else:
                   iou_cost = 1.0
              
               # Appearance cost
               if track_id in self.appearance_features:
                   appearance_cost = self._cosine_distance(
                       feat, self.appearance_features[track_id]
                   )
               else:
                   appearance_cost = 1.0
              
               # Combined cost (weighted)
               cost_matrix[i, j] = 0.5 * iou_cost + 0.5 * appearance_cost
      
       # Assignment
       row_indices, col_indices = linear_sum_assignment(cost_matrix)
      
       matched_dets = set()
       matched_tracks = set()
      
       for row, col in zip(row_indices, col_indices):
           if cost_matrix[row, col] < 0.7:  # Threshold
               track_id = track_ids[col]
               det = detections[row]
               feat = det_features[row]
              
               det.track_id = track_id
               self.tracks[track_id].detections.append(det)
              
               # Update Kalman filter
               if track_id in self.kalman_filters:
                   center = np.array(det.bbox.center, dtype=np.float32)
                   self.kalman_filters[track_id].update(center)
              
               # Update appearance features
               if track_id not in self.appearance_features:
                   self.appearance_features[track_id] = []
               self.appearance_features[track_id].append(feat)
               if len(self.appearance_features[track_id]) > self.nn_budget:
                   self.appearance_features[track_id].pop(0)
              
               self.track_ages[track_id] = 0
               self.track_hits[track_id] += 1
              
               matched_dets.add(row)
               matched_tracks.add(col)
      
       # New tracks
       for i, det in enumerate(detections):
           if i not in matched_dets:
               self._create_track(det, det_features[i])
      
       # Age unmatched
       for j, track_id in enumerate(track_ids):
           if j not in matched_tracks:
               self.track_ages[track_id] += 1
               if self.track_ages[track_id] > self.max_age:
                   self.tracks[track_id].is_active = False
      
       return detections
  
   def _create_track(self, detection: Detection, feature: np.ndarray):
       """Create new track with Kalman filter."""
       track_id = self.next_id
       self.next_id += 1
      
       detection.track_id = track_id
       self.tracks[track_id] = TrackedObject(
           track_id=track_id,
           class_name=detection.class_name,
           detections=[detection]
       )
      
       center = np.array(detection.bbox.center, dtype=np.float32)
       self.kalman_filters[track_id] = KalmanFilter2D(center)
       self.appearance_features[track_id] = [feature]
      
       self.track_ages[track_id] = 0
       self.track_hits[track_id] = 1
  
   def _age_tracks(self):
       """Age all tracks."""
       for track_id in list(self.tracks.keys()):
           self.track_ages[track_id] += 1
           if self.track_ages[track_id] > self.max_age:
               self.tracks[track_id].is_active = False
  
   def get_tracks(self) -> List[TrackedObject]:
       """Get confirmed tracks."""
       return [
           track for track in self.tracks.values()
           if track.is_active and self.track_hits.get(track.track_id, 0) >= self.min_hits
       ]
  
   def reset(self):
       """Reset tracker."""
       self.tracks.clear()
       self.kalman_filters.clear()
       self.appearance_features.clear()
       self.track_ages.clear()
       self.track_hits.clear()
       self.next_id = 1
       self.frame_count = 0




class ByteTracker(ITracker):
   """
   ByteTrack implementation.
   Tracks all detection boxes including low-confidence ones.
   """
  
   def __init__(
       self,
       high_threshold: float = 0.5,
       low_threshold: float = 0.1,
       match_threshold: float = 0.8,
       max_age: int = 30,
       min_hits: int = 3
   ):
       self.high_threshold = high_threshold
       self.low_threshold = low_threshold
       self.match_threshold = match_threshold
       self.max_age = max_age
       self.min_hits = min_hits
      
       self.tracks: Dict[int, TrackedObject] = {}
       self.kalman_filters: Dict[int, KalmanFilter2D] = {}
      
       self.next_id = 1
       self.frame_count = 0
       self.track_ages: Dict[int, int] = {}
       self.track_hits: Dict[int, int] = {}
  
   def update(self, detections: List[Detection], frame: np.ndarray) -> List[Detection]:
       """Update with ByteTrack algorithm."""
       self.frame_count += 1
      
       # Predict positions
       for kf in self.kalman_filters.values():
           kf.predict()
      
       if not detections:
           self._age_tracks()
           return []
      
       # Separate high and low confidence detections
       high_dets = [d for d in detections if d.confidence >= self.high_threshold]
       low_dets = [d for d in detections if self.low_threshold <= d.confidence < self.high_threshold]
      
       if not self.tracks:
           for det in high_dets:
               self._create_track(det)
           return detections
      
       # First association: high confidence with tracks
       track_ids = [tid for tid, t in self.tracks.items() if t.is_active]
      
       if high_dets and track_ids:
           matched_high, unmatched_tracks, unmatched_high = self._associate(
               high_dets, track_ids, self.match_threshold
           )
          
           for det_idx, track_id in matched_high:
               det = high_dets[det_idx]
               det.track_id = track_id
               self.tracks[track_id].detections.append(det)
              
               center = np.array(det.bbox.center, dtype=np.float32)
               if track_id in self.kalman_filters:
                   self.kalman_filters[track_id].update(center)
              
               self.track_ages[track_id] = 0
               self.track_hits[track_id] += 1
       else:
           unmatched_tracks = set(track_ids)
           unmatched_high = set(range(len(high_dets)))
      
       # Second association: low confidence with remaining tracks
       if low_dets and unmatched_tracks:
           remaining_track_ids = list(unmatched_tracks)
           matched_low, still_unmatched, _ = self._associate(
               low_dets, remaining_track_ids, 0.5
           )
          
           for det_idx, track_id in matched_low:
               det = low_dets[det_idx]
               det.track_id = track_id
               self.tracks[track_id].detections.append(det)
              
               center = np.array(det.bbox.center, dtype=np.float32)
               if track_id in self.kalman_filters:
                   self.kalman_filters[track_id].update(center)
              
               self.track_ages[track_id] = 0
               self.track_hits[track_id] += 1
               unmatched_tracks.discard(track_id)
      
       # Create new tracks for unmatched high-confidence detections
       for idx in unmatched_high:
           self._create_track(high_dets[idx])
      
       # Age unmatched tracks
       for track_id in unmatched_tracks:
           self.track_ages[track_id] += 1
           if self.track_ages[track_id] > self.max_age:
               self.tracks[track_id].is_active = False
      
       return detections
  
   def _associate(
       self,
       detections: List[Detection],
       track_ids: List[int],
       threshold: float
   ) -> Tuple[List[Tuple[int, int]], set, set]:
       """Associate detections with tracks using IoU."""
       if not detections or not track_ids:
           return [], set(track_ids), set(range(len(detections)))
      
       # Build IoU matrix
       iou_matrix = np.zeros((len(detections), len(track_ids)))
      
       for i, det in enumerate(detections):
           for j, track_id in enumerate(track_ids):
               track = self.tracks[track_id]
               if track.detections and track.class_name == det.class_name:
                   last_det = track.detections[-1]
                   iou_matrix[i, j] = calculate_iou(det.bbox, last_det.bbox)
      
       # Hungarian assignment
       row_indices, col_indices = linear_sum_assignment(-iou_matrix)
      
       matched = []
       unmatched_dets = set(range(len(detections)))
       unmatched_tracks = set(range(len(track_ids)))
      
       for row, col in zip(row_indices, col_indices):
           if iou_matrix[row, col] >= (1 - threshold):
               matched.append((row, track_ids[col]))
               unmatched_dets.discard(row)
               unmatched_tracks.discard(col)
      
       unmatched_track_ids = {track_ids[i] for i in unmatched_tracks}
       return matched, unmatched_track_ids, unmatched_dets
  
   def _create_track(self, detection: Detection):
       """Create new track."""
       track_id = self.next_id
       self.next_id += 1
      
       detection.track_id = track_id
       self.tracks[track_id] = TrackedObject(
           track_id=track_id,
           class_name=detection.class_name,
           detections=[detection]
       )
      
       center = np.array(detection.bbox.center, dtype=np.float32)
       self.kalman_filters[track_id] = KalmanFilter2D(center)
      
       self.track_ages[track_id] = 0
       self.track_hits[track_id] = 1
  
   def _age_tracks(self):
       """Age tracks."""
       for track_id in list(self.tracks.keys()):
           self.track_ages[track_id] += 1
           if self.track_ages[track_id] > self.max_age:
               self.tracks[track_id].is_active = False
  
   def get_tracks(self) -> List[TrackedObject]:
       """Get confirmed tracks."""
       return [
           track for track in self.tracks.values()
           if track.is_active and self.track_hits.get(track.track_id, 0) >= self.min_hits
       ]
  
   def reset(self):
       """Reset tracker."""
       self.tracks.clear()
       self.kalman_filters.clear()
       self.track_ages.clear()
       self.track_hits.clear()
       self.next_id = 1
       self.frame_count = 0




# Register trackers
ModelRegistry.register_tracker(
   TrackerType.SIMPLE,
   "Simple IoU Tracker",
   "Basic IoU-based tracking, good for slow-moving objects",
   SimpleTracker,
   {"iou_threshold": 0.3, "max_age": 10, "min_hits": 2}
)


ModelRegistry.register_tracker(
   TrackerType.DEEPSORT,
   "DeepSORT Tracker",
   "Robust tracking with Kalman filtering and appearance features",
   DeepSORTTracker,
   {"max_age": 30, "min_hits": 3, "iou_threshold": 0.3}
)


ModelRegistry.register_tracker(
   TrackerType.BYTETRACK,
   "ByteTrack",
   "State-of-the-art tracker using all detection boxes",
   ByteTracker,
   {"high_threshold": 0.5, "low_threshold": 0.1, "max_age": 30}
)