"""
Inventory Management Module
Handles object aggregation, deduplication, and inventory generation
"""


from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import numpy as np
from dataclasses import dataclass, field
import json
from datetime import datetime


from .detection import DetectedObject, ITEM_CATEGORIES




@dataclass
class InventoryItem:
   """Represents an item in the inventory."""
   name: str
   count: int
   category: str
   avg_confidence: float
   detections: List[Dict] = field(default_factory=list)
  
   def to_dict(self) -> dict:
       return {
           "name": self.name,
           "count": self.count,
           "category": self.category,
           "avg_confidence": round(self.avg_confidence, 2),
           "detection_count": len(self.detections)
       }




class InventoryManager:
   """
   Manages inventory aggregation and deduplication from detected objects.
  
   Uses spatial and temporal deduplication to prevent counting
   the same object multiple times across video frames.
   """
  
   def __init__(
       self,
       iou_threshold: float = 0.5,
       position_threshold: int = 100,
       frame_window: int = 5
   ):
       """
       Initialize InventoryManager.
      
       Args:
           iou_threshold: IoU threshold for considering objects as same
           position_threshold: Pixel distance threshold for position-based matching
           frame_window: Number of frames to look back for deduplication
       """
       self.iou_threshold = iou_threshold
       self.position_threshold = position_threshold
       self.frame_window = frame_window
      
   def calculate_iou(
       self,
       box1: Tuple[int, int, int, int],
       box2: Tuple[int, int, int, int]
   ) -> float:
       """
       Calculate Intersection over Union between two bounding boxes.
      
       Args:
           box1, box2: Bounding boxes as (x1, y1, x2, y2)
          
       Returns:
           IoU value between 0 and 1
       """
       x1_1, y1_1, x2_1, y2_1 = box1
       x1_2, y1_2, x2_2, y2_2 = box2
      
       # Calculate intersection
       xi1 = max(x1_1, x1_2)
       yi1 = max(y1_1, y1_2)
       xi2 = min(x2_1, x2_2)
       yi2 = min(y2_1, y2_2)
      
       if xi2 <= xi1 or yi2 <= yi1:
           return 0.0
          
       inter_area = (xi2 - xi1) * (yi2 - yi1)
      
       # Calculate union
       box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
       box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
       union_area = box1_area + box2_area - inter_area
      
       if union_area == 0:
           return 0.0
          
       return inter_area / union_area
  
   def calculate_center_distance(
       self,
       center1: Tuple[int, int],
       center2: Tuple[int, int]
   ) -> float:
       """Calculate Euclidean distance between two centers."""
       return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
  
   def is_same_object(
       self,
       obj1: DetectedObject,
       obj2: DetectedObject
   ) -> bool:
       """
       Determine if two detected objects are the same physical object.
      
       Args:
           obj1, obj2: Detected objects to compare
          
       Returns:
           True if likely the same object
       """
       # Must be same class
       if obj1.class_name != obj2.class_name:
           return False
          
       # Check frame proximity
       if abs(obj1.frame_index - obj2.frame_index) > self.frame_window:
           return False
          
       # Check IoU
       iou = self.calculate_iou(obj1.bbox, obj2.bbox)
       if iou > self.iou_threshold:
           return True
          
       # Check center distance
       distance = self.calculate_center_distance(obj1.center, obj2.center)
       if distance < self.position_threshold:
           return True
          
       return False
  
   def deduplicate_detections(
       self,
       all_detections: List[List[DetectedObject]]
   ) -> List[DetectedObject]:
       """
       Remove duplicate detections across frames.
      
       Uses a greedy matching approach to group detections
       that likely represent the same physical object.
      
       Args:
           all_detections: List of detection lists, one per frame
          
       Returns:
           List of unique detected objects
       """
       # Flatten all detections
       flat_detections = []
       for frame_detections in all_detections:
           flat_detections.extend(frame_detections)
          
       if not flat_detections:
           return []
          
       # Group detections by class for efficiency
       class_detections = defaultdict(list)
       for det in flat_detections:
           class_detections[det.class_name].append(det)
          
       unique_objects = []
      
       # Process each class separately
       for class_name, detections in class_detections.items():
           # Sort by frame index
           detections.sort(key=lambda x: x.frame_index)
          
           # Track groups of same objects
           groups = []  # Each group is a list of detections for same object
          
           for det in detections:
               matched_group = None
              
               # Try to match with existing groups
               for group in groups:
                   # Check against last few detections in group
                   for existing_det in group[-self.frame_window:]:
                       if self.is_same_object(det, existing_det):
                           matched_group = group
                           break
                   if matched_group:
                       break
                      
               if matched_group:
                   matched_group.append(det)
               else:
                   # Start new group
                   groups.append([det])
                  
           # Take the best detection from each group (highest confidence)
           for group in groups:
               best_det = max(group, key=lambda x: x.confidence)
               unique_objects.append(best_det)
              
       return unique_objects
  
   def aggregate_inventory(
       self,
       unique_objects: List[DetectedObject]
   ) -> Dict[str, InventoryItem]:
       """
       Aggregate unique objects into inventory items.
      
       Args:
           unique_objects: List of unique detected objects
          
       Returns:
           Dictionary mapping item names to InventoryItem instances
       """
       inventory = {}
      
       # Group by class name
       class_objects = defaultdict(list)
       for obj in unique_objects:
           class_objects[obj.class_name].append(obj)
          
       for class_name, objects in class_objects.items():
           # Calculate average confidence
           avg_conf = sum(o.confidence for o in objects) / len(objects)
          
           # Get category
           category = ITEM_CATEGORIES.get(class_name.lower(), "Other")
          
           inventory[class_name] = InventoryItem(
               name=class_name.title(),
               count=len(objects),
               category=category,
               avg_confidence=avg_conf,
               detections=[o.to_dict() for o in objects]
           )
          
       return inventory
  
   def generate_inventory_from_detections(
       self,
       all_detections: List[List[DetectedObject]]
   ) -> Tuple[Dict[str, InventoryItem], Dict]:
       """
       Full pipeline: deduplicate and aggregate into inventory.
      
       Args:
           all_detections: List of detection lists from frames
          
       Returns:
           Tuple of (inventory dict, stats dict)
       """
       # Get unique objects
       unique_objects = self.deduplicate_detections(all_detections)
      
       # Aggregate into inventory
       inventory = self.aggregate_inventory(unique_objects)
      
       # Calculate stats
       total_raw_detections = sum(len(d) for d in all_detections)
       stats = {
           "total_frames_processed": len(all_detections),
           "total_raw_detections": total_raw_detections,
           "unique_objects_found": len(unique_objects),
           "unique_item_types": len(inventory),
           "deduplication_ratio": round(
               len(unique_objects) / total_raw_detections if total_raw_detections > 0 else 0,
               2
           )
       }
      
       return inventory, stats
  
   def get_inventory_by_category(
       self,
       inventory: Dict[str, InventoryItem]
   ) -> Dict[str, List[InventoryItem]]:
       """
       Organize inventory by category.
      
       Args:
           inventory: Inventory dictionary
          
       Returns:
           Dictionary mapping categories to lists of items
       """
       by_category = defaultdict(list)
       for item in inventory.values():
           by_category[item.category].append(item)
       return dict(by_category)
  
   def get_inventory_summary(
       self,
       inventory: Dict[str, InventoryItem]
   ) -> Dict:
       """
       Generate a summary of the inventory.
      
       Args:
           inventory: Inventory dictionary
          
       Returns:
           Summary dictionary
       """
       by_category = self.get_inventory_by_category(inventory)
      
       summary = {
           "total_items": sum(item.count for item in inventory.values()),
           "unique_types": len(inventory),
           "by_category": {}
       }
      
       for category, items in by_category.items():
           summary["by_category"][category] = {
               "item_count": sum(item.count for item in items),
               "type_count": len(items),
               "items": [item.to_dict() for item in items]
           }
          
       return summary
  
   def to_json(
       self,
       inventory: Dict[str, InventoryItem],
       stats: Dict,
       output_path: Optional[str] = None
   ) -> str:
       """
       Export inventory to JSON format.
      
       Args:
           inventory: Inventory dictionary
           stats: Processing stats
           output_path: Optional file path to save
          
       Returns:
           JSON string
       """
       data = {
           "generated_at": datetime.now().isoformat(),
           "stats": stats,
           "inventory": {
               name: item.to_dict() for name, item in inventory.items()
           },
           "summary": self.get_inventory_summary(inventory)
       }
      
       json_str = json.dumps(data, indent=2)
      
       if output_path:
           with open(output_path, 'w') as f:
               f.write(json_str)
              
       return json_str
  
   def estimate_moving_volume(
       self,
       inventory: Dict[str, InventoryItem]
   ) -> Dict:
       """
       Estimate moving volume based on inventory.
      
       Args:
           inventory: Inventory dictionary
          
       Returns:
           Volume estimation dictionary
       """
       # Approximate cubic feet per item type
       VOLUME_ESTIMATES = {
           "sofa": 40,
           "couch": 40,
           "bed": 60,
           "table": 25,
           "dining table": 25,
           "chair": 8,
           "tv": 10,
           "refrigerator": 30,
           "microwave": 3,
           "oven": 10,
           "laptop": 1,
           "potted plant": 3,
           "clock": 0.5,
           "vase": 0.5,
           "book": 0.2,
           "backpack": 2,
           "suitcase": 4,
       }
      
       total_volume = 0
       item_volumes = {}
      
       for name, item in inventory.items():
           vol_per_item = VOLUME_ESTIMATES.get(name.lower(), 5)  # Default 5 cu ft
           total_vol = vol_per_item * item.count
           item_volumes[name] = {
               "unit_volume_cuft": vol_per_item,
               "count": item.count,
               "total_volume_cuft": total_vol
           }
           total_volume += total_vol
          
       # Estimate truck size
       if total_volume < 100:
           truck_size = "Small (10-12 ft truck)"
           crew_size = "2 persons"
       elif total_volume < 300:
           truck_size = "Medium (16-20 ft truck)"
           crew_size = "2-3 persons"
       elif total_volume < 600:
           truck_size = "Large (22-26 ft truck)"
           crew_size = "3-4 persons"
       else:
           truck_size = "Extra Large (Multiple trucks)"
           crew_size = "4+ persons"
          
       return {
           "total_volume_cuft": round(total_volume, 1),
           "item_volumes": item_volumes,
           "recommended_truck": truck_size,
           "recommended_crew": crew_size,
           "estimated_packing_boxes": max(1, int(total_volume / 3))  # Rough estimate
       }




# Simple aggregation for basic use case (no deduplication)
def simple_aggregate(all_detections: List[List[DetectedObject]]) -> Dict[str, int]:
   """
   Simple aggregation without deduplication.
   Uses basic heuristics to estimate counts.
  
   Args:
       all_detections: List of detection lists from frames
      
   Returns:
       Dictionary of item counts
   """
   # Count occurrences across all frames
   raw_counts = defaultdict(int)
   for frame_detections in all_detections:
       for det in frame_detections:
           raw_counts[det.class_name] += 1
          
   # Estimate actual counts (divide by number of frames, round up)
   num_frames = max(1, len(all_detections))
   estimated_counts = {}
  
   for item, count in raw_counts.items():
       # Use average per frame, but minimum 1
       avg_per_frame = count / num_frames
       # If seen frequently, likely multiple items
       estimated = max(1, round(avg_per_frame * 1.5))  # Slight adjustment
       estimated_counts[item.title()] = estimated
      
   return estimated_counts




if __name__ == "__main__":
   # Test the inventory manager
   from detection import DetectedObject
  
   # Create test detections
   test_detections = [
       [
           DetectedObject("chair", 0.9, (100, 100, 200, 200), 0),
           DetectedObject("sofa", 0.85, (300, 100, 500, 300), 0),
       ],
       [
           DetectedObject("chair", 0.88, (105, 102, 205, 202), 1),  # Same chair
           DetectedObject("sofa", 0.87, (302, 98, 498, 298), 1),   # Same sofa
           DetectedObject("tv", 0.92, (400, 50, 500, 150), 1),
       ],
       [
           DetectedObject("chair", 0.9, (600, 100, 700, 200), 2),  # Different chair
           DetectedObject("tv", 0.9, (402, 52, 502, 152), 2),      # Same TV
       ]
   ]
  
   manager = InventoryManager()
   inventory, stats = manager.generate_inventory_from_detections(test_detections)
  
   print("Stats:", stats)
   print("\nInventory:")
   for name, item in inventory.items():
       print(f"  {item.name}: {item.count} ({item.category})")
  
   # Volume estimation
   volume = manager.estimate_moving_volume(inventory)
   print(f"\nVolume Estimation: {volume['total_volume_cuft']} cu ft")
   print(f"Recommended Truck: {volume['recommended_truck']}")