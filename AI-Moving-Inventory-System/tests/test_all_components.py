

"""
Comprehensive Test Suite for AI Moving Inventory System V2
Validates all components: interfaces, trackers, depth, room, API
"""


import os
import sys
import unittest
import numpy as np
import cv2
from pathlib import Path
import tempfile
import json


# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


from utils.interfaces import (
   BoundingBox, Detection, TrackedObject, ProcessingConfig,
   ModelType, TrackerType, DepthEstimatorType, RoomClassifierType
)
from utils.model_registry import ModelRegistry, ComponentFactory, DEFAULT_CONFIG




class TestBoundingBox(unittest.TestCase):
   """Test BoundingBox class."""
  
   def test_creation(self):
       bbox = BoundingBox(10, 20, 100, 150)
       self.assertEqual(bbox.x1, 10)
       self.assertEqual(bbox.y1, 20)
       self.assertEqual(bbox.x2, 100)
       self.assertEqual(bbox.y2, 150)
  
   def test_center(self):
       bbox = BoundingBox(0, 0, 100, 100)
       self.assertEqual(bbox.center, (50, 50))
  
   def test_dimensions(self):
       bbox = BoundingBox(10, 20, 110, 170)
       self.assertEqual(bbox.width, 100)
       self.assertEqual(bbox.height, 150)
       self.assertEqual(bbox.area, 15000)
  
   def test_to_tuple(self):
       bbox = BoundingBox(10, 20, 30, 40)
       self.assertEqual(bbox.to_tuple(), (10, 20, 30, 40))




class TestDetection(unittest.TestCase):
   """Test Detection class."""
  
   def test_creation(self):
       bbox = BoundingBox(0, 0, 100, 100)
       det = Detection(
           class_name="sofa",
           class_id=57,
           confidence=0.9,
           bbox=bbox,
           frame_index=0
       )
       self.assertEqual(det.class_name, "sofa")
       self.assertEqual(det.confidence, 0.9)
       self.assertIsNone(det.track_id)
       self.assertIsNone(det.depth)
  
   def test_to_dict(self):
       bbox = BoundingBox(10, 20, 110, 120)
       det = Detection(
           class_name="chair",
           class_id=56,
           confidence=0.85,
           bbox=bbox,
           frame_index=5,
           track_id=1,
           depth=2.5,
           room="living_room"
       )
      
       d = det.to_dict()
       self.assertEqual(d["class_name"], "chair")
       self.assertEqual(d["track_id"], 1)
       self.assertEqual(d["depth"], 2.5)
       self.assertEqual(d["room"], "living_room")




class TestProcessingConfig(unittest.TestCase):
   """Test ProcessingConfig."""
  
   def test_default_config(self):
       config = ProcessingConfig()
       self.assertEqual(config.model_type, ModelType.YOLOV8_NANO)
       self.assertEqual(config.tracker_type, TrackerType.SIMPLE)
       self.assertEqual(config.confidence_threshold, 0.4)
  
   def test_custom_config(self):
       config = ProcessingConfig(
           model_type=ModelType.YOLOV8_MEDIUM,
           tracker_type=TrackerType.DEEPSORT,
           confidence_threshold=0.5,
           max_frames=200
       )
       self.assertEqual(config.model_type, ModelType.YOLOV8_MEDIUM)
       self.assertEqual(config.tracker_type, TrackerType.DEEPSORT)
       self.assertEqual(config.max_frames, 200)




class TestModelRegistry(unittest.TestCase):
   """Test ModelRegistry."""
  
   def test_detector_registration(self):
       # Import to trigger registration
       from utils.detection_v2 import YOLOv8Detector
      
       detectors = ModelRegistry.get_available_detectors()
       self.assertIn(ModelType.YOLOV8_NANO, detectors)
  
   def test_tracker_registration(self):
       from utils.trackers import SimpleTracker, DeepSORTTracker, ByteTracker
      
       trackers = ModelRegistry.get_available_trackers()
       self.assertIn(TrackerType.SIMPLE, trackers)
       self.assertIn(TrackerType.DEEPSORT, trackers)
       self.assertIn(TrackerType.BYTETRACK, trackers)
  
   def test_depth_registration(self):
       from utils.depth_estimation import RuleBasedDepthEstimator
      
       estimators = ModelRegistry.get_available_depth_estimators()
       self.assertIn(DepthEstimatorType.ESTIMATED, estimators)
  
   def test_room_registration(self):
       from utils.room_classifier import RuleBasedRoomClassifier
      
       classifiers = ModelRegistry.get_available_room_classifiers()
       self.assertIn(RoomClassifierType.RULE_BASED, classifiers)




class TestComponentFactory(unittest.TestCase):
   """Test ComponentFactory."""
  
   def test_factory_creation(self):
       config = DEFAULT_CONFIG
       factory = ComponentFactory(config)
       self.assertIsNotNone(factory)
  
   def test_get_detector(self):
       config = ProcessingConfig(model_type=ModelType.YOLOV8_NANO)
       factory = ComponentFactory(config)
      
       detector = factory.get_detector()
       self.assertIsNotNone(detector)
  
   def test_get_tracker(self):
       config = ProcessingConfig(tracker_type=TrackerType.SIMPLE)
       factory = ComponentFactory(config)
      
       tracker = factory.get_tracker()
       self.assertIsNotNone(tracker)
  
   def test_caching(self):
       config = DEFAULT_CONFIG
       factory = ComponentFactory(config)
      
       detector1 = factory.get_detector()
       detector2 = factory.get_detector()
      
       # Should be same cached instance
       self.assertIs(detector1, detector2)




class TestSimpleTracker(unittest.TestCase):
   """Test SimpleTracker."""
  
   def setUp(self):
       from utils.trackers import SimpleTracker
       self.tracker = SimpleTracker(iou_threshold=0.3, max_age=10)
  
   def test_first_frame(self):
       frame = np.zeros((480, 640, 3), dtype=np.uint8)
      
       detections = [
           Detection("sofa", 57, 0.9, BoundingBox(100, 100, 300, 250), 0),
           Detection("chair", 56, 0.85, BoundingBox(400, 150, 500, 300), 0),
       ]
      
       tracked = self.tracker.update(detections, frame)
      
       self.assertEqual(len(tracked), 2)
       self.assertIsNotNone(tracked[0].track_id)
       self.assertIsNotNone(tracked[1].track_id)
  
   def test_tracking_consistency(self):
       frame = np.zeros((480, 640, 3), dtype=np.uint8)
      
       # Frame 1
       det1 = [Detection("sofa", 57, 0.9, BoundingBox(100, 100, 300, 250), 0)]
       tracked1 = self.tracker.update(det1, frame)
       track_id_1 = tracked1[0].track_id
      
       # Frame 2 - same object, slightly moved
       det2 = [Detection("sofa", 57, 0.88, BoundingBox(105, 102, 305, 252), 1)]
       tracked2 = self.tracker.update(det2, frame)
       track_id_2 = tracked2[0].track_id
      
       # Should maintain same track ID
       self.assertEqual(track_id_1, track_id_2)
  
   def test_get_tracks(self):
       frame = np.zeros((480, 640, 3), dtype=np.uint8)
      
       detections = [
           Detection("sofa", 57, 0.9, BoundingBox(100, 100, 300, 250), 0),
       ]
      
       self.tracker.update(detections, frame)
       self.tracker.update(detections, frame)  # Second update for min_hits
      
       tracks = self.tracker.get_tracks()
       self.assertGreaterEqual(len(tracks), 1)
  
   def test_reset(self):
       frame = np.zeros((480, 640, 3), dtype=np.uint8)
       detections = [Detection("sofa", 57, 0.9, BoundingBox(100, 100, 300, 250), 0)]
      
       self.tracker.update(detections, frame)
       self.tracker.reset()
      
       self.assertEqual(len(self.tracker.tracks), 0)




class TestDeepSORTTracker(unittest.TestCase):
   """Test DeepSORT Tracker."""
  
   def setUp(self):
       from utils.trackers import DeepSORTTracker
       self.tracker = DeepSORTTracker(max_age=30, min_hits=3)
  
   def test_update(self):
       frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
      
       detections = [
           Detection("sofa", 57, 0.9, BoundingBox(100, 100, 300, 250), 0),
       ]
      
       tracked = self.tracker.update(detections, frame)
       self.assertEqual(len(tracked), 1)
       self.assertIsNotNone(tracked[0].track_id)
  
   def test_appearance_features(self):
       frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
      
       detections = [
           Detection("sofa", 57, 0.9, BoundingBox(100, 100, 300, 250), 0),
       ]
      
       self.tracker.update(detections, frame)
      
       # Should have stored appearance features
       self.assertGreater(len(self.tracker.appearance_features), 0)




class TestByteTracker(unittest.TestCase):
   """Test ByteTrack."""
  
   def setUp(self):
       from utils.trackers import ByteTracker
       self.tracker = ByteTracker(high_threshold=0.5, low_threshold=0.1)
  
   def test_high_confidence_tracking(self):
       frame = np.zeros((480, 640, 3), dtype=np.uint8)
      
       detections = [
           Detection("sofa", 57, 0.9, BoundingBox(100, 100, 300, 250), 0),
       ]
      
       tracked = self.tracker.update(detections, frame)
       self.assertEqual(len(tracked), 1)
  
   def test_low_confidence_recovery(self):
       frame = np.zeros((480, 640, 3), dtype=np.uint8)
      
       # First frame - high confidence
       det1 = [Detection("sofa", 57, 0.8, BoundingBox(100, 100, 300, 250), 0)]
       self.tracker.update(det1, frame)
      
       # Second frame - low confidence
       det2 = [Detection("sofa", 57, 0.3, BoundingBox(102, 102, 302, 252), 1)]
       tracked = self.tracker.update(det2, frame)
      
       # Low confidence detection should still be tracked
       self.assertEqual(len(tracked), 1)




class TestDepthEstimation(unittest.TestCase):
   """Test Depth Estimation."""
  
   def setUp(self):
       from utils.depth_estimation import RuleBasedDepthEstimator
       self.estimator = RuleBasedDepthEstimator()
  
   def test_depth_map_generation(self):
       frame = np.zeros((480, 640, 3), dtype=np.uint8)
      
       depth_map = self.estimator.estimate_depth(frame)
      
       self.assertEqual(depth_map.shape[:2], frame.shape[:2])
       self.assertGreater(depth_map.max(), 0)
  
   def test_object_depth(self):
       frame = np.zeros((480, 640, 3), dtype=np.uint8)
       depth_map = self.estimator.estimate_depth(frame)
      
       bbox = BoundingBox(100, 200, 300, 400)
       depth = self.estimator.get_object_depth(depth_map, bbox)
      
       self.assertGreater(depth, 0)
  
   def test_volume_estimation(self):
       frame = np.zeros((480, 640, 3), dtype=np.uint8)
       depth_map = self.estimator.estimate_depth(frame)
      
       detection = Detection(
           class_name="sofa",
           class_id=57,
           confidence=0.9,
           bbox=BoundingBox(100, 100, 400, 300),
           frame_index=0
       )
      
       volume = self.estimator.estimate_3d_volume(detection, depth_map)
      
       self.assertGreater(volume, 0)
       self.assertLess(volume, 500)  # Reasonable range




class TestRoomClassifier(unittest.TestCase):
   """Test Room Classification."""
  
   def setUp(self):
       from utils.room_classifier import RuleBasedRoomClassifier
       self.classifier = RuleBasedRoomClassifier()
  
   def test_living_room_classification(self):
       frame = np.zeros((480, 640, 3), dtype=np.uint8)
      
       detections = [
           Detection("sofa", 57, 0.9, BoundingBox(100, 100, 300, 250), 0),
           Detection("tv", 62, 0.85, BoundingBox(400, 50, 550, 200), 0),
       ]
      
       room = self.classifier.classify_room(frame, detections)
      
       self.assertEqual(room, "living_room")
  
   def test_bedroom_classification(self):
       frame = np.zeros((480, 640, 3), dtype=np.uint8)
      
       detections = [
           Detection("bed", 59, 0.9, BoundingBox(100, 100, 400, 350), 0),
       ]
      
       room = self.classifier.classify_room(frame, detections)
      
       self.assertEqual(room, "bedroom")
  
   def test_kitchen_classification(self):
       frame = np.zeros((480, 640, 3), dtype=np.uint8)
      
       detections = [
           Detection("refrigerator", 72, 0.9, BoundingBox(100, 50, 250, 400), 0),
           Detection("microwave", 68, 0.85, BoundingBox(300, 100, 400, 180), 0),
       ]
      
       room = self.classifier.classify_room(frame, detections)
      
       self.assertEqual(room, "kitchen")
  
   def test_unknown_room(self):
       frame = np.zeros((480, 640, 3), dtype=np.uint8)
      
       room = self.classifier.classify_room(frame, [])
      
       self.assertEqual(room, "unknown")
  
   def test_confidence(self):
       frame = np.zeros((480, 640, 3), dtype=np.uint8)
      
       detections = [
           Detection("sofa", 57, 0.9, BoundingBox(100, 100, 300, 250), 0),
       ]
      
       self.classifier.classify_room(frame, detections)
       conf = self.classifier.get_room_confidence()
      
       self.assertGreaterEqual(conf, 0)
       self.assertLessEqual(conf, 1)




class TestYOLOv8Detector(unittest.TestCase):
   """Test YOLOv8 Detector."""
  
   def setUp(self):
       from utils.detection_v2 import YOLOv8Detector
       self.detector = YOLOv8Detector(
           model_type=ModelType.YOLOV8_NANO,
           confidence_threshold=0.3,
           filter_household_only=False
       )
  
   def test_detection(self):
       # Load test image
       test_img_path = "test_data/bus.jpg"
       if Path(test_img_path).exists():
           frame = cv2.imread(test_img_path)
       else:
           frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
      
       detections = self.detector.detect(frame)
      
       # May or may not have detections depending on image
       self.assertIsInstance(detections, list)
  
   def test_model_info(self):
       info = self.detector.get_model_info()
      
       self.assertIn("model_type", info)
       self.assertIn("framework", info)
       self.assertEqual(info["framework"], "ultralytics")
  
   def test_annotated_frame(self):
       frame = np.zeros((480, 640, 3), dtype=np.uint8)
      
       detections = [
           Detection("sofa", 57, 0.9, BoundingBox(100, 100, 300, 250), 0),
       ]
      
       annotated = self.detector.get_annotated_frame(frame, detections)
      
       self.assertEqual(annotated.shape, frame.shape)




class TestInventoryManager(unittest.TestCase):
   """Test Inventory Manager."""
  
   def setUp(self):
       from utils.inventory import InventoryManager
       self.manager = InventoryManager()
  
   def test_deduplication(self):
       from utils.detection import DetectedObject
      
       all_detections = [
           [
               DetectedObject("sofa", 0.9, (100, 100, 300, 250), 0),
               DetectedObject("chair", 0.85, (400, 150, 500, 300), 0),
           ],
           [
               DetectedObject("sofa", 0.88, (102, 102, 302, 252), 1),  # Same sofa
               DetectedObject("chair", 0.87, (402, 152, 502, 302), 1),  # Same chair
           ],
       ]
      
       inventory, stats = self.manager.generate_inventory_from_detections(all_detections)
      
       # Should deduplicate to 1 sofa, 1 chair
       self.assertEqual(inventory["sofa"].count, 1)
       self.assertEqual(inventory["chair"].count, 1)
  
   def test_volume_estimation(self):
       from utils.detection import DetectedObject
      
       all_detections = [
           [
               DetectedObject("sofa", 0.9, (100, 100, 300, 250), 0),
               DetectedObject("bed", 0.85, (400, 150, 700, 400), 0),
           ],
       ]
      
       inventory, stats = self.manager.generate_inventory_from_detections(all_detections)
       volume = self.manager.estimate_moving_volume(inventory)
      
       self.assertIn("total_volume_cuft", volume)
       self.assertIn("recommended_truck", volume)
       self.assertGreater(volume["total_volume_cuft"], 0)




class TestVideoProcessor(unittest.TestCase):
   """Test Video Processor."""
  
   def setUp(self):
       from utils.video_processing import VideoProcessor
       self.processor = VideoProcessor(output_dir=tempfile.mkdtemp())
  
   def test_video_info(self):
       # Create a simple test video
       test_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
      
       fourcc = cv2.VideoWriter_fourcc(*'mp4v')
       out = cv2.VideoWriter(test_video.name, fourcc, 30, (640, 480))
      
       for _ in range(30):
           frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
           out.write(frame)
       out.release()
      
       info = self.processor.get_video_info(test_video.name)
      
       self.assertEqual(info["total_frames"], 30)
       self.assertEqual(info["fps"], 30)
       self.assertEqual(info["width"], 640)
       self.assertEqual(info["height"], 480)
      
       os.unlink(test_video.name)
  
   def test_frame_extraction(self):
       test_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
      
       fourcc = cv2.VideoWriter_fourcc(*'mp4v')
       out = cv2.VideoWriter(test_video.name, fourcc, 30, (640, 480))
      
       for _ in range(60):
           frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
           out.write(frame)
       out.release()
      
       frames, paths = self.processor.extract_frames(
           test_video.name,
           extraction_rate="1_per_second",
           max_frames=10
       )
      
       self.assertGreater(len(frames), 0)
       self.assertLessEqual(len(frames), 10)
      
       os.unlink(test_video.name)




class TestSummaryGenerator(unittest.TestCase):
   """Test Summary Generator."""
  
   def setUp(self):
       from utils.genai_summary import SummaryGenerator
       self.generator = SummaryGenerator()
  
   def test_availability(self):
       avail = self.generator.check_availability()
      
       self.assertIn("template_fallback", avail)
       self.assertTrue(avail["template_fallback"])
  
   def test_template_summary(self):
       inventory = {
           "sofa": {"name": "Sofa", "count": 1, "category": "Furniture", "avg_confidence": 0.9},
           "chair": {"name": "Chair", "count": 4, "category": "Furniture", "avg_confidence": 0.85},
       }
      
       volume = {
           "total_volume_cuft": 100,
           "recommended_truck": "Medium",
           "recommended_crew": "2-3 persons",
           "estimated_packing_boxes": 30
       }
      
       stats = {
           "total_frames_processed": 10,
           "unique_objects_found": 5,
           "unique_item_types": 2
       }
      
       result = self.generator.generate_summary(inventory, volume, stats)
      
       self.assertIn("summary", dir(result))
       self.assertIsNotNone(result.summary)
       self.assertIn(result.method, ["template", "transformers"])




def run_all_tests():
   """Run all tests and print summary."""
   print("""
╔══════════════════════════════════════════════════════════╗
║   AI Moving Inventory V2 - Comprehensive Test Suite     ║
╚══════════════════════════════════════════════════════════╝
""")
  
   # Create test suite
   loader = unittest.TestLoader()
   suite = unittest.TestSuite()
  
   # Add all test classes
   test_classes = [
       TestBoundingBox,
       TestDetection,
       TestProcessingConfig,
       TestModelRegistry,
       TestComponentFactory,
       TestSimpleTracker,
       TestDeepSORTTracker,
       TestByteTracker,
       TestDepthEstimation,
       TestRoomClassifier,
       TestYOLOv8Detector,
       TestInventoryManager,
       TestVideoProcessor,
       TestSummaryGenerator,
   ]
  
   for test_class in test_classes:
       tests = loader.loadTestsFromTestCase(test_class)
       suite.addTests(tests)
  
   # Run tests
   runner = unittest.TextTestRunner(verbosity=2)
   result = runner.run(suite)
  
   # Print summary
   print("\n" + "=" * 60)
   print("TEST SUMMARY")
   print("=" * 60)
   print(f"Tests Run: {result.testsRun}")
   print(f"Failures: {len(result.failures)}")
   print(f"Errors: {len(result.errors)}")
   print(f"Skipped: {len(result.skipped)}")
  
   success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
   print(f"Success Rate: {success_rate:.1f}%")
  
   if result.wasSuccessful():
       print("\n✅ ALL TESTS PASSED!")
   else:
       print("\n❌ SOME TESTS FAILED")
      
       if result.failures:
           print("\nFailures:")
           for test, trace in result.failures:
               print(f"  - {test}")
      
       if result.errors:
           print("\nErrors:")
           for test, trace in result.errors:
               print(f"  - {test}")
  
   return result




if __name__ == "__main__":
   run_all_tests()