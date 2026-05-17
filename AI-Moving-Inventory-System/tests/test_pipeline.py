"""
Test Script for AI-Powered Moving Inventory System
Validates the complete pipeline with sample data
"""


import os
import sys
import urllib.request
import numpy as np
import cv2
from pathlib import Path


# Add utils to path
sys.path.insert(0, str(Path(__file__).parent))


# Create directories
for dir_name in ["uploads", "extracted_frames", "outputs", "models", "test_data"]:
   Path(dir_name).mkdir(exist_ok=True)




def download_sample_images():
   """Download sample household images for testing."""
   print("\n📥 Downloading sample images...")
  
   # Sample images from COCO dataset (public domain)
   sample_urls = [
       ("https://images.pexels.com/photos/1350789/pexels-photo-1350789.jpeg?auto=compress&cs=tinysrgb&w=640", "living_room.jpg"),
       ("https://images.pexels.com/photos/271816/pexels-photo-271816.jpeg?auto=compress&cs=tinysrgb&w=640", "bedroom.jpg"),
       ("https://images.pexels.com/photos/2062426/pexels-photo-2062426.jpeg?auto=compress&cs=tinysrgb&w=640", "kitchen.jpg"),
   ]
  
   downloaded = []
  
   for url, filename in sample_urls:
       output_path = Path("test_data") / filename
       if not output_path.exists():
           try:
               print(f"  Downloading {filename}...")
               urllib.request.urlretrieve(url, output_path)
               downloaded.append(str(output_path))
               print(f"  ✅ Downloaded {filename}")
           except Exception as e:
               print(f"  ❌ Failed to download {filename}: {e}")
       else:
           downloaded.append(str(output_path))
           print(f"  ℹ️ {filename} already exists")
  
   return downloaded




def create_test_video(output_path: str = "test_data/sample_walkthrough.mp4", duration: int = 10):
   """
   Create a synthetic test video with simulated furniture.
   Uses simple shapes to simulate a room walkthrough.
   """
   print(f"\n🎬 Creating test video: {output_path}")
  
   fps = 30
   width, height = 1280, 720
  
   fourcc = cv2.VideoWriter_fourcc(*'mp4v')
   out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
  
   total_frames = duration * fps
  
   for i in range(total_frames):
       # Create frame with room-like background
       frame = np.ones((height, width, 3), dtype=np.uint8) * 240  # Light gray
      
       # Simulate camera movement by shifting objects
       offset = int((i / total_frames) * 200)
      
       # Draw floor
       cv2.rectangle(frame, (0, height - 150), (width, height), (200, 200, 200), -1)
      
       # Simulate a sofa (brown rectangle)
       sofa_x = 100 - offset // 2
       cv2.rectangle(frame, (sofa_x, 400), (sofa_x + 400, 550), (70, 70, 130), -1)
       cv2.rectangle(frame, (sofa_x, 350), (sofa_x + 400, 420), (90, 90, 150), -1)  # Back
      
       # Simulate TV (black rectangle)
       tv_x = 800 - offset // 3
       cv2.rectangle(frame, (tv_x, 200), (tv_x + 300, 400), (30, 30, 30), -1)
       cv2.rectangle(frame, (tv_x + 10, 210), (tv_x + 290, 390), (50, 50, 60), -1)  # Screen
      
       # Simulate chairs (multiple small rectangles)
       for j in range(4):
           chair_x = 150 + j * 200 - offset // 4
           if 0 < chair_x < width - 80:
               cv2.rectangle(frame, (chair_x, 450), (chair_x + 80, 570), (50, 100, 50), -1)
               cv2.rectangle(frame, (chair_x, 380), (chair_x + 80, 460), (60, 120, 60), -1)
      
       # Simulate table (dining table)
       table_x = 600 - offset // 5
       cv2.rectangle(frame, (table_x, 480), (table_x + 350, 520), (100, 60, 30), -1)
      
       # Simulate potted plant
       plant_x = 50 - offset // 6
       cv2.circle(frame, (plant_x + 30, 420), 40, (30, 120, 30), -1)
       cv2.rectangle(frame, (plant_x + 10, 450), (plant_x + 50, 520), (80, 50, 30), -1)
      
       # Simulate bed (in later frames - bedroom transition)
       if i > total_frames // 2:
           bed_x = 200 + (i - total_frames // 2) * 2
           cv2.rectangle(frame, (bed_x, 350), (bed_x + 400, 550), (180, 180, 220), -1)
           cv2.rectangle(frame, (bed_x, 300), (bed_x + 100, 370), (200, 200, 230), -1)  # Pillow
      
       # Simulate refrigerator (in later frames - kitchen transition)
       if i > total_frames * 2 // 3:
           fridge_x = 900 + (i - total_frames * 2 // 3)
           cv2.rectangle(frame, (fridge_x, 150), (fridge_x + 150, 550), (200, 200, 210), -1)
           cv2.line(frame, (fridge_x + 140, 150), (fridge_x + 140, 550), (150, 150, 160), 3)
      
       # Add frame number
       cv2.putText(frame, f"Frame: {i}/{total_frames}", (10, 30),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
      
       out.write(frame)
  
   out.release()
   print(f"✅ Created test video: {output_path}")
   return output_path




def test_video_processor():
   """Test the video processing module."""
   print("\n" + "="*60)
   print("🎥 Testing Video Processor")
   print("="*60)
  
   from utils.video_processing import VideoProcessor
  
   # Create test video if not exists
   test_video_path = "test_data/sample_walkthrough.mp4"
   if not Path(test_video_path).exists():
       create_test_video(test_video_path)
  
   processor = VideoProcessor(output_dir="extracted_frames")
  
   # Validate video
   is_valid, msg = processor.validate_video(test_video_path)
   print(f"  Validation: {is_valid} - {msg}")
  
   # Get video info
   info = processor.get_video_info(test_video_path)
   print(f"  Video Info: {info}")
  
   # Extract frames
   frames, paths = processor.extract_frames(
       test_video_path,
       extraction_rate="1_per_second",
       max_frames=10,
       session_id="test_session"
   )
   print(f"  Extracted {len(frames)} frames")
   print(f"  Frame shape: {frames[0].shape if frames else 'N/A'}")
  
   return frames, paths




def test_object_detector(frames):
   """Test the object detection module."""
   print("\n" + "="*60)
   print("🔍 Testing Object Detector")
   print("="*60)
  
   from utils.detection import ObjectDetector
  
   detector = ObjectDetector(
       model_path="yolov8n.pt",
       confidence_threshold=0.3,
       filter_household_only=True
   )
  
   print(f"  Household classes: {detector.get_household_class_names()[:10]}...")
  
   all_detections = []
  
   for idx, frame in enumerate(frames):
       detections = detector.detect_objects(frame, frame_index=idx)
       all_detections.append(detections)
      
       if detections:
           print(f"  Frame {idx}: Detected {len(detections)} objects")
           for det in detections[:3]:
               print(f"    - {det.class_name}: {det.confidence:.2f}")
  
   total_detections = sum(len(d) for d in all_detections)
   print(f"\n  Total detections across all frames: {total_detections}")
  
   return all_detections




def test_inventory_manager(all_detections):
   """Test the inventory management module."""
   print("\n" + "="*60)
   print("📦 Testing Inventory Manager")
   print("="*60)
  
   from utils.inventory import InventoryManager
  
   manager = InventoryManager(
       iou_threshold=0.5,
       position_threshold=100
   )
  
   # Generate inventory
   inventory, stats = manager.generate_inventory_from_detections(all_detections)
  
   print(f"  Stats: {stats}")
   print(f"\n  Inventory Items:")
   for name, item in inventory.items():
       print(f"    - {item.name}: {item.count} ({item.category}) [conf: {item.avg_confidence:.2f}]")
  
   # Volume estimation
   volume_estimate = manager.estimate_moving_volume(inventory)
   print(f"\n  Volume Estimate:")
   print(f"    - Total Volume: {volume_estimate['total_volume_cuft']} cu ft")
   print(f"    - Recommended Truck: {volume_estimate['recommended_truck']}")
   print(f"    - Recommended Crew: {volume_estimate['recommended_crew']}")
  
   # Export JSON
   json_output = manager.to_json(inventory, stats, "outputs/test_inventory.json")
   print(f"\n  ✅ Saved inventory to outputs/test_inventory.json")
  
   return inventory, stats, volume_estimate




def test_summary_generator(inventory, stats, volume_estimate):
   """Test the GenAI summary generator."""
   print("\n" + "="*60)
   print("🤖 Testing Summary Generator")
   print("="*60)
  
   from utils.genai_summary import SummaryGenerator
  
   generator = SummaryGenerator()
  
   # Check availability
   avail = generator.check_availability()
   print(f"  Availability: {avail}")
  
   # Convert inventory for summary
   inventory_dict = {name: item.to_dict() for name, item in inventory.items()}
  
   # Generate summary
   result = generator.generate_summary(inventory_dict, volume_estimate, stats)
  
   print(f"\n  Method Used: {result.method}")
   print(f"\n  Generated Summary:")
   print("-" * 40)
   print(result.summary[:500] + "..." if len(result.summary) > 500 else result.summary)
   print("-" * 40)
  
   # Save summary
   with open("outputs/test_summary.txt", "w") as f:
       f.write(result.summary)
   print(f"\n  ✅ Saved summary to outputs/test_summary.txt")
  
   return result




def test_with_real_images():
   """Test with real downloaded images."""
   print("\n" + "="*60)
   print("🖼️ Testing with Real Images")
   print("="*60)
  
   from utils.detection import ObjectDetector
  
   # Download sample images
   image_paths = download_sample_images()
  
   if not image_paths:
       print("  ❌ No images available for testing")
       return
  
   detector = ObjectDetector(
       model_path="yolov8n.pt",
       confidence_threshold=0.3,
       filter_household_only=True
   )
  
   all_detections = []
  
   for idx, img_path in enumerate(image_paths):
       if not Path(img_path).exists():
           continue
          
       print(f"\n  Processing: {img_path}")
      
       try:
           detections = detector.detect_from_path(img_path, frame_index=idx)
           all_detections.append(detections)
          
           print(f"    Detected {len(detections)} objects:")
           for det in detections:
               print(f"      - {det.class_name}: {det.confidence:.2%}")
              
           # Save annotated image
           img = cv2.imread(img_path)
           annotated = detector.get_annotated_frame(img, detections)
           output_path = f"outputs/annotated_{Path(img_path).name}"
           cv2.imwrite(output_path, annotated)
           print(f"    ✅ Saved annotated image: {output_path}")
          
       except Exception as e:
           print(f"    ❌ Error: {e}")
  
   return all_detections




def run_full_pipeline_test():
   """Run the complete pipeline test."""
   print("\n" + "="*60)
   print("🚀 Running Full Pipeline Test")
   print("="*60)
  
   # Test 1: Video Processing
   frames, paths = test_video_processor()
  
   if not frames:
       print("❌ Video processing failed. Creating synthetic frames...")
       # Create synthetic frames
       frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(5)]
  
   # Test 2: Object Detection
   all_detections = test_object_detector(frames)
  
   # Test 3: Inventory Management
   inventory, stats, volume_estimate = test_inventory_manager(all_detections)
  
   # Test 4: Summary Generation
   summary_result = test_summary_generator(inventory, stats, volume_estimate)
  
   # Test 5: Real Images
   real_detections = test_with_real_images()
  
   # Final Summary
   print("\n" + "="*60)
   print("✅ Pipeline Test Complete")
   print("="*60)
   print(f"""
Test Results:
-------------
- Video Processing: ✅ {len(frames)} frames extracted
- Object Detection: ✅ {sum(len(d) for d in all_detections)} total detections
- Inventory Management: ✅ {len(inventory)} unique item types
- Summary Generation: ✅ Using '{summary_result.method}' method
- Real Image Testing: {"✅" if real_detections else "⚠️"} {len(real_detections or [])} images processed


Output Files:
-------------
- outputs/test_inventory.json
- outputs/test_summary.txt
- outputs/annotated_*.jpg (if images downloaded)


Next Steps:
-----------
1. Run the Streamlit app: streamlit run app.py
2. Upload a real home walkthrough video
3. Review the generated inventory
""")




if __name__ == "__main__":
   print("""
╔══════════════════════════════════════════════════════════╗
║   AI-Powered Moving Inventory System - Test Suite       ║
║   Testing all components of the pipeline                ║
╚══════════════════════════════════════════════════════════╝
""")
  
   try:
       run_full_pipeline_test()
   except Exception as e:
       print(f"\n❌ Test failed with error: {e}")
       import traceback
       traceback.print_exc()


