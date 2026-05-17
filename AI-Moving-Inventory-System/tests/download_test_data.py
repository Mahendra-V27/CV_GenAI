


import os
import urllib.request
import ssl
import cv2
import numpy as np
from pathlib import Path


# Create directories
Path("test_data").mkdir(exist_ok=True)
Path("outputs").mkdir(exist_ok=True)


# SSL context for HTTPS downloads
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE




def download_image(url: str, output_path: str) -> bool:
   """Download image from URL."""
   try:
       headers = {
           'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
       }
       request = urllib.request.Request(url, headers=headers)
       with urllib.request.urlopen(request, context=ssl_context) as response:
           with open(output_path, 'wb') as f:
               f.write(response.read())
       return True
   except Exception as e:
       print(f"  ❌ Failed: {e}")
       return False




def download_test_images():
   """Download sample household images from Unsplash."""
   print("\n📥 Downloading test images from Unsplash...")
  
   # Using Unsplash Source API (free, no auth needed)
   # These are direct links to specific Unsplash images
   test_images = [
       # Living room with furniture
       ("https://images.unsplash.com/photo-1555041469-a586c61ea9bc?w=800", "living_room_1.jpg"),
       # Living room with TV
       ("https://images.unsplash.com/photo-1586023492125-27b2c045efd7?w=800", "living_room_2.jpg"),
       # Bedroom
       ("https://images.unsplash.com/photo-1540518614846-7eded433c457?w=800", "bedroom_1.jpg"),
       # Kitchen with appliances
       ("https://images.unsplash.com/photo-1556909114-f6e7ad7d3136?w=800", "kitchen_1.jpg"),
       # Dining room
       ("https://images.unsplash.com/photo-1617806118233-18e1de247200?w=800", "dining_room_1.jpg"),
       # Office/study
       ("https://images.unsplash.com/photo-1518455027359-f3f8164ba6bd?w=800", "office_1.jpg"),
   ]
  
   downloaded = []
   for url, filename in test_images:
       output_path = f"test_data/{filename}"
       print(f"  Downloading {filename}...")
       if download_image(url, output_path):
           downloaded.append(output_path)
           print(f"  ✅ {filename}")
       else:
           print(f"  ⚠️ Skipped {filename}")
  
   return downloaded




def create_video_from_images(image_paths: list, output_path: str = "test_data/test_walkthrough.mp4"):
   """Create a test video from downloaded images (simulating a walkthrough)."""
   print(f"\n🎬 Creating test video from {len(image_paths)} images...")
  
   if not image_paths:
       print("  ❌ No images available")
       return None
  
   # Load first image to get dimensions
   first_img = cv2.imread(image_paths[0])
   if first_img is None:
       print(f"  ❌ Cannot read {image_paths[0]}")
       return None
  
   # Set video properties
   height, width = first_img.shape[:2]
   target_size = (1280, 720)
   fps = 30
   frames_per_image = fps * 3  # 3 seconds per image
  
   # Create video writer
   fourcc = cv2.VideoWriter_fourcc(*'mp4v')
   out = cv2.VideoWriter(output_path, fourcc, fps, target_size)
  
   for img_path in image_paths:
       img = cv2.imread(img_path)
       if img is None:
           continue
      
       # Resize to target size
       img_resized = cv2.resize(img, target_size)
      
       # Create smooth transition effect
       for i in range(frames_per_image):
           # Add slight zoom/pan effect
           scale = 1.0 + 0.05 * (i / frames_per_image)
           center = (target_size[0] // 2, target_size[1] // 2)
          
           M = cv2.getRotationMatrix2D(center, 0, scale)
           frame = cv2.warpAffine(img_resized, M, target_size)
          
           # Add frame info
           cv2.putText(frame, f"Room View - Testing", (20, 40),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
          
           out.write(frame)
  
   out.release()
   print(f"  ✅ Created video: {output_path}")
   return output_path




def run_detection_on_images(image_paths: list):
   """Run YOLO detection on test images."""
   print("\n🔍 Running object detection on test images...")
  
   import sys
   sys.path.insert(0, str(Path(__file__).parent))
   from utils.detection import ObjectDetector
  
   detector = ObjectDetector(
       model_path="yolov8n.pt",
       confidence_threshold=0.3,
       filter_household_only=False  # Detect all objects for testing
   )
  
   all_detections = {}
  
   for img_path in image_paths:
       if not Path(img_path).exists():
           continue
      
       print(f"\n  📷 {img_path}")
       img = cv2.imread(img_path)
       if img is None:
           print(f"    ❌ Cannot read image")
           continue
      
       # Detect objects
       detections = detector.detect_objects(img, frame_index=0)
       all_detections[img_path] = detections
      
       print(f"    Detected {len(detections)} objects:")
       for det in detections:
           print(f"      - {det.class_name}: {det.confidence:.1%}")
      
       # Save annotated image
       annotated = detector.get_annotated_frame(img, detections)
       output_name = f"outputs/detected_{Path(img_path).name}"
       cv2.imwrite(output_name, annotated)
       print(f"    ✅ Saved: {output_name}")
  
   return all_detections




def create_inventory_report(all_detections: dict):
   """Create inventory report from detections."""
   print("\n📦 Generating inventory report...")
  
   import sys
   sys.path.insert(0, str(Path(__file__).parent))
   from utils.inventory import InventoryManager
   from utils.genai_summary import SummaryGenerator
  
   # Flatten detections
   frame_detections = [list(dets) for dets in all_detections.values()]
  
   # Generate inventory
   manager = InventoryManager()
   inventory, stats = manager.generate_inventory_from_detections(frame_detections)
  
   print(f"\n  Processing Stats:")
   print(f"    - Images processed: {stats['total_frames_processed']}")
   print(f"    - Raw detections: {stats['total_raw_detections']}")
   print(f"    - Unique items: {stats['unique_objects_found']}")
  
   print(f"\n  Inventory:")
   for name, item in inventory.items():
       print(f"    - {item.name}: {item.count} ({item.category})")
  
   # Volume estimation
   volume = manager.estimate_moving_volume(inventory)
   print(f"\n  Moving Estimate:")
   print(f"    - Volume: {volume['total_volume_cuft']} cubic feet")
   print(f"    - Truck: {volume['recommended_truck']}")
   print(f"    - Crew: {volume['recommended_crew']}")
  
   # Generate summary
   generator = SummaryGenerator()
   inventory_dict = {name: item.to_dict() for name, item in inventory.items()}
   summary = generator.generate_summary(inventory_dict, volume, stats)
  
   # Save report
   report_path = "outputs/inventory_report.txt"
   with open(report_path, "w") as f:
       f.write("=" * 60 + "\n")
       f.write("AI-POWERED MOVING INVENTORY REPORT\n")
       f.write("=" * 60 + "\n\n")
       f.write(summary.summary)
  
   print(f"\n  ✅ Report saved: {report_path}")
  
   return inventory, volume, summary




def main():
   """Main test execution."""
   print("""
╔══════════════════════════════════════════════════════════╗
║  AI Moving Inventory - Test Data Download & Validation  ║
╚══════════════════════════════════════════════════════════╝
""")
  
   # Step 1: Download test images
   images = download_test_images()
  
   if not images:
       print("\n⚠️ No images downloaded. Using existing images if available...")
       images = list(Path("test_data").glob("*.jpg"))
       images = [str(p) for p in images]
  
   if not images:
       print("❌ No test images available. Please add images to test_data/ folder.")
       return
  
   # Step 2: Run detection on images
   all_detections = run_detection_on_images(images)
  
   # Step 3: Generate inventory report
   if all_detections:
       inventory, volume, summary = create_inventory_report(all_detections)
  
   # Step 4: Create test video
   video_path = create_video_from_images(images)
  
   print("\n" + "=" * 60)
   print("✅ TEST DATA PREPARATION COMPLETE")
   print("=" * 60)
   print(f"""
Files Created:
--------------
- test_data/*.jpg        - Test images
- test_data/test_walkthrough.mp4 - Test video
- outputs/detected_*.jpg - Annotated images with detections
- outputs/inventory_report.txt - Inventory summary


Next Steps:
-----------
1. Run: streamlit run app.py
2. Upload test_data/test_walkthrough.mp4
3. Or upload your own home walkthrough video
""")




if __name__ == "__main__":
   main()
