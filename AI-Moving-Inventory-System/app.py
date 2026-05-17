"""
AI-Powered Moving Inventory Generation System
Main Streamlit Application
"""


import streamlit as st
import os
import sys
import uuid
import tempfile
from pathlib import Path
import pandas as pd
import time
import json
from datetime import datetime


# Add utils to path
sys.path.insert(0, str(Path(__file__).parent))


from utils.video_processing import VideoProcessor
from utils.detection import ObjectDetector, ITEM_CATEGORIES
from utils.inventory import InventoryManager
from utils.genai_summary import SummaryGenerator


# Page configuration
st.set_page_config(
   page_title="AI Moving Inventory",
   page_icon="📦",
   layout="wide",
   initial_sidebar_state="expanded"
)


# Custom CSS for modern UI
st.markdown("""
<style>
   /* Main container styling */
   .main > div {
       padding-top: 2rem;
   }
  
   /* Header styling */
   .main-header {
       background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
       padding: 2rem;
       border-radius: 15px;
       color: white;
       text-align: center;
       margin-bottom: 2rem;
   }
  
   .main-header h1 {
       margin: 0;
       font-size: 2.5rem;
   }
  
   .main-header p {
       margin: 0.5rem 0 0 0;
       opacity: 0.9;
   }
  
   /* Card styling */
   .stat-card {
       background: white;
       padding: 1.5rem;
       border-radius: 10px;
       box-shadow: 0 2px 10px rgba(0,0,0,0.1);
       text-align: center;
       border-left: 4px solid #667eea;
   }
  
   .stat-card h3 {
       margin: 0;
       font-size: 2rem;
       color: #667eea;
   }
  
   .stat-card p {
       margin: 0.5rem 0 0 0;
       color: #666;
   }
  
   /* Inventory table styling */
   .inventory-table {
       width: 100%;
       border-collapse: collapse;
   }
  
   .inventory-table th {
       background: #667eea;
       color: white;
       padding: 12px;
       text-align: left;
   }
  
   .inventory-table td {
       padding: 10px;
       border-bottom: 1px solid #eee;
   }
  
   /* Category badges */
   .category-badge {
       display: inline-block;
       padding: 4px 12px;
       border-radius: 20px;
       font-size: 0.85rem;
       font-weight: 500;
   }
  
   .category-furniture { background: #e3f2fd; color: #1565c0; }
   .category-electronics { background: #fce4ec; color: #c2185b; }
   .category-kitchen { background: #fff3e0; color: #ef6c00; }
   .category-decor { background: #e8f5e9; color: #2e7d32; }
   .category-storage { background: #f3e5f5; color: #7b1fa2; }
  
   /* Summary box */
   .summary-box {
       background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
       padding: 1.5rem;
       border-radius: 10px;
       margin: 1rem 0;
   }
  
   /* Progress styling */
   .stProgress > div > div > div > div {
       background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
   }
  
   /* Button styling */
   .stButton > button {
       background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
       color: white;
       border: none;
       padding: 0.75rem 2rem;
       border-radius: 25px;
       font-weight: 600;
       transition: transform 0.2s;
   }
  
   .stButton > button:hover {
       transform: scale(1.02);
   }
  
   /* Sidebar styling */
   .css-1d391kg {
       background-color: #f8f9fa;
   }
  
   /* Hide Streamlit branding */
   #MainMenu {visibility: hidden;}
   footer {visibility: hidden;}
  
   /* Detection result card */
   .detection-card {
       background: white;
       border-radius: 10px;
       overflow: hidden;
       box-shadow: 0 2px 10px rgba(0,0,0,0.1);
       margin: 0.5rem 0;
   }
  
   .detection-card img {
       width: 100%;
       height: auto;
   }
</style>
""", unsafe_allow_html=True)




# Initialize session state
def init_session_state():
   """Initialize session state variables."""
   if 'processed' not in st.session_state:
       st.session_state.processed = False
   if 'inventory' not in st.session_state:
       st.session_state.inventory = None
   if 'stats' not in st.session_state:
       st.session_state.stats = None
   if 'volume_estimate' not in st.session_state:
       st.session_state.volume_estimate = None
   if 'summary' not in st.session_state:
       st.session_state.summary = None
   if 'annotated_frames' not in st.session_state:
       st.session_state.annotated_frames = []
   if 'session_id' not in st.session_state:
       st.session_state.session_id = str(uuid.uuid4())[:8]




def render_header():
   """Render the main header."""
   st.markdown("""
   <div class="main-header">
       <h1>📦 AI Moving Inventory System</h1>
       <p>Upload a home walkthrough video to automatically generate your moving inventory</p>
   </div>
   """, unsafe_allow_html=True)




def render_sidebar():
   """Render sidebar with settings."""
   with st.sidebar:
       st.markdown("## ⚙️ Settings")
      
       st.markdown("### Video Processing")
       extraction_rate = st.selectbox(
           "Frame Extraction Rate",
           options=["1_per_second", "2_per_second", "every_10_frames", "every_30_frames"],
           format_func=lambda x: {
               "1_per_second": "1 frame per second",
               "2_per_second": "2 frames per second",
               "every_10_frames": "Every 10 frames",
               "every_30_frames": "Every 30 frames"
           }[x],
           index=0
       )
      
       max_frames = st.slider(
           "Maximum Frames to Process",
           min_value=10,
           max_value=200,
           value=50,
           step=10
       )
      
       st.markdown("### Object Detection")
       confidence_threshold = st.slider(
           "Detection Confidence Threshold",
           min_value=0.2,
           max_value=0.9,
           value=0.4,
           step=0.05
       )
      
       filter_household = st.checkbox(
           "Detect Household Items Only",
           value=True,
           help="Filter to only detect furniture, electronics, and household items"
       )
      
       st.markdown("### Deduplication")
       iou_threshold = st.slider(
           "IoU Threshold",
           min_value=0.3,
           max_value=0.8,
           value=0.5,
           step=0.1,
           help="Higher values require more overlap to consider objects as same"
       )
      
       st.markdown("---")
      
       # GenAI status
       st.markdown("### 🤖 GenAI Status")
       generator = SummaryGenerator()
       avail = generator.check_availability()
      
       if avail["ollama_running"]:
           st.success("✅ Ollama (Local LLM)")
       elif avail["transformers_installed"]:
           st.warning("⚠️ Using Transformers")
       else:
           st.info("📝 Using Template Mode")
      
       st.markdown("---")
       st.markdown("### 📋 About")
       st.markdown("""
       This application uses:
       - **YOLOv8** for object detection
       - **OpenCV** for video processing
       - **Ollama/LLM** for summary generation
      
       [View Documentation](#)
       """)
      
   return {
       "extraction_rate": extraction_rate,
       "max_frames": max_frames,
       "confidence_threshold": confidence_threshold,
       "filter_household": filter_household,
       "iou_threshold": iou_threshold
   }




def process_video(video_path: str, settings: dict, progress_placeholder):
   """Process the uploaded video and generate inventory."""
  
   # Initialize components
   video_processor = VideoProcessor(output_dir="extracted_frames")
   detector = ObjectDetector(
       model_path="yolov8n.pt",
       confidence_threshold=settings["confidence_threshold"],
       filter_household_only=settings["filter_household"]
   )
   inventory_manager = InventoryManager(
       iou_threshold=settings["iou_threshold"]
   )
   summary_generator = SummaryGenerator()
  
   # Step 1: Validate video
   progress_placeholder.progress(5, text="Validating video...")
   is_valid, msg = video_processor.validate_video(video_path)
   if not is_valid:
       st.error(f"Video validation failed: {msg}")
       return None
  
   # Get video info
   video_info = video_processor.get_video_info(video_path)
  
   # Step 2: Extract frames
   progress_placeholder.progress(15, text="Extracting frames from video...")
   frames, frame_paths = video_processor.extract_frames(
       video_path,
       extraction_rate=settings["extraction_rate"],
       max_frames=settings["max_frames"],
       session_id=st.session_state.session_id
   )
  
   if not frames:
       st.error("No frames could be extracted from the video.")
       return None
  
   # Step 3: Run object detection
   progress_placeholder.progress(30, text=f"Detecting objects in {len(frames)} frames...")
  
   all_detections = []
   annotated_frames = []
  
   for idx, frame in enumerate(frames):
       # Update progress
       progress = 30 + int((idx / len(frames)) * 40)
       progress_placeholder.progress(
           progress,
           text=f"Processing frame {idx + 1}/{len(frames)}..."
       )
      
       # Detect objects
       detections = detector.detect_objects(frame, frame_index=idx)
       all_detections.append(detections)
      
       # Create annotated frame (for first few frames only)
       if idx < 5 and detections:
           annotated = detector.get_annotated_frame(frame, detections)
           annotated_frames.append(annotated)
  
   # Step 4: Generate inventory
   progress_placeholder.progress(75, text="Aggregating and deduplicating detections...")
   inventory, stats = inventory_manager.generate_inventory_from_detections(all_detections)
  
   # Step 5: Estimate volume
   progress_placeholder.progress(85, text="Estimating moving volume...")
   volume_estimate = inventory_manager.estimate_moving_volume(inventory)
  
   # Step 6: Generate AI summary
   progress_placeholder.progress(95, text="Generating AI summary...")
  
   # Convert inventory to dict format for summary generator
   inventory_dict = {
       name: item.to_dict() for name, item in inventory.items()
   }
  
   summary_result = summary_generator.generate_summary(
       inventory_dict,
       volume_estimate,
       stats
   )
  
   progress_placeholder.progress(100, text="Complete!")
  
   # Store results
   results = {
       "inventory": inventory,
       "stats": stats,
       "volume_estimate": volume_estimate,
       "summary": summary_result,
       "video_info": video_info,
       "annotated_frames": annotated_frames
   }
  
   return results




def render_results(results: dict):
   """Render the processing results."""
   inventory = results["inventory"]
   stats = results["stats"]
   volume_estimate = results["volume_estimate"]
   summary = results["summary"]
   video_info = results["video_info"]
   annotated_frames = results["annotated_frames"]
  
   # Stats cards
   st.markdown("## 📊 Processing Statistics")
  
   col1, col2, col3, col4 = st.columns(4)
  
   with col1:
       st.metric(
           label="Frames Processed",
           value=stats.get("total_frames_processed", 0)
       )
  
   with col2:
       st.metric(
           label="Objects Detected",
           value=stats.get("total_raw_detections", 0)
       )
  
   with col3:
       st.metric(
           label="Unique Items",
           value=stats.get("unique_objects_found", 0)
       )
  
   with col4:
       st.metric(
           label="Item Types",
           value=stats.get("unique_item_types", 0)
       )
  
   st.markdown("---")
  
   # Detection preview
   if annotated_frames:
       st.markdown("## 🔍 Detection Preview")
       cols = st.columns(min(len(annotated_frames), 5))
       for idx, frame in enumerate(annotated_frames[:5]):
           with cols[idx]:
               st.image(frame, caption=f"Frame {idx + 1}", use_container_width=True)
  
   st.markdown("---")
  
   # Main content - 2 columns
   col_left, col_right = st.columns([3, 2])
  
   with col_left:
       st.markdown("## 📦 Inventory Items")
      
       if inventory:
           # Create DataFrame for display
           data = []
           for name, item in inventory.items():
               data.append({
                   "Item": item.name,
                   "Count": item.count,
                   "Category": item.category,
                   "Confidence": f"{item.avg_confidence:.0%}"
               })
          
           df = pd.DataFrame(data)
          
           # Style the dataframe
           st.dataframe(
               df,
               use_container_width=True,
               hide_index=True,
               column_config={
                   "Item": st.column_config.TextColumn("Item", width="medium"),
                   "Count": st.column_config.NumberColumn("Count", width="small"),
                   "Category": st.column_config.TextColumn("Category", width="medium"),
                   "Confidence": st.column_config.TextColumn("Confidence", width="small")
               }
           )
          
           # Category breakdown
           st.markdown("### By Category")
          
           categories = {}
           for item in inventory.values():
               cat = item.category
               if cat not in categories:
                   categories[cat] = {"count": 0, "items": []}
               categories[cat]["count"] += item.count
               categories[cat]["items"].append(item.name)
          
           for cat, info in categories.items():
               with st.expander(f"**{cat}** ({info['count']} items)"):
                   st.write(", ".join(info["items"]))
       else:
           st.warning("No household items detected in the video.")
  
   with col_right:
       st.markdown("## 🚚 Moving Estimate")
      
       # Volume estimate card
       st.markdown(f"""
       <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                   padding: 1.5rem; border-radius: 10px; color: white; margin-bottom: 1rem;">
           <h3 style="margin: 0;">Estimated Volume</h3>
           <h1 style="margin: 0.5rem 0;">{volume_estimate.get('total_volume_cuft', 0)} cu ft</h1>
       </div>
       """, unsafe_allow_html=True)
      
       st.markdown("### Recommendations")
       st.markdown(f"**🚛 Truck Size:** {volume_estimate.get('recommended_truck', 'N/A')}")
       st.markdown(f"**👥 Crew Size:** {volume_estimate.get('recommended_crew', 'N/A')}")
       st.markdown(f"**📦 Est. Boxes:** {volume_estimate.get('estimated_packing_boxes', 0)}")
      
       # Video info
       st.markdown("### Video Details")
       st.markdown(f"- **Duration:** {video_info.get('duration_seconds', 0):.1f} seconds")
       st.markdown(f"- **Resolution:** {video_info.get('width', 0)}x{video_info.get('height', 0)}")
       st.markdown(f"- **FPS:** {video_info.get('fps', 0):.0f}")
  
   st.markdown("---")
  
   # AI Summary
   st.markdown("## 🤖 AI-Generated Summary")
  
   method_badge = {
       "ollama": "🟢 Ollama (Local LLM)",
       "transformers": "🟡 HuggingFace",
       "template": "🔵 Template"
   }
  
   st.caption(f"Generated using: {method_badge.get(summary.method, summary.method)}")
  
   st.markdown(summary.summary)
  
   st.markdown("---")
  
   # Export options
   st.markdown("## 💾 Export Options")
  
   col_exp1, col_exp2, col_exp3 = st.columns(3)
  
   with col_exp1:
       # JSON export
       export_data = {
           "generated_at": datetime.now().isoformat(),
           "video_info": video_info,
           "stats": stats,
           "inventory": {name: item.to_dict() for name, item in inventory.items()},
           "volume_estimate": volume_estimate,
           "summary": summary.summary
       }
      
       json_str = json.dumps(export_data, indent=2)
       st.download_button(
           label="📥 Download JSON",
           data=json_str,
           file_name=f"inventory_{st.session_state.session_id}.json",
           mime="application/json"
       )
  
   with col_exp2:
       # CSV export
       if inventory:
           csv_data = df.to_csv(index=False)
           st.download_button(
               label="📥 Download CSV",
               data=csv_data,
               file_name=f"inventory_{st.session_state.session_id}.csv",
               mime="text/csv"
           )
  
   with col_exp3:
       # Summary text export
       st.download_button(
           label="📥 Download Summary",
           data=summary.summary,
           file_name=f"summary_{st.session_state.session_id}.txt",
           mime="text/plain"
       )




def main():
   """Main application entry point."""
   init_session_state()
   render_header()
   settings = render_sidebar()
  
   # Main content area
   st.markdown("## 📤 Upload Video")
  
   # File uploader
   uploaded_file = st.file_uploader(
       "Choose a home walkthrough video",
       type=["mp4", "mov", "avi", "mkv"],
       help="Upload a video of your home walkthrough. Supported formats: MP4, MOV, AVI, MKV"
   )
  
   if uploaded_file is not None:
       # Display video info
       col1, col2 = st.columns([2, 1])
      
       with col1:
           st.video(uploaded_file)
      
       with col2:
           st.markdown("### Video Info")
           st.markdown(f"**Filename:** {uploaded_file.name}")
           st.markdown(f"**Size:** {uploaded_file.size / (1024*1024):.2f} MB")
           st.markdown(f"**Type:** {uploaded_file.type}")
      
       # Process button
       st.markdown("---")
      
       if st.button("🚀 Generate Inventory", type="primary", use_container_width=True):
           # Reset session state
           st.session_state.session_id = str(uuid.uuid4())[:8]
           st.session_state.processed = False
          
           # Save uploaded file temporarily
           with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
               tmp_file.write(uploaded_file.getvalue())
               tmp_path = tmp_file.name
          
           try:
               # Create progress placeholder
               progress_placeholder = st.progress(0, text="Starting processing...")
              
               # Process video
               with st.spinner("Processing video..."):
                   results = process_video(tmp_path, settings, progress_placeholder)
              
               if results:
                   st.session_state.processed = True
                   st.session_state.results = results
                  
                   # Clear progress
                   progress_placeholder.empty()
                  
                   # Show success message
                   st.success("✅ Inventory generated successfully!")
                  
                   # Render results
                   render_results(results)
                  
           except Exception as e:
               st.error(f"Error processing video: {str(e)}")
               import traceback
               st.code(traceback.format_exc())
          
           finally:
               # Cleanup temp file
               if os.path.exists(tmp_path):
                   os.unlink(tmp_path)
  
   else:
       # Show demo/instructions
       st.markdown("""
       <div style="background: #f8f9fa; padding: 2rem; border-radius: 10px; text-align: center;">
           <h3>📹 How it works</h3>
           <p style="color: #666;">
               1. Upload a walkthrough video of your home<br>
               2. Our AI will detect and count household items<br>
               3. Get an instant inventory with moving estimates
           </p>
           <br>
           <p style="font-size: 0.9rem; color: #888;">
               <strong>Tip:</strong> For best results, walk slowly through each room
               and ensure good lighting.
           </p>
       </div>
       """, unsafe_allow_html=True)
      
       # Sample items that can be detected
       st.markdown("### 🏠 Detectable Items")
      
       col1, col2, col3, col4 = st.columns(4)
      
       with col1:
           st.markdown("**Furniture**")
           st.markdown("- Sofa / Couch\n- Chair\n- Bed\n- Dining Table\n- Desk")
      
       with col2:
           st.markdown("**Electronics**")
           st.markdown("- TV\n- Laptop\n- Keyboard\n- Cell Phone\n- Monitor")
      
       with col3:
           st.markdown("**Kitchen**")
           st.markdown("- Refrigerator\n- Microwave\n- Oven\n- Toaster\n- Sink")
      
       with col4:
           st.markdown("**Storage & Decor**")
           st.markdown("- Suitcase\n- Backpack\n- Potted Plant\n- Clock\n- Vase")




if __name__ == "__main__":
   main()