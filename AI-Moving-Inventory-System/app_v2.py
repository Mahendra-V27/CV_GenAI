"""
AI-Powered Moving Inventory Generation System V2
Enhanced Streamlit Application with Advanced Features
"""


import streamlit as st
import os
import sys
import uuid
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
import json
from datetime import datetime
from typing import List, Dict, Optional


# Add utils to path
sys.path.insert(0, str(Path(__file__).parent))


from utils.interfaces import (
   ProcessingConfig, ModelType, TrackerType,
   DepthEstimatorType, RoomClassifierType, SummaryGeneratorType,
   Detection, FrameResult
)
from utils.model_registry import ModelRegistry, ComponentFactory, DEFAULT_CONFIG
from utils.video_processing import VideoProcessor
from utils.detection import ITEM_CATEGORIES


# Page configuration
st.set_page_config(
   page_title="AI Moving Inventory V2",
   page_icon="📦",
   layout="wide",
   initial_sidebar_state="expanded"
)


# Custom CSS
st.markdown("""
<style>
   .main > div { padding-top: 1rem; }
  
   .main-header {
       background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
       padding: 1.5rem;
       border-radius: 15px;
       color: white;
       text-align: center;
       margin-bottom: 1.5rem;
   }
  
   .config-section {
       background: #f8f9fa;
       padding: 1rem;
       border-radius: 10px;
       margin-bottom: 1rem;
   }
  
   .frame-gallery {
       display: flex;
       flex-wrap: wrap;
       gap: 10px;
   }
  
   .frame-card {
       border: 2px solid #ddd;
       border-radius: 8px;
       overflow: hidden;
       cursor: pointer;
       transition: border-color 0.2s;
   }
  
   .frame-card:hover {
       border-color: #667eea;
   }
  
   .frame-card.selected {
       border-color: #764ba2;
       box-shadow: 0 0 10px rgba(118, 75, 162, 0.3);
   }
  
   .stat-card {
       background: white;
       padding: 1rem;
       border-radius: 10px;
       box-shadow: 0 2px 8px rgba(0,0,0,0.1);
       text-align: center;
       border-left: 4px solid #667eea;
   }
  
   .model-badge {
       display: inline-block;
       padding: 4px 12px;
       border-radius: 20px;
       font-size: 0.8rem;
       margin: 2px;
   }
            
   video {
        width: 400px !important;
        height: 500px !important;
        display: block;
        margin: auto;
    }
  
   .model-badge.active { background: #d4edda; color: #155724; }
   .model-badge.inactive { background: #f8d7da; color: #721c24; }
  
   #MainMenu {visibility: hidden;}
   footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)




def init_session_state():
   """Initialize session state."""
   defaults = {
       'processed': False,
       'results': None,
       'all_frames': [],
       'all_annotated_frames': [],
       'selected_frames': [],
       'session_id': str(uuid.uuid4())[:8],
       'config': DEFAULT_CONFIG,
       'frame_idx': 0,
       'selection_method': 'Slider',
       'indices_input': '0, 5, 10',
       'range_start': 0,
       'range_end': 5,
       'range_step': 1,
   }
   for key, value in defaults.items():
       if key not in st.session_state:
           st.session_state[key] = value




def render_header():
   """Render header."""
   st.markdown("""
   <div class="main-header">
       <h1>📦 AI Moving Inventory System V2</h1>
       <p>Advanced Computer Vision + GenAI with Multi-Model Support</p>
   </div>
   """, unsafe_allow_html=True)




def render_model_config():
   """Render model configuration sidebar."""
   with st.sidebar:
       st.markdown("## ⚙️ Configuration")
      
       # Load available models
       try:
           # Import to register models
           from utils.detection_v2 import YOLOv8Detector
           from utils.trackers import SimpleTracker, DeepSORTTracker, ByteTracker
           from utils.depth_estimation import RuleBasedDepthEstimator
           from utils.room_classifier import RuleBasedRoomClassifier
       except Exception as e:
           st.warning(f"Some modules not loaded: {e}")
      
       available_detectors = ModelRegistry.get_available_detectors()
       available_trackers = ModelRegistry.get_available_trackers()
       available_depth = ModelRegistry.get_available_depth_estimators()
       available_room = ModelRegistry.get_available_room_classifiers()
      
       # Detection Model
       st.markdown("### 🔍 Detection Model")
       detector_options = {k.value: v.name for k, v in available_detectors.items()}
       selected_detector = st.selectbox(
           "Model",
           options=list(detector_options.keys()),
           format_func=lambda x: detector_options.get(x, x),
           index=0,
           help="YOLOv8 variants: n=nano (fastest), s=small, m=medium, l=large, x=xlarge (most accurate)"
       )
      
       confidence = st.slider(
           "Confidence Threshold",
           min_value=0.1, max_value=0.9, value=0.4, step=0.05,
           help="Higher = fewer but more confident detections"
       )
      
       filter_household = st.checkbox("Household Items Only", value=True)
      
       # Tracking
       st.markdown("### 🎯 Object Tracking")
       tracker_options = {"none": "None (Basic Dedup)"} | {k.value: v.name for k, v in available_trackers.items()}
       selected_tracker = st.selectbox(
           "Tracker",
           options=list(tracker_options.keys()),
           format_func=lambda x: tracker_options.get(x, x),
           index=1,  # Default to simple
           help="Advanced tracking prevents duplicate counting"
       )
      
       # Depth Estimation
       st.markdown("### 📏 3D Volume Estimation")
       depth_options = {"none": "Disabled"} | {k.value: v.name for k, v in available_depth.items()}
       selected_depth = st.selectbox(
           "Depth Estimator",
           options=list(depth_options.keys()),
           format_func=lambda x: depth_options.get(x, x),
           index=1,  # Default to estimated
       )
      
       # Room Classification
       st.markdown("### 🏠 Room Classification")
       room_options = {"none": "Disabled"} | {k.value: v.name for k, v in available_room.items()}
       selected_room = st.selectbox(
           "Room Classifier",
           options=list(room_options.keys()),
           format_func=lambda x: room_options.get(x, x),
           index=1,
       )
      
       # Video Processing
       st.markdown("### 🎬 Video Processing")
       extraction_rate = st.selectbox(
           "Frame Extraction Rate",
           options=["1_per_second", "2_per_second", "every_10_frames", "every_30_frames"],
           format_func=lambda x: {
               "1_per_second": "1 frame/second",
               "2_per_second": "2 frames/second",
               "every_10_frames": "Every 10 frames",
               "every_30_frames": "Every 30 frames"
           }.get(x, x),
           index=0
       )
      
       max_frames = st.slider("Max Frames", 10, 200, 50, 10)
      
       # Preview settings
       st.markdown("### 🖼️ Preview Settings")
       preview_count = st.slider(
           "Default Preview Frames",
           min_value=1, max_value=20, value=5,
           help="Number of frames to show in preview"
       )
      
       st.markdown("---")
      
       # Model status
       st.markdown("### 📊 Model Status")
       cols = st.columns(2)
       with cols[0]:
           st.markdown(f"**Detectors:** {len(available_detectors)}")
           st.markdown(f"**Trackers:** {len(available_trackers)}")
       with cols[1]:
           st.markdown(f"**Depth:** {len(available_depth)}")
           st.markdown(f"**Room:** {len(available_room)}")
      
       # Build config
       model_type_map = {
           "yolov8n": ModelType.YOLOV8_NANO,
           "yolov8s": ModelType.YOLOV8_SMALL,
           "yolov8m": ModelType.YOLOV8_MEDIUM,
           "yolov8l": ModelType.YOLOV8_LARGE,
           "yolov8x": ModelType.YOLOV8_XLARGE,
           "onnx": ModelType.ONNX,
       }
      
       tracker_type_map = {
           "none": TrackerType.NONE,
           "simple": TrackerType.SIMPLE,
           "deepsort": TrackerType.DEEPSORT,
           "bytetrack": TrackerType.BYTETRACK,
       }
      
       depth_type_map = {
           "none": DepthEstimatorType.NONE,
           "estimated": DepthEstimatorType.ESTIMATED,
           "midas": DepthEstimatorType.MIDAS,
           "depth_anything": DepthEstimatorType.DEPTH_ANYTHING,
       }
      
       room_type_map = {
           "none": RoomClassifierType.NONE,
           "rule_based": RoomClassifierType.RULE_BASED,
           "clip": RoomClassifierType.CLIP,
           "cnn": RoomClassifierType.CNN,
       }
      
       config = ProcessingConfig(
           model_type=model_type_map.get(selected_detector, ModelType.YOLOV8_NANO),
           confidence_threshold=confidence,
           filter_household_only=filter_household,
           tracker_type=tracker_type_map.get(selected_tracker, TrackerType.SIMPLE),
           depth_estimator_type=depth_type_map.get(selected_depth, DepthEstimatorType.ESTIMATED),
           room_classifier_type=room_type_map.get(selected_room, RoomClassifierType.RULE_BASED),
           extraction_rate=extraction_rate,
           max_frames=max_frames,
           preview_frame_count=preview_count,
       )
      
       return config




def process_video_v2(video_path: str, config: ProcessingConfig, progress_placeholder):
   """Process video with V2 pipeline."""
  
   # Initialize components
   factory = ComponentFactory(config)
   video_processor = VideoProcessor(output_dir="extracted_frames")
  
   # Step 1: Validate and get info
   progress_placeholder.progress(5, text="Validating video...")
   is_valid, msg = video_processor.validate_video(video_path)
   if not is_valid:
       st.error(f"Video validation failed: {msg}")
       return None
  
   video_info = video_processor.get_video_info(video_path)
  
   # Step 2: Extract frames
   progress_placeholder.progress(15, text="Extracting frames...")
   frames, frame_paths = video_processor.extract_frames(
       video_path,
       extraction_rate=config.extraction_rate,
       max_frames=config.max_frames,
       session_id=st.session_state.session_id
   )
  
   if not frames:
       st.error("No frames extracted")
       return None
  
   # Step 3: Get components
   progress_placeholder.progress(20, text="Initializing models...")
  
   try:
       from utils.detection_v2 import YOLOv8Detector
       detector = YOLOv8Detector(
           model_type=config.model_type,
           confidence_threshold=config.confidence_threshold,
           filter_household_only=config.filter_household_only
       )
   except Exception as e:
       st.warning(f"Using fallback detector: {e}")
       from utils.detection import ObjectDetector
       detector = ObjectDetector(
           confidence_threshold=config.confidence_threshold,
           filter_household_only=config.filter_household_only
       )
  
   tracker = factory.get_tracker()
   depth_estimator = factory.get_depth_estimator()
   room_classifier = factory.get_room_classifier()
  
   # Step 4: Process frames
   all_detections = []
   all_annotated = []
   room_labels = []
   frame_results = []
  
   for idx, frame in enumerate(frames):
       progress = 20 + int((idx / len(frames)) * 50)
       progress_placeholder.progress(progress, text=f"Processing frame {idx+1}/{len(frames)}...")
      
       # Detect
       if hasattr(detector, 'detect'):
           detections = detector.detect(frame, frame_index=idx)
       else:
           # Old detector
           from utils.interfaces import Detection, BoundingBox
           old_dets = detector.detect_objects(frame, frame_index=idx)
           detections = [
               Detection(
                   class_name=d.class_name,
                   class_id=0,
                   confidence=d.confidence,
                   bbox=BoundingBox(*d.bbox),
                   frame_index=d.frame_index
               )
               for d in old_dets
           ]
      
       # Track
       if tracker:
           detections = tracker.update(detections, frame)
      
       # Depth
       depth_map = None
       if depth_estimator:
           depth_map = depth_estimator.estimate_depth(frame)
           for det in detections:
               det.depth = depth_estimator.get_object_depth(depth_map, det.bbox)
               det.volume_3d = depth_estimator.estimate_3d_volume(det, depth_map)
      
       # Room
       room_label = None
       if room_classifier:
           room_label = room_classifier.classify_room(frame, detections)
           room_labels.append(room_label)
           for det in detections:
               det.room = room_label
      
       all_detections.append(detections)
      
       # Annotate
       if hasattr(detector, 'get_annotated_frame'):
           annotated = detector.get_annotated_frame(
               frame, detections,
               show_depth=depth_estimator is not None,
               show_room=room_classifier is not None
           )
       else:
           # Use old method
           from utils.detection import DetectedObject
           old_dets = [
               DetectedObject(
                   class_name=d.class_name,
                   confidence=d.confidence,
                   bbox=d.bbox.to_tuple() if hasattr(d.bbox, 'to_tuple') else d.bbox,
                   frame_index=d.frame_index
               )
               for d in detections
           ]
           annotated = detector.get_annotated_frame(frame, old_dets)
      
       all_annotated.append(annotated)
      
       frame_results.append(FrameResult(
           frame_index=idx,
           detections=detections,
           annotated_frame=annotated,
           depth_map=depth_map,
           room_label=room_label
       ))
  
   # Step 5: Generate inventory
   progress_placeholder.progress(75, text="Generating inventory...")
  
   from utils.inventory import InventoryManager
   from utils.detection import DetectedObject
  
   inventory_manager = InventoryManager()
  
   # Convert to old format
   old_format = []
   for frame_dets in all_detections:
       old_frame = []
       for d in frame_dets:
           old_frame.append(DetectedObject(
               class_name=d.class_name,
               confidence=d.confidence,
               bbox=d.bbox.to_tuple() if hasattr(d.bbox, 'to_tuple') else d.bbox,
               frame_index=d.frame_index
           ))
       old_format.append(old_frame)
  
   inventory, stats = inventory_manager.generate_inventory_from_detections(old_format)
   volume_estimate = inventory_manager.estimate_moving_volume(inventory)
  
   # Room breakdown
   from collections import Counter
   room_breakdown = dict(Counter(room_labels)) if room_labels else None
  
   # Step 6: Summary
   progress_placeholder.progress(90, text="Generating summary...")
  
   from utils.genai_summary import SummaryGenerator
   summary_gen = SummaryGenerator()
   inv_dict = {n: i.to_dict() for n, i in inventory.items()}
   summary_result = summary_gen.generate_summary(inv_dict, volume_estimate, stats)
  
   progress_placeholder.progress(100, text="Complete!")
  
   return {
       "video_info": video_info,
       "inventory": inventory,
       "stats": stats,
       "volume_estimate": volume_estimate,
       "room_breakdown": room_breakdown,
       "summary": summary_result,
       "frames": frames,
       "annotated_frames": all_annotated,
       "frame_results": frame_results,
       "config_used": {
           "model": config.model_type.value,
           "tracker": config.tracker_type.value,
           "depth": config.depth_estimator_type.value,
           "room": config.room_classifier_type.value,
       }
   }




def render_frame_browser(results: dict):
   """Render frame browser for selecting specific frames."""
   st.markdown("## 🖼️ Frame Browser")
  
   annotated_frames = results.get("annotated_frames", [])
  
   if not annotated_frames:
       st.info("No frames available")
       return
  
   total_frames = len(annotated_frames)
  
   # Initialize frame browser state if needed
   if 'frame_idx' not in st.session_state:
       st.session_state.frame_idx = 0
   if st.session_state.frame_idx >= total_frames:
       st.session_state.frame_idx = 0
  
   col1, col2, col3 = st.columns([1, 2, 1])
  
   with col1:
       st.markdown("### Navigation")
      
       # Frame selection method - use key to maintain state
       selection_method = st.radio(
           "Selection Method",
           ["Slider", "Specific Index", "Range"],
           horizontal=True,
           key="fb_selection_method"
       )
      
       if selection_method == "Slider":
           # Use callback to avoid full page refresh feel
           frame_idx = st.slider(
               "Select Frame",
               0, total_frames - 1,
               st.session_state.frame_idx,
               key="fb_slider",
               help="Drag to browse frames"
           )
           st.session_state.frame_idx = frame_idx
           selected_indices = [frame_idx]
          
       elif selection_method == "Specific Index":
           indices_input = st.text_input(
               "Frame Indices (comma-separated)",
               st.session_state.get('indices_input', '0, 5, 10'),
               key="fb_indices",
               help="Enter specific frame numbers"
           )
           st.session_state.indices_input = indices_input
           try:
               selected_indices = [
                   int(i.strip()) for i in indices_input.split(",")
                   if i.strip().isdigit() and 0 <= int(i.strip()) < total_frames
               ]
           except:
               selected_indices = [0]
              
       else:  # Range
           range_cols = st.columns(2)
           with range_cols[0]:
               start = st.number_input(
                   "Start", 0, total_frames-1,
                   min(st.session_state.get('range_start', 0), total_frames-1),
                   key="fb_range_start"
               )
               st.session_state.range_start = start
           with range_cols[1]:
               default_end = min(st.session_state.get('range_end', 5), total_frames-1)
               end = st.number_input("End", 0, total_frames-1, default_end, key="fb_range_end")
               st.session_state.range_end = end
           step = st.number_input("Step", 1, total_frames, st.session_state.get('range_step', 1), key="fb_range_step")
           st.session_state.range_step = step
           selected_indices = list(range(int(start), int(end)+1, int(step)))
      
       st.markdown(f"**Showing:** {len(selected_indices)} frame(s)")
       st.markdown(f"**Total Available:** {total_frames}")
      
       # Quick navigation buttons
       st.markdown("### Quick Nav")
       nav_cols = st.columns(4)
       with nav_cols[0]:
           if st.button("⏮️", help="First", key="fb_first"):
               st.session_state.frame_idx = 0
               st.rerun()
       with nav_cols[1]:
           if st.button("◀️", help="Previous", key="fb_prev"):
               st.session_state.frame_idx = max(0, st.session_state.frame_idx - 1)
               st.rerun()
       with nav_cols[2]:
           if st.button("▶️", help="Next", key="fb_next"):
               st.session_state.frame_idx = min(total_frames - 1, st.session_state.frame_idx + 1)
               st.rerun()
       with nav_cols[3]:
           if st.button("⏭️", help="Last", key="fb_last"):
               st.session_state.frame_idx = total_frames - 1
               st.rerun()
  
   with col2:
       st.markdown("### Preview")
      
       if len(selected_indices) == 1:
           # Single frame view
           idx = selected_indices[0]
           frame = annotated_frames[idx]
          
           # Convert BGR to RGB for display
           if len(frame.shape) == 3 and frame.shape[2] == 3:
               frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
           else:
               frame_rgb = frame
          
           st.image(frame_rgb, caption=f"Frame {idx}", width='stretch')
          
           # Frame details
           frame_result = results.get("frame_results", [])[idx] if idx < len(results.get("frame_results", [])) else None
           if frame_result:
               st.markdown(f"**Detections:** {len(frame_result.detections)}")
               if frame_result.room_label:
                   st.markdown(f"**Room:** {frame_result.room_label}")
       else:
           # Multiple frames
           cols = st.columns(min(len(selected_indices), 4))
           for i, idx in enumerate(selected_indices[:8]):
               with cols[i % 4]:
                   frame = annotated_frames[idx]
                   if len(frame.shape) == 3 and frame.shape[2] == 3:
                       frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                   else:
                       frame_rgb = frame
                   st.image(frame_rgb, caption=f"Frame {idx}", use_container_width=True)
  
   with col3:
       st.markdown("### Actions")
      
       # Download selected frames
       if st.button("📥 Download Selected"):
           import zipfile
           import io
          
           zip_buffer = io.BytesIO()
           with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
               for idx in selected_indices:
                   frame = annotated_frames[idx]
                   _, buffer = cv2.imencode('.jpg', frame)
                   zf.writestr(f"frame_{idx:04d}.jpg", buffer.tobytes())
          
           st.download_button(
               "⬇️ Download ZIP",
               data=zip_buffer.getvalue(),
               file_name=f"frames_{st.session_state.session_id}.zip",
               mime="application/zip"
           )
      
       # Export frame info
       if st.button("📋 Export Frame Data"):
           frame_data = []
           for idx in selected_indices:
               if idx < len(results.get("frame_results", [])):
                   fr = results["frame_results"][idx]
                   frame_data.append({
                       "frame_index": idx,
                       "detection_count": len(fr.detections),
                       "room": fr.room_label,
                       "objects": [d.class_name for d in fr.detections]
                   })
          
           st.json(frame_data)




def render_results_v2(results: dict):
   """Render enhanced results."""
   inventory = results["inventory"]
   stats = results["stats"]
   volume_estimate = results["volume_estimate"]
   summary = results["summary"]
   room_breakdown = results.get("room_breakdown")
   config_used = results.get("config_used", {})
  
   # Config used
   st.markdown("### 🔧 Configuration Used")
   cfg_cols = st.columns(4)
   with cfg_cols[0]:
       st.info(f"**Model:** {config_used.get('model', 'N/A')}")
   with cfg_cols[1]:
       st.info(f"**Tracker:** {config_used.get('tracker', 'N/A')}")
   with cfg_cols[2]:
       st.info(f"**Depth:** {config_used.get('depth', 'N/A')}")
   with cfg_cols[3]:
       st.info(f"**Room:** {config_used.get('room', 'N/A')}")
  
   st.markdown("---")
  
   # Stats
   st.markdown("## 📊 Processing Statistics")
   stat_cols = st.columns(5)
  
   metrics = [
       ("Frames", stats.get("total_frames_processed", 0)),
       ("Raw Detections", stats.get("total_raw_detections", 0)),
       ("Unique Objects", stats.get("unique_objects_found", 0)),
       ("Item Types", stats.get("unique_item_types", 0)),
       ("Dedup Ratio", f"{stats.get('deduplication_ratio', 0):.0%}"),
   ]
  
   for col, (label, value) in zip(stat_cols, metrics):
       with col:
           st.metric(label, value)
  
   st.markdown("---")
  
   # Room breakdown
   if room_breakdown:
       st.markdown("## 🏠 Room Breakdown")
       room_cols = st.columns(len(room_breakdown))
       for col, (room, count) in zip(room_cols, room_breakdown.items()):
           with col:
               st.metric(room.replace("_", " ").title(), f"{count} frames")
  
   st.markdown("---")
  
   # Main content
   col_left, col_right = st.columns([3, 2])
  
   with col_left:
       st.markdown("## 📦 Inventory")
      
       if inventory:
           data = []
           for name, item in inventory.items():
               data.append({
                   "Item": item.name,
                   "Count": item.count,
                   "Category": item.category,
                   "Confidence": f"{item.avg_confidence:.0%}"
               })
          
           df = pd.DataFrame(data)
           st.dataframe(df, use_container_width=True, hide_index=True)
          
           # Category chart
           cat_counts = df.groupby("Category")["Count"].sum()
           st.bar_chart(cat_counts)
  
   with col_right:
       st.markdown("## 🚚 Moving Estimate")
      
       st.markdown(f"""
       <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                   padding: 1.5rem; border-radius: 10px; color: white;">
           <h3 style="margin: 0;">Estimated Volume</h3>
           <h1 style="margin: 0.5rem 0;">{volume_estimate.get('total_volume_cuft', 0)} cu ft</h1>
       </div>
       """, unsafe_allow_html=True)
      
       st.markdown("### Recommendations")
       st.markdown(f"🚛 **Truck:** {volume_estimate.get('recommended_truck', 'N/A')}")
       st.markdown(f"👥 **Crew:** {volume_estimate.get('recommended_crew', 'N/A')}")
       st.markdown(f"📦 **Boxes:** {volume_estimate.get('estimated_packing_boxes', 0)}")
  
   st.markdown("---")
  
   # Frame browser
   render_frame_browser(results)
  
   st.markdown("---")
  
   # AI Summary
   st.markdown("## 🤖 AI Summary")
   st.caption(f"Generated using: {summary.method}")
   st.markdown(summary.summary)
  
   st.markdown("---")
  
   # Exports
   st.markdown("## 💾 Export")
  
   exp_cols = st.columns(4)
  
   with exp_cols[0]:
       export_data = {
           "generated_at": datetime.now().isoformat(),
           "config": config_used,
           "stats": stats,
           "inventory": {n: i.to_dict() for n, i in inventory.items()},
           "volume_estimate": volume_estimate,
           "room_breakdown": room_breakdown,
           "summary": summary.summary
       }
       st.download_button(
           "📥 JSON",
           json.dumps(export_data, indent=2),
           f"inventory_{st.session_state.session_id}.json",
           "application/json"
       )
  
   with exp_cols[1]:
       if inventory:
           st.download_button(
               "📥 CSV",
               df.to_csv(index=False),
               f"inventory_{st.session_state.session_id}.csv",
               "text/csv"
           )
  
   with exp_cols[2]:
       st.download_button(
           "📥 Summary",
           summary.summary,
           f"summary_{st.session_state.session_id}.txt",
           "text/plain"
       )
  
   with exp_cols[3]:
       # Full report
       report = f"""
AI Moving Inventory Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*50}


CONFIGURATION
{'-'*50}
Model: {config_used.get('model')}
Tracker: {config_used.get('tracker')}
Depth: {config_used.get('depth')}
Room: {config_used.get('room')}


STATISTICS
{'-'*50}
Frames Processed: {stats.get('total_frames_processed')}
Raw Detections: {stats.get('total_raw_detections')}
Unique Objects: {stats.get('unique_objects_found')}


INVENTORY
{'-'*50}
{chr(10).join(f'- {item.name}: {item.count} ({item.category})' for item in inventory.values())}


VOLUME ESTIMATE
{'-'*50}
Total Volume: {volume_estimate.get('total_volume_cuft')} cu ft
Recommended Truck: {volume_estimate.get('recommended_truck')}
Recommended Crew: {volume_estimate.get('recommended_crew')}


AI SUMMARY
{'-'*50}
{summary.summary}
"""
       st.download_button(
           "📥 Full Report",
           report,
           f"full_report_{st.session_state.session_id}.txt",
           "text/plain"
       )




def main():
   """Main application."""
   init_session_state()
   render_header()
   config = render_model_config()
  
   st.markdown("## 📤 Upload Video")
  
   uploaded_file = st.file_uploader(
       "Choose a home walkthrough video",
       type=["mp4", "mov", "avi", "mkv"],
       help="Supported: MP4, MOV, AVI, MKV",
       key="video_uploader"
   )
  
   if uploaded_file:
       col1, col2 = st.columns([2, 1])
      
       with col1:
           st.video(uploaded_file)
      
       with col2:
           st.markdown("### Video Info")
           st.markdown(f"**File:** {uploaded_file.name}")
           st.markdown(f"**Size:** {uploaded_file.size / (1024*1024):.2f} MB")
      
       st.markdown("---")
      
       # Show process button OR results based on state
       process_button = st.button(
           "🚀 Generate Inventory",
           type="primary",
           use_container_width=True,
           disabled=st.session_state.processed,
           key="process_btn"
       )
      
       # Reset button if already processed
       if st.session_state.processed:
           if st.button("🔄 Process New Video", key="reset_btn"):
               st.session_state.processed = False
               st.session_state.results = None
               st.session_state.frame_idx = 0
               st.rerun()
      
       if process_button and not st.session_state.processed:
           st.session_state.session_id = str(uuid.uuid4())[:8]
           st.session_state.frame_idx = 0
          
           with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
               tmp.write(uploaded_file.getvalue())
               tmp_path = tmp.name
          
           try:
               progress = st.progress(0, text="Starting...")
              
               with st.spinner("Processing..."):
                   results = process_video_v2(tmp_path, config, progress)
              
               if results:
                   st.session_state.processed = True
                   st.session_state.results = results
                   progress.empty()
                   st.rerun()  # Rerun to show results cleanly
                  
           except Exception as e:
               st.error(f"Error: {e}")
               import traceback
               st.code(traceback.format_exc())
          
           finally:
               if os.path.exists(tmp_path):
                   os.unlink(tmp_path)
      
       # Display results if already processed
       if st.session_state.processed and st.session_state.results:
           st.success("✅ Inventory generated!")
           render_results_v2(st.session_state.results)
  
   else:
       # Instructions
       st.markdown("""
       <div style="background: #f8f9fa; padding: 2rem; border-radius: 10px; text-align: center;">
           <h3>📹 How to use</h3>
           <p>1. Configure models in sidebar<br>
           2. Upload walkthrough video<br>
           3. Click Generate Inventory<br>
           4. Browse frames and export results</p>
       </div>
       """, unsafe_allow_html=True)
      
       # Features
       st.markdown("### ✨ New Features in V2")
      
       feat_cols = st.columns(3)
      
       with feat_cols[0]:
           st.markdown("""
           **🎯 Advanced Tracking**
           - DeepSORT
           - ByteTrack
           - Kalman filtering
           """)
      
       with feat_cols[1]:
           st.markdown("""
           **📏 3D Volume**
           - MiDaS depth
           - Rule-based estimation
           - Per-object volumes
           """)
      
       with feat_cols[2]:
           st.markdown("""
           **🏠 Room Detection**
           - Auto classification
           - CLIP support
           - Room-wise inventory
           """)




if __name__ == "__main__":
   main()



