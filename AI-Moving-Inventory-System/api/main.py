"""
FastAPI Backend for AI Moving Inventory System
Cloud-ready REST API with async processing
"""


import os
import sys
import uuid
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime
import tempfile
import shutil


from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn


# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


from utils.interfaces import (
   ProcessingConfig, ModelType, TrackerType,
   DepthEstimatorType, RoomClassifierType, SummaryGeneratorType
)
from utils.model_registry import ComponentFactory, DEFAULT_CONFIG


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Initialize FastAPI app
app = FastAPI(
   title="AI Moving Inventory API",
   description="Computer Vision + GenAI powered inventory generation",
   version="2.0.0",
   docs_url="/docs",
   redoc_url="/redoc"
)


# CORS middleware
app.add_middleware(
   CORSMiddleware,
   allow_origins=["*"],
   allow_credentials=True,
   allow_methods=["*"],
   allow_headers=["*"],
)


# Job storage (in production, use Redis/database)
jobs: Dict[str, Dict] = {}


# Directory setup
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
FRAMES_DIR = Path("extracted_frames")


for d in [UPLOAD_DIR, OUTPUT_DIR, FRAMES_DIR]:
   d.mkdir(exist_ok=True)




# Pydantic Models
class ProcessingConfigRequest(BaseModel):
   """Request model for processing configuration."""
   model_type: str = Field(default="yolov8n", description="Detection model type")
   tracker_type: str = Field(default="simple", description="Tracking algorithm")
   depth_estimator: str = Field(default="estimated", description="Depth estimation method")
   room_classifier: str = Field(default="rule_based", description="Room classification method")
   confidence_threshold: float = Field(default=0.4, ge=0.1, le=0.95)
   extraction_rate: str = Field(default="1_per_second")
   max_frames: int = Field(default=100, ge=10, le=500)
   preview_frame_count: int = Field(default=5, ge=1, le=20)




class JobStatus(BaseModel):
   """Job status response."""
   job_id: str
   status: str  # pending, processing, completed, failed
   progress: float
   message: str
   created_at: str
   updated_at: str
   result: Optional[Dict] = None




class InventoryItem(BaseModel):
   """Inventory item model."""
   name: str
   count: int
   category: str
   avg_confidence: float




class InventoryResponse(BaseModel):
   """Full inventory response."""
   job_id: str
   video_info: Dict
   stats: Dict
   inventory: List[InventoryItem]
   volume_estimate: Dict
   room_breakdown: Optional[Dict] = None
   summary: str
   annotated_frames: List[str]




class FrameRequest(BaseModel):
   """Request specific frames."""
   job_id: str
   frame_indices: List[int] = Field(default=[], description="Specific frame indices to retrieve")
   count: int = Field(default=5, description="Number of frames if indices not specified")




class ModelListResponse(BaseModel):
   """Available models response."""
   detectors: List[Dict]
   trackers: List[Dict]
   depth_estimators: List[Dict]
   room_classifiers: List[Dict]




# Helper functions
def get_config_from_request(req: ProcessingConfigRequest) -> ProcessingConfig:
   """Convert request to ProcessingConfig."""
   model_map = {
       "yolov8n": ModelType.YOLOV8_NANO,
       "yolov8s": ModelType.YOLOV8_SMALL,
       "yolov8m": ModelType.YOLOV8_MEDIUM,
       "yolov8l": ModelType.YOLOV8_LARGE,
       "yolov8x": ModelType.YOLOV8_XLARGE,
       "onnx": ModelType.ONNX,
   }
  
   tracker_map = {
       "none": TrackerType.NONE,
       "simple": TrackerType.SIMPLE,
       "deepsort": TrackerType.DEEPSORT,
       "bytetrack": TrackerType.BYTETRACK,
   }
  
   depth_map = {
       "none": DepthEstimatorType.NONE,
       "estimated": DepthEstimatorType.ESTIMATED,
       "midas": DepthEstimatorType.MIDAS,
       "depth_anything": DepthEstimatorType.DEPTH_ANYTHING,
   }
  
   room_map = {
       "none": RoomClassifierType.NONE,
       "rule_based": RoomClassifierType.RULE_BASED,
       "clip": RoomClassifierType.CLIP,
       "cnn": RoomClassifierType.CNN,
   }
  
   return ProcessingConfig(
       model_type=model_map.get(req.model_type, ModelType.YOLOV8_NANO),
       tracker_type=tracker_map.get(req.tracker_type, TrackerType.SIMPLE),
       depth_estimator_type=depth_map.get(req.depth_estimator, DepthEstimatorType.ESTIMATED),
       room_classifier_type=room_map.get(req.room_classifier, RoomClassifierType.RULE_BASED),
       confidence_threshold=req.confidence_threshold,
       extraction_rate=req.extraction_rate,
       max_frames=req.max_frames,
       preview_frame_count=req.preview_frame_count,
   )




async def process_video_task(job_id: str, video_path: str, config: ProcessingConfig):
   """Background task for video processing."""
   try:
       jobs[job_id]["status"] = "processing"
       jobs[job_id]["updated_at"] = datetime.now().isoformat()
      
       # Import processing modules
       from utils.video_processing import VideoProcessor
       from utils.detection_v2 import YOLOv8Detector
       from utils.inventory import InventoryManager
       from utils.genai_summary import SummaryGenerator
      
       # Create components
       factory = ComponentFactory(config)
       video_processor = VideoProcessor(output_dir=str(FRAMES_DIR))
      
       # Step 1: Extract frames
       jobs[job_id]["progress"] = 0.1
       jobs[job_id]["message"] = "Extracting frames..."
      
       video_info = video_processor.get_video_info(video_path)
       frames, frame_paths = video_processor.extract_frames(
           video_path,
           extraction_rate=config.extraction_rate,
           max_frames=config.max_frames,
           session_id=job_id
       )
      
       # Step 2: Run detection
       jobs[job_id]["progress"] = 0.3
       jobs[job_id]["message"] = f"Detecting objects in {len(frames)} frames..."
      
       detector = factory.get_detector()
       tracker = factory.get_tracker()
       depth_estimator = factory.get_depth_estimator()
       room_classifier = factory.get_room_classifier()
      
       all_detections = []
       annotated_paths = []
       room_labels = []
      
       for idx, frame in enumerate(frames):
           # Detect
           detections = detector.detect(frame, frame_index=idx)
          
           # Track
           if tracker:
               detections = tracker.update(detections, frame)
          
           # Depth estimation
           if depth_estimator:
               depth_map = depth_estimator.estimate_depth(frame)
               for det in detections:
                   det.depth = depth_estimator.get_object_depth(depth_map, det.bbox)
                   det.volume_3d = depth_estimator.estimate_3d_volume(det, depth_map)
          
           # Room classification
           if room_classifier:
               room = room_classifier.classify_room(frame, detections)
               room_labels.append(room)
               for det in detections:
                   det.room = room
          
           all_detections.append(detections)
          
           # Save annotated frame
           annotated = detector.get_annotated_frame(
               frame, detections,
               show_depth=depth_estimator is not None,
               show_room=room_classifier is not None
           )
          
           ann_path = OUTPUT_DIR / f"{job_id}_frame_{idx:04d}.jpg"
           import cv2
           cv2.imwrite(str(ann_path), annotated)
           annotated_paths.append(str(ann_path))
          
           jobs[job_id]["progress"] = 0.3 + (0.4 * (idx + 1) / len(frames))
      
       # Step 3: Generate inventory
       jobs[job_id]["progress"] = 0.75
       jobs[job_id]["message"] = "Generating inventory..."
      
       inventory_manager = InventoryManager()
      
       # Convert detections to old format for compatibility
       from utils.detection import DetectedObject
       old_format_detections = []
       for frame_dets in all_detections:
           frame_old = []
           for d in frame_dets:
               frame_old.append(DetectedObject(
                   class_name=d.class_name,
                   confidence=d.confidence,
                   bbox=d.bbox.to_tuple(),
                   frame_index=d.frame_index
               ))
           old_format_detections.append(frame_old)
      
       inventory, stats = inventory_manager.generate_inventory_from_detections(old_format_detections)
       volume_estimate = inventory_manager.estimate_moving_volume(inventory)
      
       # Room breakdown
       from collections import Counter
       room_breakdown = dict(Counter(room_labels)) if room_labels else None
      
       # Step 4: Generate summary
       jobs[job_id]["progress"] = 0.9
       jobs[job_id]["message"] = "Generating AI summary..."
      
       summary_generator = SummaryGenerator()
       inventory_dict = {name: item.to_dict() for name, item in inventory.items()}
       summary_result = summary_generator.generate_summary(
           inventory_dict, volume_estimate, stats
       )
      
       # Store result
       jobs[job_id]["status"] = "completed"
       jobs[job_id]["progress"] = 1.0
       jobs[job_id]["message"] = "Processing complete"
       jobs[job_id]["updated_at"] = datetime.now().isoformat()
       jobs[job_id]["result"] = {
           "video_info": video_info,
           "stats": stats,
           "inventory": [
               {
                   "name": item.name,
                   "count": item.count,
                   "category": item.category,
                   "avg_confidence": item.avg_confidence
               }
               for item in inventory.values()
           ],
           "volume_estimate": volume_estimate,
           "room_breakdown": room_breakdown,
           "summary": summary_result.summary,
           "annotated_frames": annotated_paths[:config.preview_frame_count],
           "all_frame_paths": annotated_paths
       }
      
   except Exception as e:
       logger.error(f"Processing failed: {e}")
       jobs[job_id]["status"] = "failed"
       jobs[job_id]["message"] = str(e)
       jobs[job_id]["updated_at"] = datetime.now().isoformat()




# API Endpoints


@app.get("/")
async def root():
   """Health check."""
   return {"status": "healthy", "service": "AI Moving Inventory API", "version": "2.0.0"}




@app.get("/models", response_model=ModelListResponse)
async def list_models():
   """List available models and their configurations."""
   from utils.model_registry import ModelRegistry
  
   return {
       "detectors": [
           {"type": k.value, "name": v.name, "description": v.description}
           for k, v in ModelRegistry.get_available_detectors().items()
       ],
       "trackers": [
           {"type": k.value, "name": v.name, "description": v.description}
           for k, v in ModelRegistry.get_available_trackers().items()
       ],
       "depth_estimators": [
           {"type": k.value, "name": v.name, "description": v.description}
           for k, v in ModelRegistry.get_available_depth_estimators().items()
       ],
       "room_classifiers": [
           {"type": k.value, "name": v.name, "description": v.description}
           for k, v in ModelRegistry.get_available_room_classifiers().items()
       ]
   }




@app.post("/process", response_model=JobStatus)
async def process_video(
   background_tasks: BackgroundTasks,
   file: UploadFile = File(...),
   model_type: str = Query(default="yolov8n"),
   tracker_type: str = Query(default="simple"),
   depth_estimator: str = Query(default="estimated"),
   room_classifier: str = Query(default="rule_based"),
   confidence_threshold: float = Query(default=0.4),
   extraction_rate: str = Query(default="1_per_second"),
   max_frames: int = Query(default=100),
   preview_frame_count: int = Query(default=5)
):
   """Upload video and start processing."""
   # Validate file
   if not file.filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
       raise HTTPException(400, "Unsupported file format. Use MP4, MOV, AVI, or MKV.")
  
   # Create job
   job_id = str(uuid.uuid4())[:8]
  
   # Save uploaded file
   video_path = UPLOAD_DIR / f"{job_id}_{file.filename}"
   with open(video_path, "wb") as f:
       content = await file.read()
       f.write(content)
  
   # Create config
   config_req = ProcessingConfigRequest(
       model_type=model_type,
       tracker_type=tracker_type,
       depth_estimator=depth_estimator,
       room_classifier=room_classifier,
       confidence_threshold=confidence_threshold,
       extraction_rate=extraction_rate,
       max_frames=max_frames,
       preview_frame_count=preview_frame_count
   )
   config = get_config_from_request(config_req)
  
   # Initialize job
   jobs[job_id] = {
       "job_id": job_id,
       "status": "pending",
       "progress": 0.0,
       "message": "Job queued",
       "created_at": datetime.now().isoformat(),
       "updated_at": datetime.now().isoformat(),
       "video_path": str(video_path),
       "result": None
   }
  
   # Start background processing
   background_tasks.add_task(process_video_task, job_id, str(video_path), config)
  
   return JobStatus(**jobs[job_id])




@app.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
   """Get processing job status."""
   if job_id not in jobs:
       raise HTTPException(404, "Job not found")
   return JobStatus(**jobs[job_id])




@app.get("/jobs/{job_id}/result")
async def get_job_result(job_id: str):
   """Get complete job result."""
   if job_id not in jobs:
       raise HTTPException(404, "Job not found")
  
   job = jobs[job_id]
   if job["status"] != "completed":
       raise HTTPException(400, f"Job status is {job['status']}, not completed")
  
   return job["result"]




@app.get("/jobs/{job_id}/frames")
async def get_frames(
   job_id: str,
   indices: str = Query(default="", description="Comma-separated frame indices"),
   count: int = Query(default=5, description="Number of frames to return")
):
   """Get specific annotated frames."""
   if job_id not in jobs:
       raise HTTPException(404, "Job not found")
  
   job = jobs[job_id]
   if job["status"] != "completed" or not job.get("result"):
       raise HTTPException(400, "Job not completed")
  
   all_paths = job["result"].get("all_frame_paths", [])
  
   if indices:
       # Get specific indices
       idx_list = [int(i.strip()) for i in indices.split(",") if i.strip().isdigit()]
       selected = [all_paths[i] for i in idx_list if 0 <= i < len(all_paths)]
   else:
       # Get evenly spaced frames
       step = max(1, len(all_paths) // count)
       selected = all_paths[::step][:count]
  
   return {"frames": selected, "total_available": len(all_paths)}




@app.get("/frames/{job_id}/{frame_index}")
async def get_single_frame(job_id: str, frame_index: int):
   """Get a single annotated frame image."""
   if job_id not in jobs:
       raise HTTPException(404, "Job not found")
  
   job = jobs[job_id]
   if job["status"] != "completed":
       raise HTTPException(400, "Job not completed")
  
   all_paths = job["result"].get("all_frame_paths", [])
  
   if frame_index < 0 or frame_index >= len(all_paths):
       raise HTTPException(404, "Frame index out of range")
  
   frame_path = all_paths[frame_index]
  
   if not Path(frame_path).exists():
       raise HTTPException(404, "Frame file not found")
  
   return FileResponse(frame_path, media_type="image/jpeg")




@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
   """Delete a job and its files."""
   if job_id not in jobs:
       raise HTTPException(404, "Job not found")
  
   job = jobs[job_id]
  
   # Clean up files
   if "video_path" in job and Path(job["video_path"]).exists():
       os.remove(job["video_path"])
  
   if job.get("result", {}).get("all_frame_paths"):
       for path in job["result"]["all_frame_paths"]:
           if Path(path).exists():
               os.remove(path)
  
   del jobs[job_id]
  
   return {"message": "Job deleted", "job_id": job_id}




@app.post("/export/onnx")
async def export_model_to_onnx(model_type: str = Query(default="yolov8n")):
   """Export detection model to ONNX format."""
   try:
       from utils.detection_v2 import YOLOv8Detector
      
       model_map = {
           "yolov8n": ModelType.YOLOV8_NANO,
           "yolov8s": ModelType.YOLOV8_SMALL,
           "yolov8m": ModelType.YOLOV8_MEDIUM,
       }
      
       mt = model_map.get(model_type, ModelType.YOLOV8_NANO)
       detector = YOLOv8Detector(model_type=mt)
      
       output_path = f"models/{model_type}.onnx"
       Path("models").mkdir(exist_ok=True)
      
       success = detector.export_to_onnx(output_path)
      
       if success:
           return {"message": "Export successful", "path": output_path}
       else:
           raise HTTPException(500, "Export failed")
          
   except Exception as e:
       raise HTTPException(500, str(e))




# Run server
if __name__ == "__main__":
   uvicorn.run(
       "main:app",
       host="0.0.0.0",
       port=8000,
       reload=True,
       workers=1
   )