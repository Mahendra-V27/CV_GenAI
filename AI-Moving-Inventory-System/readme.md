
# AI-Powered Moving Inventory Generation System V2


An end-to-end Computer Vision + GenAI solution that automatically generates moving inventory from home walkthrough videos.

## Demo Video

https://github.com/user-attachments/assets/b1467011-0680-405c-8da6-696fc2bfe2bd



## 🚀 New in V2


- **Interface-based Architecture**: Loosely coupled design with dependency injection
- **Advanced Object Tracking**: DeepSORT, ByteTrack with Kalman filtering
- **3D Volume Estimation**: MiDaS/Depth Anything depth estimation
- **Multi-Room Segmentation**: Auto room classification (CLIP/rule-based)
- **REST API**: FastAPI backend for cloud deployment
- **ONNX Export**: Optimized models for edge inference
- **Model Selection UI**: Choose detection models, trackers, depth estimators
- **Frame Browser**: Navigate and export specific annotated frames


## Features


- **Video Upload**: Upload MP4/MOV/AVI/MKV walkthrough videos
- **Frame Extraction**: Intelligent frame sampling using OpenCV
- **Object Detection**: YOLOv8 (nano to xlarge) + ONNX runtime support
- **Smart Tracking**: DeepSORT, ByteTrack for accurate counting
- **Depth Estimation**: 3D volume estimation per object
- **Room Classification**: Auto-detect living room, bedroom, kitchen, etc.
- **GenAI Summary**: Natural language inventory reports (Ollama/Transformers)
- **Export Options**: JSON, CSV, annotated frames, full reports


## Architecture


```
User Uploads Video
       ↓
┌─────────────────────────────────────────┐
│         Configuration Selection          │
│  (Model, Tracker, Depth, Room, Rate)    │
└─────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────┐
│     Streamlit UI / FastAPI Backend      │
└─────────────────────────────────────────┘
       ↓
Frame Extraction (OpenCV)
       ↓
┌─────────────────────────────────────────┐
│      YOLOv8 Object Detection            │
│   (n/s/m/l/x variants + ONNX)           │
└─────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────┐
│      Advanced Tracking                   │
│   (Simple/DeepSORT/ByteTrack)           │
└─────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────┐
│      Depth Estimation + Room Detection  │
│   (MiDaS/Rule-based + CLIP/Rule-based)  │
└─────────────────────────────────────────┘
       ↓
Inventory Aggregation & Deduplication
       ↓
GenAI Summary (Ollama/Transformers/Template)
       ↓
Results + Export Options
```


## Tech Stack


| Component | Technology |
|-----------|------------|
| Frontend | Streamlit |
| REST API | FastAPI |
| Video Processing | OpenCV |
| Object Detection | YOLOv8, ONNX Runtime |
| Object Tracking | DeepSORT, ByteTrack |
| Depth Estimation | MiDaS, Depth Anything, Rule-based |
| Room Classification | CLIP, Rule-based |
| GenAI Summary | Ollama, HuggingFace Transformers |
| Deployment | Docker, Kubernetes |
| Backend | Python |


## Installation


### 1. Clone and Setup


```bash
cd inventory_list
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```


### 2. Install Ollama (for GenAI summary)


```bash
# macOS
brew install ollama


# Linux
curl -fsSL https://ollama.ai/install.sh | sh


# Start Ollama service
ollama serve


# Pull a model (in another terminal)
ollama pull llama3.2
```


---


## Execution & Deployment


### On-Premises Deployment


#### Option 1: Streamlit UI (Development/Local)


```bash
# V1 - Basic UI
streamlit run app.py --server.port 8501


# V2 - Enhanced UI with model selection
streamlit run app_v2.py --server.port 8502
```


Access at: `http://localhost:8501` or `http://localhost:8502`


#### Option 2: FastAPI Backend (Production)


```bash
# Development
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000


# Production (with workers)
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```


API docs at: `http://localhost:8000/docs`


#### Option 3: Docker (Recommended for Production)


```bash
# Build image
docker build -t inventory-system:latest .


# Run container
docker run -d \
 --name inventory-api \
 -p 8000:8000 \
 -v $(pwd)/uploads:/app/uploads \
 -v $(pwd)/outputs:/app/outputs \
 inventory-system:latest


# With GPU support (NVIDIA)
docker run -d --gpus all \
 --name inventory-api \
 -p 8000:8000 \
 inventory-system:latest
```


#### Option 4: Docker Compose (Full Stack)


Create `docker-compose.yml`:
```yaml
version: '3.8'
services:
 api:
   build: .
   ports:
     - "8000:8000"
   volumes:
     - ./uploads:/app/uploads
     - ./outputs:/app/outputs
   environment:
     - WORKERS=4
   restart: unless-stopped
  
 streamlit:
   build: .
   command: streamlit run app_v2.py --server.port 8501 --server.headless true
   ports:
     - "8501:8501"
   volumes:
     - ./uploads:/app/uploads
   depends_on:
     - api
   restart: unless-stopped
```


Run with:
```bash
docker-compose up -d
```


---


### Cloud Deployment


#### AWS (ECS/EKS)


```bash
# Push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com
docker tag inventory-system:latest <account>.dkr.ecr.us-east-1.amazonaws.com/inventory-system:latest
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/inventory-system:latest


# Deploy to EKS
kubectl apply -f k8s/deployment.yaml
```


#### Google Cloud (GKE/Cloud Run)


```bash
# Push to GCR
gcloud auth configure-docker
docker tag inventory-system:latest gcr.io/<project-id>/inventory-system:latest
docker push gcr.io/<project-id>/inventory-system:latest


# Deploy to Cloud Run (serverless)
gcloud run deploy inventory-api \
 --image gcr.io/<project-id>/inventory-system:latest \
 --platform managed \
 --region us-central1 \
 --memory 4Gi \
 --cpu 2 \
 --allow-unauthenticated


# Deploy to GKE
kubectl apply -f k8s/deployment.yaml
```


#### Azure (AKS/Container Apps)


```bash
# Push to ACR
az acr login --name <registry-name>
docker tag inventory-system:latest <registry-name>.azurecr.io/inventory-system:latest
docker push <registry-name>.azurecr.io/inventory-system:latest


# Deploy to AKS
kubectl apply -f k8s/deployment.yaml


# Deploy to Container Apps
az containerapp create \
 --name inventory-api \
 --resource-group <rg-name> \
 --image <registry-name>.azurecr.io/inventory-system:latest \
 --target-port 8000 \
 --ingress external \
 --cpu 2 --memory 4Gi
```


#### Kubernetes (Any Cloud)


```bash
# Apply manifests
kubectl apply -f k8s/deployment.yaml


# Check status
kubectl get pods -l app=inventory-api
kubectl get svc inventory-api-service


# Scale up/down
kubectl scale deployment inventory-api --replicas=5


# View logs
kubectl logs -f deployment/inventory-api
```


---


### Environment Variables


| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_TYPE` | Detection model (yolov8n/s/m/l/x) | yolov8n |
| `CONFIDENCE_THRESHOLD` | Detection confidence | 0.4 |
| `MAX_FRAMES` | Max frames to process | 100 |
| `WORKERS` | Uvicorn workers | 1 |
| `UPLOAD_DIR` | Upload directory | uploads |
| `OUTPUT_DIR` | Output directory | outputs |


---


## Usage


### Streamlit UI


1. Open the app in your browser (`http://localhost:8501` or `8502`)
2. Configure models in the sidebar (V2 only)
3. Upload a home walkthrough video (MP4/MOV/AVI/MKV)
4. Click "Generate Inventory"
5. Browse frames, view results, and export


### REST API


```bash
# Upload and process video
curl -X POST "http://localhost:8000/api/v1/process" \
 -F "file=@walkthrough.mp4" \
 -F "model_type=yolov8n" \
 -F "tracker_type=simple"


# Check job status
curl "http://localhost:8000/api/v1/jobs/{job_id}"


# Get results
curl "http://localhost:8000/api/v1/results/{job_id}"


# Export to ONNX
curl -X POST "http://localhost:8000/api/v1/export/onnx?model_type=yolov8n"
```


---


## Detected Object Categories


The system detects common household items including:
- **Furniture**: Sofa, Chair, Bed, Dining Table, Desk
- **Electronics**: TV, Laptop, Monitor, Microwave, Refrigerator
- **Storage**: Suitcase, Backpack, Potted Plant
- **Kitchen**: Oven, Toaster, Sink, Refrigerator
- And more from COCO dataset classes (80 classes)


---


## Project Structure


```
inventory_list/
├── app.py                    # V1 Streamlit application
├── app_v2.py                 # V2 Enhanced Streamlit with model selection
├── requirements.txt          # Python dependencies
├── Dockerfile                # Docker container definition
├── README.md                 # Documentation
│
├── api/                      # FastAPI backend
│   ├── __init__.py
│   └── main.py               # REST API endpoints
│
├── k8s/                      # Kubernetes manifests
│   └── deployment.yaml       # Deployment, Service, HPA
│
├── tests/                    # Test suite
│   └── test_all_components.py
│
├── utils/                    # Core modules
│   ├── __init__.py
│   ├── interfaces.py         # Abstract interfaces (IDetector, ITracker, etc.)
│   ├── model_registry.py     # Model registry & ComponentFactory
│   ├── video_processing.py   # Frame extraction
│   ├── detection.py          # V1 YOLOv8 detection
│   ├── detection_v2.py       # V2 Detection with ONNX export
│   ├── trackers.py           # SimpleTracker, DeepSORT, ByteTrack
│   ├── depth_estimation.py   # Rule-based, MiDaS depth estimators
│   ├── room_classifier.py    # Rule-based, CLIP room classifiers
│   ├── inventory.py          # Aggregation & deduplication
│   └── genai_summary.py      # LLM summary generation
│
├── uploads/                  # Uploaded videos (auto-created)
├── extracted_frames/         # Extracted video frames
├── outputs/                  # Generated reports
└── models/                   # Downloaded/exported models
```


## Sample Output


### Inventory Table


| Item | Count | Category |
|------|-------|----------|
| Sofa | 1 | Furniture |
| Chair | 4 | Furniture |
| TV | 1 | Electronics |
| Bed | 2 | Furniture |
| Dining Table | 1 | Furniture |


### AI Summary


> "Based on the video walkthrough analysis, your home inventory includes:
> - **Living Room**: 1 sofa, 4 chairs, 1 television, 1 coffee table
> - **Bedroom**: 2 beds, 1 wardrobe
> - **Kitchen**: 1 refrigerator, 1 microwave
>
> Estimated moving volume: Medium (requires 1 standard moving truck)
> Recommended crew: 3-4 persons"


## Future Enhancements


- [ ] DeepSORT/ByteTrack for advanced tracking
- [ ] 3D volume estimation with depth sensors
- [ ] Multi-room segmentation
- [ ] Cloud deployment (FastAPI + Kubernetes)
- [ ] Real-time mobile inference (ONNX/TensorRT)


## License


MIT License
