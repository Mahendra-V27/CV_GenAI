# AI-Powered Moving Inventory Generation System


An end-to-end Computer Vision + GenAI solution that automatically generates moving inventory from home walkthrough videos.


## Features


- **Video Upload**: Upload MP4/MOV walkthrough videos
- **Frame Extraction**: Intelligent frame sampling using OpenCV
- **Object Detection**: YOLOv8-based household item detection
- **Smart Deduplication**: Prevents duplicate counting across frames
- **GenAI Summary**: Natural language inventory reports using local LLM (Ollama)
- **Room-wise Inventory**: Organized item categorization


## Architecture


```
User Uploads Video
       ↓
Streamlit UI
       ↓
Frame Extraction (OpenCV)
       ↓
YOLO Object Detection
       ↓
Object Aggregation & Deduplication
       ↓
Inventory Generation
       ↓
GenAI Summary (Ollama/Llama)
       ↓
Display Final Inventory
```


## Tech Stack


| Component | Technology |
|-----------|------------|
| Frontend | Streamlit |
| Video Processing | OpenCV |
| Object Detection | YOLOv8 |
| Deduplication | Spatial + Temporal Logic |
| GenAI Summary | Ollama (Llama3/Mistral) |
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


# Start Ollama service
ollama serve


# Pull a model (in another terminal)
ollama pull llama3.2
```


### 3. Run the Application


```bash
streamlit run app.py
```


## Usage


1. Open the Streamlit app in your browser (http://localhost:8501)
2. Upload a home walkthrough video (MP4/MOV)
3. Adjust settings (frame extraction rate, confidence threshold)
4. Click "Generate Inventory"
5. View detected objects, inventory table, and AI-generated summary


## Detected Object Categories


The system detects common household items including:
- **Furniture**: Sofa, Chair, Bed, Dining Table, Desk
- **Electronics**: TV, Laptop, Monitor, Microwave, Refrigerator
- **Storage**: Suitcase, Backpack, Potted Plant
- **Kitchen**: Oven, Toaster, Sink
- And more from COCO dataset classes


## Project Structure


```
inventory_list/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # Documentation
├── uploads/              # Uploaded videos (auto-created)
├── extracted_frames/     # Extracted video frames
├── outputs/              # Generated reports
└── utils/
   ├── __init__.py
   ├── video_processing.py   # Frame extraction logic
   ├── detection.py          # YOLOv8 detection
   ├── inventory.py          # Aggregation & deduplication
   └── genai_summary.py      # LLM summary generation
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
