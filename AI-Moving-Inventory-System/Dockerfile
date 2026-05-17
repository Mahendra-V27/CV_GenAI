# AI Moving Inventory System - Docker Image
FROM python:3.11-slim


# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive


# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
   libgl1-mesa-glx \
   libglib2.0-0 \
   libsm6 \
   libxext6 \
   libxrender-dev \
   libgomp1 \
   ffmpeg \
   curl \
   && rm -rf /var/lib/apt/lists/*


# Set working directory
WORKDIR /app


# Copy requirements first for caching
COPY requirements.txt .


# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
   pip install --no-cache-dir -r requirements.txt && \
   pip install --no-cache-dir fastapi uvicorn python-multipart


# Copy application code
COPY . .


# Create necessary directories
RUN mkdir -p uploads extracted_frames outputs models


# Download YOLOv8 model
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"


# Expose ports
EXPOSE 8000 8501


# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
   CMD curl -f http://localhost:8000/ || exit 1


# Default command (FastAPI)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
