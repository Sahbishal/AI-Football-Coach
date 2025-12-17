import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, Request, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse, StreamingResponse
import shutil
import cv2
import base64
import asyncio
import numpy as np
from pathlib import Path
from processor import VideoProcessor

app = FastAPI()

# Mount processed videos so they can be played
app.mount("/processed", StaticFiles(directory="processed"), name="processed")

templates = Jinja2Templates(directory="app/templates")

UPLOAD_DIR = Path("uploads")
PROCESSED_DIR = Path("processed")

# Ensure directories exist
UPLOAD_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)

# Initialize processor
processor = VideoProcessor()

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    file_location = UPLOAD_DIR / file.filename
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    processed_filename = f"processed_{file.filename}"
    processed_location = PROCESSED_DIR / processed_filename

    # Process the video
    output_path, match_data = processor.process_video(file_location, processed_location)

    # Save match data
    json_filename = f"processed_{file.filename}.json"
    json_location = PROCESSED_DIR / json_filename
    import json
    with open(json_location, "w") as f:
        json.dump(match_data, f)

    return JSONResponse({
        "filename": file.filename,
        "video_url": f"/processed/{processed_filename}",
        "data_url": f"/processed/{json_filename}"
    })

@app.post("/connect_ip_camera")
async def connect_ip_camera(request_data: dict):
    """Connect to an IP camera via RTSP URL"""
    url = request_data.get("url")
    if not url:
        return JSONResponse({"success": False, "error": "No URL provided"})

    try:
        # In a real implementation, you would:
        # 1. Validate the RTSP URL
        # 2. Start a stream processing thread
        # 3. Return a WebRTC or HLS stream URL

        # For now, we'll simulate success
        return JSONResponse({
            "success": True,
            "stream_url": f"/stream/{hash(url) % 1000}",  # Mock stream URL
            "message": "IP camera connected successfully"
        })
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)})

# WebSocket endpoint for real-time analysis
@app.websocket("/ws/analyze")
async def websocket_analyze(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_json()

            if data["type"] == "frame":
                # Decode base64 image
                import base64
                image_data = base64.b64decode(data["data"])
                nparr = np.frombuffer(image_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                # Run YOLO detection
                results = processor.model(frame, verbose=False)

                # Process detections
                detections = []
                if results[0].boxes is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    classes = results[0].boxes.cls.cpu().numpy()
                    confidences = results[0].boxes.conf.cpu().numpy()

                    for box, cls, conf in zip(boxes, classes, confidences):
                        class_name = "person" if int(cls) == 0 else "ball" if int(cls) == 32 else "unknown"
                        detections.append({
                            "bbox": box.tolist(),
                            "class": class_name,
                            "confidence": float(conf)
                        })

                # Send detections back
                await websocket.send_json({
                    "type": "detections",
                    "detections": detections
                })

    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
