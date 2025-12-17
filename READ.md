# AI Football Coach

An AI-powered football analytics platform that provides comprehensive match analysis from pre-match strategy to real-time tracking and post-match insights.

<img width="1825" height="814" alt="image" src="https://github.com/user-attachments/assets/d9f2b477-b463-444d-b9db-9197f1bb45b7" />

<img width="1565" height="736" alt="image" src="https://github.com/user-attachments/assets/31214a16-1dab-41c0-8d0f-8fd103104718" />



## Features

### Pre-Match Analysis
- Interactive formation planning with visual tactics board
- Strategy analysis with charts comparing strengths and weaknesses
- Tactical recommendations based on formation and playing style
- Preparation checklists for match readiness

  <img width="1695" height="736" alt="image" src="https://github.com/user-attachments/assets/66b908e2-a9c8-40c8-9eda-d40329d1d47a" />
  <img width="993" height="1154" alt="image" src="https://github.com/user-attachments/assets/4425a597-79ca-4800-b5ed-d347fed6564c" />



### Live Match Analysis
- Real-time video streaming from cameras
- AI-powered player and ball detection using YOLO
- Live bounding boxes and tracking overlays
- Support for USB cameras, IP cameras, and WiFi streaming

<img width="1406" height="454" alt="image" src="https://github.com/user-attachments/assets/822f760d-d7e3-468e-b474-b19096ef23af" />



### Post-Match Analysis
- Video upload and processing
- Automated possession and touch statistics
- Heatmaps and player positioning
- Detailed match reports with suggestions

  <img width="1585" height="580" alt="image" src="https://github.com/user-attachments/assets/ae0b0a95-7485-4bbb-9016-36b74af29640" />


## Technologies Used

- **Backend**: FastAPI (Python)
- **AI/ML**: YOLOv8 for object detection
- **Frontend**: HTML, Tailwind CSS, JavaScript, Chart.js
- **Real-time**: WebSockets for live analysis
- **Video Processing**: OpenCV

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai-football-coach.git
cd ai-football-coach
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Download the YOLO model:
```bash
# The yolov8n.pt file should be in the root directory
# If not, download from https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

## Usage

1. Start the server:
```bash
python -m uvicorn app.main:app --reload
```

2. Open your browser and go to `http://localhost:8000`

3. Navigate through the sections:
   - **Pre-Match**: Plan formations and analyze strategies
   - **Live Match**: Connect cameras for real-time analysis
   - **Post-Match**: Upload videos for detailed insights

## Camera Setup

### USB Camera
- Connect any USB webcam or DSLR to your computer
- Click "USB Camera" in Live Match section

### IP Camera
- Enter RTSP URL for network cameras
- Example: `rtsp://192.168.1.100:554/stream`

### Mobile Camera (WiFi)
- Use apps like IP Webcam on your phone
- Start server and enter the video URL
- Example: `http://192.168.1.100:8080/video.mjpeg`

## Project Structure

```
ai-football-coach/
├── app/
│   ├── main.py          # FastAPI application
│   ├── processor.py     # Video processing and AI logic
│   ├── templates/
│   │   └── index.html   # Main UI
│   └── __init__.py
├── processed/           # Processed videos and data
├── uploads/             # Uploaded videos
├── requirements.txt     # Python dependencies
├── yolov8n.pt          # YOLO model
└── README.md
```



## API Endpoints

- `GET /` - Main application
- `POST /upload` - Upload video for processing
- `WebSocket /ws/analyze` - Real-time analysis
- `GET /processed/{filename}` - Access processed videos

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## Acknowledgments

- YOLOv8 by Ultralytics
- FastAPI framework
- Tailwind CSS
- Chart.js for visualizations
