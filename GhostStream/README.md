# GhostStream

## Overview
GhostStream is a real-time computer vision application that removes human subjects from a moving camera feed, effectively creating an "invisible cloak" effect. Unlike static background removal tools (like Zoom/Google Meet), GhostStream supports dynamic camera movement (panning/tilting) by tracking the environment and warping historical background data to fit the current perspective.

## Features


## Project Structure
```
video-background-remover/
├── src/
│   ├── capture/
│   │   ├── camera_stream.py    # Raw camera access
│   │   └── worker.py           # Background thread (QThread) for handling AI
│   ├── gui/
│   │   ├── app_window.py       # Main UI (Side-by-side view)
│   │   └── selector.py         # Camera selection dialog
│   ├── motion/
│   │   └── motion_estimator.py # ORB + Homography logic
│   ├── processing/
│   │   ├── background_model.py # Buffer management & Inpainting logic
│   │   └── pipeline.py         # Coordinator (connects YOLO, Motion, & BG Model)
│   ├── segmentation/
│   │   └── segmentation_engine.py # YOLOv8 wrapper
│   └── main.py                 # Application Entry Point
├── requirements.txt
└── README.md
```

## Installation
1. Create the environment named 'GhostStream':
   ```
   python3 -m venv GhostStream
   ```
2. Activate the environment:
   ```
   source GhostStream/bin/activate
   ```
3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage


## Contributing


