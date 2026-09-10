# IBVAP - Real Working Prototype

A simple local prototype for the SIH Intelligent Border Video Analytics Platform.

## What works
- Webcam or uploaded video input
- YOLO object detection for person/vehicle/animal
- Bounding boxes and confidence
- Virtual restricted zone
- Intrusion alert when a detected person enters the zone
- Dynamic threat score
- Event log
- Offline mode simulation
- Camera/system health display

## Run
1. Install Python 3.10+.
2. Open a terminal in this folder.
3. Run: `pip install -r requirements.txt`
4. Run: `python app.py`
5. Open: http://127.0.0.1:5000

The first YOLO run downloads the small pretrained model automatically when internet is available.
For a demo without a camera, upload an MP4 from the dashboard.

## Important
This is a demonstration prototype, not a production border-security system. Face recognition/identity matching and ANPR are intentionally represented as modules to integrate later and should only be used with appropriate authorization, privacy controls and approved data.
