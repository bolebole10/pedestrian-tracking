# Pedestrian Tracking with YOLO and SORT

This project implements pedestrian tracking using the YOLOv8 object detection model and the SORT tracking algorithm. It processes video input to detect and track pedestrians in real-time.

## Requirements

- Python 3.12
- YOLOv8 (via `ultralytics` library)
- OpenCV
- NumPy
- SORT dependencies (see `sort/requirements.txt`)

## Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd pedestrian-tracking
   ```

2. Activate the virtual environment:
   ```bash
   source pytorch_env/bin/activate  # Linux/Mac
   pytorch_env\Scripts\activate     # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r sort/requirements.txt
   pip install ultralytics opencv-python numpy
   ```

## Usage

Run the main script to process a video and track pedestrians:
```bash
python tracking-YOLO-SORT.py
```

