import cv2
import numpy as np
from ultralytics import YOLO
from sort.sort import Sort

# Load YOLOv8n model
model = YOLO('yolov8n.pt')  # downloads automatically if not present

# Video input (change if needed)
video_path = 'test-videos/video1.mp4'
cap = cv2.VideoCapture(video_path)

# Tracker
tracker = Sort()
frame_id = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for performance
    frame = cv2.resize(frame, (640, 360))

    detections = []
    if frame_id % 1 == 0:
        results = model(frame, verbose=False)[0]
        for box in results.boxes:
            if int(box.cls) == 0:  # class 0 = person
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf)
                detections.append([x1, y1, x2, y2, conf])

   # Track objects only if there are detections
    if len(detections) > 0:
        tracked = tracker.update(np.array(detections))
        for obj in tracked:
            x1, y1, x2, y2, track_id = map(int, obj)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'ID {track_id}', (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        tracked = []
    # Show live
    cv2.imshow("YOLOv8 Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_id += 1

cap.release()
cv2.destroyAllWindows()
