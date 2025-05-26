🕳️ Real-Time Pothole Detection using YOLOv8

This project implements a **real-time pothole detection and tracking system** using the YOLOv8 object detection model. The application processes video input (e.g., dashcam footage), identifies potholes in each frame, tracks them across frames, and saves the annotated output as an `.mp4` video.

 🚀 Features

- ✅ Real-time pothole detection using a trained **YOLOv8** model (`best2.pt`)
- ✅ Object tracking with persistent IDs to monitor potholes across frames
- ✅ Processes every 3rd frame for faster performance
- ✅ Annotated bounding boxes with class labels and unique object IDs
- ✅ Outputs a high-resolution `.mp4` video with all detections


🧠 Model

- Model Used: `YOLOv8` (Ultralytics)
- Format: PyTorch model file (`best2.pt`)
- Input: RGB video frames
- Output: Detected potholes with bounding boxes and class labels

---

 🖥️ Requirements

- Python 3.8+
- OpenCV
- Ultralytics YOLO

Install dependencies:

```bash
pip install ultralytics opencv-python
