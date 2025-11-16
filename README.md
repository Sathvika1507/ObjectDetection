# 🚀 Task 4 — Object Detection & Tracking System  
### CodeAlpha Artificial Intelligence Internship  
### Developed by: **Sathvika Bogam**

This project implements a **real-time Object Detection + Tracking System** using:

- **YOLOv8 (Ultralytics)** for object detection  
- **Custom SimpleTracker (IoU-based)** for object tracking  
- Supports **Webcam**, **Video files**, and **Single Image input**  
- Exports results to **CSV**, saves **output video**, displays **FPS**, and shows **Tracking IDs**

This is the **upgraded and most stable version** designed to run smoothly on **macOS** using **Python 3.11**.

---

## ⭐ Features

### 🔍 **Object Detection**
- Detects **ALL 80 COCO classes**  
- Accurate bounding boxes  
- Class name + confidence displayed  
- Model auto-download support

### 🎯 **Object Tracking**
- Assigns unique **Track IDs**
- Uses a lightweight **IOU-based tracker**
- No external installations needed (works offline)
- Stable tracking for multiple objects

### 🎦 **Modes Supported**
| Mode | Description |
|------|-------------|
| **Webcam Mode** | Real-time detection & tracking |
| **Video Mode** | Detect objects in any `.mp4` file |
| **Image Mode** | Detect objects in a single image |

### 💾 **Output Features**
- Saves processed video as: `detections_output.mp4`
- Exports object tracking data to: `detection_results.csv`
- Shows **FPS** on screen
- Clean and readable bounding box labels

---

## 📁 Project Structure


Task4_ObjectDetection/
│
├── detect_tracker.py # Main object detection + tracking script
├── simple_tracker.py # Lightweight IOU-based tracker
├── requirements.txt # Python dependencies
├── input_video.mp4 # (optional) place your video here
└── README.md # Project documentation

---

## 🛠 Installation

### 📌 1. Navigate to the project folder
```bash
cd /Users/bogamsathvika/Desktop/CodeAlpha_AI_Projects/Task4_ObjectDetection
📌 2. Install all dependencies (USE PYTHON 3.11)
/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 -m pip install -r requirements.txt
▶️ Running the Program
1️⃣ Webcam Mode (Default)
Just run:
/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 detect_tracker.py
Press q to quit.
2️⃣ Video Mode
Open detect_tracker.py and set:
USE_WEBCAM = False
VIDEO_PATH = "input_video.mp4"
Then run:
python3 detect_tracker.py
3️⃣ Image Mode
Set this in the script:
USE_WEBCAM = False
IMAGE_PATH = "your_image.jpg"
📦 Output Files
After running the script, you will get:
🎥 detections_output.mp4
Saved processed video
Includes boxes, labels, and tracking IDs
🧾 detection_results.csv
Each row contains:
timestamp, frame_idx, track_id, class, confidence, x1, y1, x2, y2
Useful for:
Analysis
Research
Reports
Dataset generation
🔐 macOS Camera Permission Fix (Important)
If webcam shows:
OpenCV: camera failed to initialize
Then:
Run Python once:
/Library/Frameworks/Python.framework/Versions/3.11/Resources/Python.app/Contents/MacOS/Python
Go to:
System Settings → Privacy & Security → Camera → Enable Python
If still not working:
sudo tccutil reset Camera
Then run the script again and click Allow.
🧠 Tracking Logic :
The custom SimpleTracker uses:
Intersection-over-Union (IoU) matching
Track persistence
Auto ID assignment
Age-out system for lost objects
Advantages:
✔ No installation issues
✔ Works offline
✔ Lightweight and fast
✔ Perfect for internship tasks
🎯 Demo Tips 
To create an impressive demonstration:
Run webcam detection
Move multiple objects (bottle, pen, hand)
Ensure tracking IDs remain stable
Show FPS at the top-left
Quit → Show detections_output.mp4
Open detection_results.csv in Excel and show data
This gives a strong, professional demo.
🏁 Conclusion
This project showcases:
Real-time AI computer vision
Tracking algorithm development
Python-based image/video processing
YOLOv8 integration
Data export + output video creation
Professional-grade code quality
Perfect for AI internships, computer vision roles, and portfolio projects.
👩‍💻 Developer
Sathvika Bogam
