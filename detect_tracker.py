# detect_tracker.py
import os
import time
import csv
from datetime import datetime
import cv2
import numpy as np
from ultralytics import YOLO
from simple_tracker import SimpleTracker

# --- Configuration ---
MODEL = "yolov8n.pt"   # change to yolov8s.pt / yolov8m.pt for better accuracy
CONF_THRESHOLD = 0.25
SAVE_OUTPUT_VIDEO = True
OUTPUT_VIDEO_NAME = "detections_output.mp4"
EXPORT_CSV = True
CSV_NAME = "detection_results.csv"
USE_WEBCAM = True    # if False will try to read video_path or image_path
VIDEO_PATH = "input_video.mp4"   # set to your video file if not using webcam
IMAGE_PATH = None    # set to a path to process a single image

# --- Init model & tracker ---
print("Loading model...")
model = YOLO(MODEL)  # will auto-download if missing
tracker = SimpleTracker(iou_threshold=0.3, max_age=18)

# --- Helper to draw box/label ---
def draw_box(frame, box, cls_name, conf, tid=None, color=(0,255,0)):
    x1,y1,x2,y2 = map(int, box)
    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
    label = f"{cls_name} {conf:.2f}"
    if tid is not None:
        label = f"ID:{tid} {label}"
    (w,h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - 18), (x1 + w, y1), color, -1)
    cv2.putText(frame, label, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

# --- Prepare output writer later ---
out_writer = None
frame_size = None
fps_display = 0

# CSV file writer
csv_file = None
csv_writer = None
if EXPORT_CSV:
    csv_file = open(CSV_NAME, mode='w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["timestamp","frame_idx","track_id","class","confidence","x1","y1","x2","y2"])

# Input source
if IMAGE_PATH:
    source_mode = "image"
elif not USE_WEBCAM:
    source_mode = "video"
else:
    source_mode = "webcam"

if source_mode == "webcam":
    cap = cv2.VideoCapture(0)
elif source_mode == "video":
    cap = cv2.VideoCapture(VIDEO_PATH)
else:
    cap = None

frame_idx = 0
start_time = time.time()

try:
    if source_mode == "image":
        # process one image
        frame = cv2.imread(IMAGE_PATH)
        results = model(frame)[0]
        detections = []
        for box, conf, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
            conf = float(conf)
            if conf < CONF_THRESHOLD: 
                continue
            x1,y1,x2,y2 = map(int, box.tolist())
            detections.append([x1,y1,x2,y2, conf, int(cls)])
        boxes_only = [d[:4] for d in detections]
        tracks = tracker.update(boxes_only)
        # apply IoU matching same as video mode
        for t in tracks:
            best_det = None
            best_iou = 0
            for d in detections:
                ax1,ay1,ax2,ay2 = t.box
                bx1,by1,bx2,by2 = d[0],d[1],d[2],d[3]
                ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
                ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
                iw = max(0, ix2-ix1); ih = max(0, iy2-iy1)
                inter = iw*ih
                aarea = max(0,(ax2-ax1)) * max(0,(ay2-ay1))
                barea = max(0,(bx2-bx1)) * max(0,(by2-by1))
                union = aarea + barea - inter
                iou_val = inter/union if union>0 else 0.0
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_det = d
            if best_det is not None:
                cls_id = int(best_det[5])
                conf_val = float(best_det[4])
                cls_name = model.names.get(cls_id, str(cls_id))
            else:
                cls_name = "object"; conf_val = 0.0
            draw_box(frame, t.box, cls_name, conf_val, tid=t.id)
            if EXPORT_CSV and csv_writer:
                ts = datetime.now().isoformat()
                x1,y1,x2,y2 = t.box
                csv_writer.writerow([ts, frame_idx, t.id, cls_name, conf_val, x1, y1, x2, y2])
        cv2.imshow("Result", frame)
        cv2.waitKey(0)

    else:
        # realtime / video loop
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            # inference
            results = model(frame)[0]
            detections = []
            for r in results.boxes:
                box = r.xyxy.cpu().numpy().astype(int)[0]
                conf = float(r.conf.cpu().numpy()[0]) if hasattr(r, 'conf') else float(r.conf)
                cls = int(r.cls.cpu().numpy()[0]) if hasattr(r, 'cls') else int(r.cls)
                if conf < CONF_THRESHOLD:
                    continue
                x1,y1,x2,y2 = box.tolist()
                detections.append([x1,y1,x2,y2, conf, cls])
            boxes_only = [d[:4] for d in detections]
            tracks = tracker.update(boxes_only)
            # draw detections + track ids
            for t in tracks:
                best_det = None
                best_iou = 0
                for d in detections:
                    ax1,ay1,ax2,ay2 = t.box
                    bx1,by1,bx2,by2 = d[0],d[1],d[2],d[3]
                    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
                    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
                    iw = max(0, ix2-ix1); ih = max(0, iy2-iy1)
                    inter = iw*ih
                    aarea = max(0,(ax2-ax1)) * max(0,(ay2-ay1))
                    barea = max(0,(bx2-bx1)) * max(0,(by2-by1))
                    union = aarea + barea - inter
                    iou_val = inter/union if union>0 else 0.0
                    if iou_val > best_iou:
                        best_iou = iou_val
                        best_det = d
                if best_det is not None:
                    cls_id = int(best_det[5])
                    conf_val = float(best_det[4])
                    cls_name = model.names.get(cls_id, str(cls_id))
                else:
                    cls_name = "object"; conf_val = 0.0
                draw_box(frame, t.box, cls_name, conf_val, tid=t.id)
                if EXPORT_CSV and csv_writer:
                    ts = datetime.now().isoformat()
                    x1,y1,x2,y2 = t.box
                    csv_writer.writerow([ts, frame_idx, t.id, cls_name, conf_val, x1, y1, x2, y2])
            # FPS
            elapsed = time.time() - start_time
            fps_display = frame_idx / elapsed if elapsed>0 else 0.0
            cv2.putText(frame, f"FPS: {fps_display:.2f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            cv2.imshow("YOLOv8 + SimpleTracker", frame)
            # Prepare writer
            if SAVE_OUTPUT_VIDEO:
                if out_writer is None:
                    h,w = frame.shape[:2]
                    frame_size = (w,h)
                    out_writer = cv2.VideoWriter(OUTPUT_VIDEO_NAME, cv2.VideoWriter_fourcc(*"mp4v"), 20, frame_size)
                out_writer.write(frame)
            # key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # cleanup
        if out_writer:
            out_writer.release()
        if csv_file:
            csv_file.close()
        cap.release()
        cv2.destroyAllWindows()

except Exception as e:
    print("Error:", e)
    if csv_file:
        csv_file.close()
    if out_writer:
        out_writer.release()
    if source_mode != "image" and 'cap' in locals():
        cap.release()
    cv2.destroyAllWindows()
