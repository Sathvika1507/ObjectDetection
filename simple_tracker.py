# simple_tracker.py
# Lightweight tracker: assigns IDs using IoU + simple lifetime handling

import numpy as np

class Track:
    def __init__(self, tid, box):
        self.id = tid
        self.box = box  # [x1,y1,x2,y2]
        self.hits = 1
        self.age = 0

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA); interH = max(0, yB - yA)
    inter = interW * interH
    boxAArea = max(0,(boxA[2]-boxA[0])) * max(0,(boxA[3]-boxA[1]))
    boxBArea = max(0,(boxB[2]-boxB[0])) * max(0,(boxB[3]-boxB[1]))
    union = boxAArea + boxBArea - inter
    return inter / union if union>0 else 0.0

class SimpleTracker:
    def __init__(self, iou_threshold=0.3, max_age=15):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks = []
        self.next_id = 1

    def update(self, detections):
        """
        detections: list of boxes [x1,y1,x2,y2]
        returns list of track objects (Track)
        """
        updated_tracks = []
        used = set()
        # match detections to existing tracks by IoU
        for det in detections:
            best_iou = 0
            best_t = None
            for t in self.tracks:
                if t in used: 
                    continue
                val = iou(det, t.box)
                if val > best_iou:
                    best_iou = val
                    best_t = t
            if best_iou >= self.iou_threshold and best_t is not None:
                # update that track
                best_t.box = det
                best_t.hits += 1
                best_t.age = 0
                updated_tracks.append(best_t)
                used.add(best_t)
            else:
                # create new track
                tr = Track(self.next_id, det)
                self.next_id += 1
                updated_tracks.append(tr)
        # age previous tracks that were not updated
        for t in self.tracks:
            if t not in used:
                t.age += 1
                if t.age <= self.max_age:
                    updated_tracks.append(t)
        # remove old tracks
        self.tracks = [t for t in updated_tracks if t.age <= self.max_age]
        return self.tracks
