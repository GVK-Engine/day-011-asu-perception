# detect_asu.py
# runs YOLOv8 on all extracted ASU campus frames
# saves detections with confidence, class, bbox, distance
#
# distance estimation uses similar triangles:
#   distance = (real_height * focal_length) / pixel_height
#   works because objects closer = taller in frame
#
# also flags every frame as:
#   DETECTED    at least one object found
#   FAILED      objects visible but nothing detected
#   EMPTY       genuinely nothing there
#
# Nani — MS Robotics ASU

import cv2
import os
import json
import numpy as np
from ultralytics import YOLO

FRAMES_DIR  = r"D:\day-011-asu-perception\data\frames"
RESULTS_DIR = r"D:\day-011-asu-perception\data\detections"
MODEL_PATH  = "yolov8x.pt"

CONF_THRESH = 0.25   # minimum confidence to count

# iPhone 15 Pro Max approximate focal length in pixels
# calculated from 77 degree horizontal FOV at 1920px wide
# focal_length = (image_width / 2) / tan(FOV/2)
FOCAL_LENGTH = 1050.0

# real-world heights in meters for distance estimation
OBJECT_HEIGHTS = {
    "person":     1.70,
    "car":        1.50,
    "truck":      3.50,
    "bus":        3.20,
    "bicycle":    1.10,
    "motorcycle": 1.20,
}

# COCO classes that matter for AV perception
AV_CLASSES = {
    "person", "bicycle", "car", "motorcycle",
    "bus", "truck", "traffic light", "stop sign"
}


def estimate_distance(class_name, bbox_h_px):
    """
    Estimate distance to object using similar triangles.

    bbox_h_px: height of detection box in pixels
    returns:   distance in meters (None if unknown class)

    The math:
      A person 1.7m tall at distance D
      appears bbox_h_px pixels tall in the image.
      focal_length = pixels per meter at 1m distance
      distance = (real_height * focal_length) / bbox_h_px
    """
    real_h = OBJECT_HEIGHTS.get(class_name)
    if real_h is None or bbox_h_px < 5:
        return None
    return round((real_h * FOCAL_LENGTH) / bbox_h_px, 1)


def run_detection(scenario_dir, model, scenario_name):
    """
    Run YOLOv8 on all frames in one scenario folder.
    Returns list of per-frame detection results.
    """
    frames = sorted([
        f for f in os.listdir(scenario_dir)
        if f.endswith('.jpg')
    ])

    results = []

    for fname in frames:
        fpath = os.path.join(scenario_dir, fname)
        img   = cv2.imread(fpath)
        if img is None:
            continue

        H, W = img.shape[:2]

        # run YOLO
        preds = model(fpath, conf=CONF_THRESH,
                      verbose=False)[0]

        detections = []
        for box in preds.boxes:
            cls_id = int(box.cls[0])
            cls_nm = model.names[cls_id]
            conf   = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            bbox_h  = y2 - y1
            dist    = estimate_distance(cls_nm, bbox_h)

            # distance band
            if dist is None:
                band = "unknown"
            elif dist < 5:
                band = "0-5m"
            elif dist < 15:
                band = "5-15m"
            elif dist < 30:
                band = "15-30m"
            else:
                band = "30m+"

            detections.append({
                "class":      cls_nm,
                "confidence": round(conf, 3),
                "bbox":       [round(x1), round(y1),
                               round(x2), round(y2)],
                "bbox_h_px":  round(bbox_h),
                "distance_m": dist,
                "dist_band":  band,
                "is_av":      cls_nm in AV_CLASSES,
            })

        results.append({
            "frame":      fname,
            "scenario":   scenario_name,
            "n_det":      len(detections),
            "detections": detections,
        })

    return results


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    model     = YOLO(MODEL_PATH)
    scenarios = sorted(os.listdir(FRAMES_DIR))

    print(f"running detection on {len(scenarios)} scenarios")
    print(f"model: YOLOv8n  conf threshold: {CONF_THRESH}")

    all_results  = {}
    summary      = {}

    for scenario in scenarios:
        s_dir = os.path.join(FRAMES_DIR, scenario)
        if not os.path.isdir(s_dir):
            continue

        print(f"\n  {scenario}")
        results = run_detection(s_dir, model, scenario)

        # compute summary stats
        n_frames   = len(results)
        n_with_det = sum(1 for r in results
                         if r['n_det'] > 0)
        total_det  = sum(r['n_det'] for r in results)

        # per-class counts
        class_counts = {}
        conf_vals    = []
        dist_vals    = []

        for r in results:
            for d in r['detections']:
                c = d['class']
                class_counts[c] = \
                    class_counts.get(c, 0) + 1
                conf_vals.append(d['confidence'])
                if d['distance_m']:
                    dist_vals.append(d['distance_m'])

        det_rate = n_with_det / n_frames * 100 \
                   if n_frames > 0 else 0
        avg_conf = float(np.mean(conf_vals)) \
                   if conf_vals else 0
        avg_dist = float(np.mean(dist_vals)) \
                   if dist_vals else 0

        summary[scenario] = {
            "n_frames":    n_frames,
            "n_detected":  n_with_det,
            "det_rate":    round(det_rate, 1),
            "total_det":   total_det,
            "avg_conf":    round(avg_conf, 3),
            "avg_dist_m":  round(avg_dist, 1),
            "class_counts": class_counts,
        }

        all_results[scenario] = results

        print(f"    {n_frames} frames  "
              f"{det_rate:.1f}% detected  "
              f"avg conf {avg_conf:.2f}")
        print(f"    classes: {class_counts}")

    # save all detections
    det_path = os.path.join(RESULTS_DIR,
                             "all_detections.json")
    with open(det_path, 'w') as f:
        json.dump(all_results, f)

    sum_path = os.path.join(RESULTS_DIR, "summary.json")
    with open(sum_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nresults saved to {RESULTS_DIR}")
    print(f"\nSUMMARY:")
    print(f"  {'scenario':<30} {'det%':>6} "
          f"{'conf':>6} {'dist':>6}")
    print(f"  {'-'*52}")
    for s, v in summary.items():
        print(f"  {s:<30} {v['det_rate']:>5.1f}%"
              f"  {v['avg_conf']:>5.3f}"
              f"  {v['avg_dist_m']:>5.1f}m")