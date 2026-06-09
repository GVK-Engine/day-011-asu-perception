# visualize.py
# Visual outputs for Day 11: ASU Campus Perception
#
# outputs:
#   1. detection_demo.mp4      YOLOv8 boxes on all ASU footage
#   2. failure_highlights.gif  best failure moments compiled
#   3. germany_vs_asu.gif      KITTI Germany vs ASU Arizona
#   4. failure_heatmap.mp4     spatial failure overlay
#
# Vamshikrishna Gadde
# MS Robotics and Autonomous Systems, ASU

import cv2
import json
import numpy as np
import os
import warnings
from pathlib import Path

import imageio.v2 as imageio
from ultralytics import YOLO

warnings.filterwarnings("ignore")

BASE      = Path(r"D:\day-011-asu-perception")
FRAMES    = BASE / "data" / "frames"
RAW       = BASE / "data" / "raw_footage"
DET_DIR   = BASE / "data" / "detections"
RES_DIR   = BASE / "results"
KITTI_DIR = Path(
    r"C:\Users\vamsh\Downloads\kitti"
    r"\2011_09_26_drive_0001_sync"
    r"\2011_09_26"
    r"\2011_09_26_drive_0001_sync"
    r"\image_02\data"
)
MODEL_PT  = BASE / "yolov8x.pt"
RES_DIR.mkdir(exist_ok=True)

AUTHOR = "Vamshikrishna Gadde | MS Robotics ASU"

# BGR colors for OpenCV
COL = {
    "bg":       (46,  26,  26),
    "white":    (255, 255, 255),
    "grey":     (160, 160, 160),
    "green":    (60,  210,   0),
    "yellow":   (0,   200, 230),
    "red":      (30,   30, 220),
    "cyan":     (255, 220,   0),
    "orange":   (35,  107, 255),
    "purple":   (200,   0, 160),
}

# class colors for detection boxes (BGR)
CLASS_COL = {
    "person":        (60,  210,   0),
    "car":           (255, 220,   0),
    "truck":         (35,  107, 255),
    "bus":           (200, 160,   0),
    "bicycle":       (0,   200, 230),
    "motorcycle":    (0,   230, 200),
    "traffic light": (0,   255, 255),
    "stop sign":     (30,   30, 220),
    "train":         (200,   0, 160),
    "fire hydrant":  (0,   140, 255),
    "bench":         (128, 128, 128),
    "potted plant":  (0,   180,  60),
}
DEFAULT_COL = (200, 200, 200)

FONT       = cv2.FONT_HERSHEY_SIMPLEX
CONF_THRESH = 0.25

# known misclassifications for labeling
MISCLASS = {
    "fire hydrant": "bollard",
    "potted plant": "desert plant",
    "train":        "ASU tram",
    "bench":        "ledge/wall",
    "skateboard":   "scooter",
    "kite":         "glare",
    "sports ball":  "lens flare",
    "giraffe":      "person?",
}

SCENARIOS = [
    "CART_AND_SLOW_WALK",
    "CYCLES_AND_CARS",
    "MU_MULTI_PEOPLE",
    "PED_AND_SLOW_WALKING",
    "PED_CROSS_AND_CARS",
    "SUN_GLARE",
    "TRUCK_AND_BUS",
    "VID_WITH_TRAM",
]


def load_detections():
    with open(DET_DIR / "all_detections.json") as f:
        return json.load(f)


def get_frame_paths(scenario):
    folder = FRAMES / scenario
    paths  = sorted(folder.glob("*.jpg"))
    return paths


def put_text_bg(img, text, x, y, scale=0.55,
                color=(255,255,255), thickness=1):
    (w, h), _ = cv2.getTextSize(
        text, FONT, scale, thickness)
    cv2.rectangle(img, (x-2, y-h-4),
                  (x+w+2, y+4), (0,0,0), -1)
    cv2.putText(img, text, (x, y), FONT,
                scale, color, thickness,
                cv2.LINE_AA)


def draw_boxes(frame, detections):
    for d in detections:
        cls  = d["class"]
        conf = d["confidence"]
        x1, y1, x2, y2 = d["bbox"]
        dist = d.get("distance_m")
        col  = CLASS_COL.get(cls, DEFAULT_COL)

        cv2.rectangle(frame, (x1,y1), (x2,y2),
                      col, 2)

        real_name = MISCLASS.get(cls, cls)
        label = f"{real_name} {conf:.2f}"
        if dist:
            label += f" {dist:.0f}m"

        put_text_bg(frame, label, x1, y1-6,
                    0.45, col, 1)
    return frame


def pad_to_width(img, target_w):
    h, w = img.shape[:2]
    if w >= target_w:
        return cv2.resize(img, (target_w, h))
    pad   = target_w - w
    left  = pad // 2
    right = pad - left
    return cv2.copyMakeBorder(
        img, 0, 0, left, right,
        cv2.BORDER_CONSTANT, value=(0,0,0))


def make_detection_demo():
    print("[1/4] detection_demo.mp4")
    all_det = load_detections()
    model   = YOLO(str(MODEL_PT))

    out_path   = str(RES_DIR / "detection_demo.mp4")
    writer     = None
    frame_size = None

    for scenario in SCENARIOS:
        paths = get_frame_paths(scenario)
        dets  = all_det.get(scenario, [])
        det_map = {d["frame"]: d["detections"]
                   for d in dets}

        print(f"  {scenario} ({len(paths)} frames)")

        for fi, fpath in enumerate(paths):
            frame = cv2.imread(str(fpath))
            if frame is None:
                continue

            H, W   = frame.shape[:2]
            fname  = fpath.name
            d_list = det_map.get(fname, [])

            frame = draw_boxes(frame, d_list)

            # info bar at top
            bar = np.zeros((36, W, 3), dtype=np.uint8)
            bar[:] = COL["bg"]
            scene_label = scenario.replace("_", " ")
            n_det = len(d_list)
            put_text_bg(bar,
                f"{scene_label}  |  frame {fi+1}/{len(paths)}"
                f"  |  {n_det} detections",
                8, 24, 0.55, COL["white"])
            put_text_bg(bar, AUTHOR,
                W - 320, 24, 0.45, COL["grey"])

            panel = np.vstack([bar, frame])

            if writer is None:
                frame_size = (panel.shape[1],
                              panel.shape[0])
                writer = cv2.VideoWriter(
                    out_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    10, frame_size)

            writer.write(panel)

    if writer:
        writer.release()
        print(f"  saved {out_path}")


def make_failure_gif():
    print("[2/4] failure_highlights.gif")
    all_det = load_detections()

    # failure moments to highlight
    # (scenario, description, color, frames_to_show)
    targets = [
        ("CART_AND_SLOW_WALK",
         "GOLF CART: 0 detections (vocabulary gap)",
         COL["red"], 20),
        ("SUN_GLARE",
         "SUN GLARE: 40.9% det rate  glare = lens flare",
         COL["orange"], 20),
        ("VID_WITH_TRAM",
         "ASU TRAM: detected as TRAIN (wrong class)",
         COL["yellow"], 20),
        ("PED_AND_SLOW_WALKING",
         "BOLLARD: detected as FIRE HYDRANT (442 times)",
         COL["cyan"], 20),
        ("MU_MULTI_PEOPLE",
         "DESERT PLANT: detected as POTTED PLANT",
         COL["green"], 20),
    ]

    gif_frames = []
    GIF_W, GIF_H = 960, 576

    for scenario, desc, color, n in targets:
        paths   = get_frame_paths(scenario)
        dets    = all_det.get(scenario, [])
        det_map = {d["frame"]: d["detections"]
                   for d in dets}

        # pick frames spread across the scenario
        indices = np.linspace(0, len(paths)-1,
                              n, dtype=int)

        for idx in indices:
            fpath = paths[idx]
            frame = cv2.imread(str(fpath))
            if frame is None:
                continue

            frame = draw_boxes(
                frame, det_map.get(fpath.name, []))

            # resize maintaining aspect ratio
            h, w = frame.shape[:2]
            scale = min(GIF_W/w, GIF_H/h)
            nw    = int(w * scale)
            nh    = int(h * scale)
            frame = cv2.resize(frame, (nw, nh))

            # pad to GIF size
            canvas = np.zeros(
                (GIF_H, GIF_W, 3), dtype=np.uint8)
            canvas[:] = COL["bg"]
            y0 = (GIF_H - nh) // 2
            x0 = (GIF_W - nw) // 2
            canvas[y0:y0+nh, x0:x0+nw] = frame

            # label bar at bottom
            cv2.rectangle(canvas,
                          (0, GIF_H-50),
                          (GIF_W, GIF_H),
                          COL["bg"], -1)
            put_text_bg(canvas, desc,
                        8, GIF_H-28,
                        0.55, color, 2)
            put_text_bg(canvas, AUTHOR,
                        8, GIF_H-8,
                        0.4, COL["grey"], 1)

            rgb = cv2.cvtColor(canvas,
                               cv2.COLOR_BGR2RGB)
            gif_frames.append(rgb)

    if gif_frames:
        out = str(RES_DIR / "failure_highlights.gif")
        imageio.mimsave(out, gif_frames,
                        duration=0.15, loop=0)
        print(f"  saved {out}")


def make_domain_shift_gif():
    print("[3/4] germany_vs_asu.gif")

    kitti_paths = sorted(KITTI_DIR.glob("*.png"))
    if not kitti_paths:
        kitti_paths = sorted(KITTI_DIR.glob("*.jpg"))

    if not kitti_paths:
        print("  KITTI frames not found, skipping")
        return

    all_det = load_detections()
    model   = YOLO(str(MODEL_PT))

    # pick 5 interesting ASU frames per scenario
    asu_frames = []
    for scenario in SCENARIOS:
        paths = get_frame_paths(scenario)
        dets  = all_det.get(scenario, [])
        det_map = {d["frame"]: d["detections"]
                   for d in dets}
        if not paths:
            continue
        indices = np.linspace(0, len(paths)-1,
                              5, dtype=int)
        for idx in indices:
            asu_frames.append(
                (paths[idx],
                 det_map.get(paths[idx].name, []),
                 scenario))

    # pick matching number of KITTI frames
    n = min(len(asu_frames), len(kitti_paths), 40)
    kitti_sel = [kitti_paths[i]
                 for i in np.linspace(
                     0, len(kitti_paths)-1,
                     n, dtype=int)]
    asu_sel   = asu_frames[:n]

    GIF_W = 1280
    ROW_H = 360   # height for each row
    PANEL_W = GIF_W // 2

    gif_frames = []

    for i, (k_path, (a_path, a_dets, scenario)) in \
            enumerate(zip(kitti_sel, asu_sel)):

        # KITTI frame with detections
        k_img = cv2.imread(str(k_path))
        if k_img is None:
            continue
        k_preds = model(str(k_path),
                        conf=CONF_THRESH,
                        verbose=False)[0]
        k_dets  = []
        for box in k_preds.boxes:
            cls_id = int(box.cls[0])
            cls_nm = model.names[cls_id]
            conf   = float(box.conf[0])
            x1,y1,x2,y2 = [int(v)
                            for v in box.xyxy[0]]
            k_dets.append({
                "class": cls_nm,
                "confidence": conf,
                "bbox": [x1,y1,x2,y2],
                "distance_m": None,
            })
        k_img = draw_boxes(k_img, k_dets)

        # ASU frame with detections
        a_img = cv2.imread(str(a_path))
        if a_img is None:
            continue
        a_img = draw_boxes(a_img, a_dets)

        # resize both to same height
        # pad width to PANEL_W
        def resize_pad(img, target_h, target_w):
            h, w   = img.shape[:2]
            scale  = target_h / h
            nw     = int(w * scale)
            img    = cv2.resize(img, (nw, target_h))
            return pad_to_width(img, target_w)

        k_panel = resize_pad(k_img, ROW_H, PANEL_W)
        a_panel = resize_pad(a_img, ROW_H, PANEL_W)

        # label bars
        def label_bar(text, sub, color, w):
            bar = np.zeros((28, w, 3), dtype=np.uint8)
            bar[:] = COL["bg"]
            put_text_bg(bar, text, 6, 20,
                        0.55, color, 1)
            put_text_bg(bar, sub,
                        w - len(sub)*7 - 10, 20,
                        0.4, COL["grey"], 1)
            return bar

        k_n   = len(k_dets)
        a_n   = len(a_dets)
        s_lbl = scenario.replace("_", " ")

        k_bar = label_bar(
            f"KITTI Germany  Overcast  HDL-64E LiDAR",
            f"{k_n} det", COL["cyan"], PANEL_W)
        a_bar = label_bar(
            f"ASU Arizona  Direct Sun  iPhone  {s_lbl}",
            f"{a_n} det", COL["orange"], PANEL_W)

        k_col = np.vstack([k_bar, k_panel])
        a_col = np.vstack([a_bar, a_panel])

        # separator line
        sep = np.zeros(
            (k_col.shape[0], 3, 3), dtype=np.uint8)
        sep[:] = (80, 80, 80)

        combined = np.hstack([k_col, sep, a_col])

        # title bar
        title = np.zeros(
            (30, combined.shape[1], 3),
            dtype=np.uint8)
        title[:] = COL["bg"]
        put_text_bg(
            title,
            f"Domain Shift: Same Detector, Different World"
            f"  |  {AUTHOR}  |  frame {i+1}/{n}",
            6, 20, 0.5, COL["white"], 1)

        canvas = np.vstack([title, combined])
        rgb    = cv2.cvtColor(canvas,
                              cv2.COLOR_BGR2RGB)
        gif_frames.append(rgb)

        if (i+1) % 10 == 0:
            print(f"  {i+1}/{n} frames")

    if gif_frames:
        out = str(RES_DIR / "germany_vs_asu.gif")
        imageio.mimsave(out, gif_frames,
                        duration=0.5, loop=0)
        print(f"  saved {out}")


def make_failure_heatmap():
    print("[4/4] failure_heatmap.mp4")
    all_det  = load_detections()
    out_path = str(RES_DIR / "failure_heatmap.mp4")
    writer   = None
    frame_size = None

    for scenario in SCENARIOS:
        paths   = get_frame_paths(scenario)
        dets    = all_det.get(scenario, [])
        det_map = {d["frame"]: d["detections"]
                   for d in dets}

        print(f"  {scenario}")

        # accumulate heatmap across scenario
        heatmap_acc = None

        for fi, fpath in enumerate(paths):
            frame = cv2.imread(str(fpath))
            if frame is None:
                continue

            H, W = frame.shape[:2]

            if heatmap_acc is None:
                heatmap_acc = np.zeros(
                    (H, W), dtype=np.float32)

            d_list = det_map.get(fpath.name, [])

            # mark failure regions (no detection)
            # and success regions (has detection)
            frame_mask = np.zeros(
                (H, W), dtype=np.float32)

            if len(d_list) == 0:
                # full frame failure
                frame_mask[:] = 1.0
            else:
                # partial failure: mark regions
                # between detections as uncertain
                for d in d_list:
                    x1,y1,x2,y2 = d["bbox"]
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(W-1, x2)
                    y2 = min(H-1, y2)
                    frame_mask[y1:y2, x1:x2] -= 0.3

            heatmap_acc += frame_mask

            # normalize and colorize
            norm = cv2.normalize(
                heatmap_acc, None, 0, 255,
                cv2.NORM_MINMAX).astype(np.uint8)
            colored = cv2.applyColorMap(
                norm, cv2.COLORMAP_JET)

            # blend with original frame
            overlay = cv2.addWeighted(
                frame, 0.55, colored, 0.45, 0)

            # info bar
            bar = np.zeros((36, W, 3), dtype=np.uint8)
            bar[:] = COL["bg"]
            s_lbl = scenario.replace("_", " ")
            n_det = len(d_list)
            put_text_bg(bar,
                f"{s_lbl}  |  frame {fi+1}/{len(paths)}"
                f"  |  {n_det} det  |  "
                f"red=failure  blue=success",
                8, 24, 0.5, COL["white"])
            put_text_bg(bar, AUTHOR,
                W-320, 24, 0.4, COL["grey"])

            panel = np.vstack([bar, overlay])

            if writer is None:
                frame_size = (panel.shape[1],
                              panel.shape[0])
                writer = cv2.VideoWriter(
                    out_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    10, frame_size)

            writer.write(panel)

    if writer:
        writer.release()
        print(f"  saved {out_path}")


if __name__ == "__main__":
    print(f"Day 11: ASU Campus Perception")
    print(f"Vamshikrishna Gadde | MS Robotics ASU")

    make_detection_demo()
    make_failure_gif()
    make_domain_shift_gif()
    make_failure_heatmap()

    print("\nall outputs saved to results/")
    print("  detection_demo.mp4")
    print("  failure_highlights.gif")
    print("  germany_vs_asu.gif")
    print("  failure_heatmap.mp4")