# extract_frames.py
# extracts frames from ASU campus footage
# at 5 FPS for efficient processing
#
# why 5 FPS not 30 FPS:
#   at 30 FPS consecutive frames are nearly identical
#   5 FPS gives enough variety for analysis
#   reduces 16,000 frames to ~2,700
#   keeps processing time reasonable
#
# output structure:
#   data/frames/VID_WITH_TRAM/frame_0001.jpg
#   data/frames/PED_CROSS_AND_CARS/frame_0001.jpg
#   etc.
#
# Nani — MS Robotics ASU

import cv2
import os

RAW_DIR    = r"D:\day-011-asu-perception\data\raw_footage"
FRAMES_DIR = r"D:\day-011-asu-perception\data\frames"
SAMPLE_FPS = 5     # extract 5 frames per second
JPEG_Q     = 95    # JPEG quality (0-100)


def video_info(path):
    cap   = cv2.VideoCapture(path)
    fps   = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, total, w, h


def extract(video_path, out_dir, sample_fps=SAMPLE_FPS):
    """
    Extract frames from one video at sample_fps rate.

    How frame sampling works:
      video is 30 FPS
      we want 5 FPS
      step = 30 / 5 = 6
      extract frame 0, 6, 12, 18, 24 ...
      every 6th frame = 5 times per second
    """
    os.makedirs(out_dir, exist_ok=True)

    cap       = cv2.VideoCapture(video_path)
    src_fps   = cap.get(cv2.CAP_PROP_FPS)
    total     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step      = max(1, round(src_fps / sample_fps))
    n_extract = total // step

    frame_idx  = 0
    saved      = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            fname = os.path.join(
                out_dir, f"frame_{saved:04d}.jpg")
            cv2.imwrite(fname, frame,
                        [cv2.IMWRITE_JPEG_QUALITY,
                         JPEG_Q])
            saved += 1

        frame_idx += 1

    cap.release()
    return saved


if __name__ == "__main__":

    # find all video files
    videos = sorted([
        f for f in os.listdir(RAW_DIR)
        if f.lower().endswith(('.mov', '.mp4'))
    ])

    print(f"found {len(videos)} videos")
    print(f"sample rate: {SAMPLE_FPS} FPS")

    total_frames = 0

    for vname in videos:
        vpath    = os.path.join(RAW_DIR, vname)
        fps, total, w, h = video_info(vpath)
        duration = total / fps

        # folder name: remove extension, replace spaces
        folder   = vname.replace('.MOV', '')\
                        .replace('.mov', '')\
                        .replace('.MP4', '')\
                        .replace('.mp4', '')\
                        .replace(' ', '_')
        out_dir  = os.path.join(FRAMES_DIR, folder)

        n_expect = int(duration * SAMPLE_FPS)
        print(f"\n  {vname}")
        print(f"    {duration:.0f}s  {total} frames  "
              f"{w}x{h}  → ~{n_expect} extracted")

        saved = extract(vpath, out_dir)
        total_frames += saved
        print(f"    saved {saved} frames to {out_dir}")

    print(f"\ntotal frames extracted: {total_frames}")
    print(f"saved to: {FRAMES_DIR}")