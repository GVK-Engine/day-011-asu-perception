# Day 11 — ASU Campus Perception: When KITTI Meets Arizona

> MS Robotics and Autonomous Systems, Arizona State University, Dec 2026

---

## The Real Question

Every AV perception paper trains on KITTI, nuScenes, or Waymo Open Dataset and reports mAP numbers on the same distribution. Nobody asks what happens when you take that trained model and drive it somewhere it has never seen.

I filmed 8 scenarios on ASU Tempe campus with my phone, ran a KITTI-trained YOLOv8 detector on the footage, and measured exactly what fails, why it fails, and whether throwing a bigger model at the problem fixes it.

The short answer: it does not.

---

## Live Demo

[![Detection Demo](https://drive.google.com/thumbnail?id=1Y1hKLwYCygZ6e-bYi_DzgnDSyB1guhob&sz=w1280)](https://drive.google.com/file/d/1Y1hKLwYCygZ6e-bYi_DzgnDSyB1guhob/view)

*YOLOv8x running on all 8 ASU campus scenarios. 2,565 frames. Real Arizona footage. Click to view.*

---

## Germany vs Arizona: Same Detector, Different World

[![Domain Shift](https://drive.google.com/thumbnail?id=1MZrBCFBThu1TFwLlpQ3WAkCEkNYJ_fHB&sz=w1280)](https://drive.google.com/file/d/1MZrBCFBThu1TFwLlpQ3WAkCEkNYJ_fHB/view)

*Left: KITTI Germany, overcast, HDL-64E LiDAR car. Right: ASU Arizona, direct sun, iPhone. Same YOLOv8x model. Click to view.*

---

## Failure Highlights

[![Failure Highlights](https://drive.google.com/thumbnail?id=1xqhuXlebJZVao5eUxHyx4cOHvy7SkEnU&sz=w1280)](https://drive.google.com/file/d/1xqhuXlebJZVao5eUxHyx4cOHvy7SkEnU/view)

*Golf cart: 0 detections. Sun glare: classified as sports ball. ASU tram: labeled as train. Arizona bollards: called fire hydrants. These are not edge cases. They are systematic. Click to view.*

---

## Spatial Failure Heatmap

[![Failure Heatmap](https://drive.google.com/thumbnail?id=16MoiJ5gXWdKKqA8d4zYHC7u8CT_ioWtI&sz=w1280)](https://drive.google.com/file/d/16MoiJ5gXWdKKqA8d4zYHC7u8CT_ioWtI/view)

*Red regions accumulate where the detector fails across frames. Blue regions are reliable. Click to view.*

---

## Experiment 1: The Vocabulary Gap

![Vocabulary Gap](https://drive.google.com/thumbnail?id=1Qf5zxdGNAYzX0GowhzaR63DPOFS0pQqx&sz=w1280)

COCO has 80 object classes. None of them are golf cart. None of them are ASU tram. None of them are Arizona bollard.

When the model encounters these objects it does not say unknown. It finds the nearest class it knows and outputs that instead. A golf cart becomes a car. An ASU tram becomes a train. A bollard becomes a fire hydrant.

This is not a model failure. It is a vocabulary failure. No amount of training on COCO data will teach the model what a golf cart is because golf cart is not in the vocabulary.

| Object | Frames | Correct Detections | Substituted As |
|--------|--------|-------------------|----------------|
| Golf Cart | 303 | 0 | car, bus |
| ASU Tram | 336 | 0 | train, bus |
| Arizona Bollard | all | 0 | fire hydrant (442 times) |
| Desert Vegetation | all | 0 | potted plant (242 times) |
| ASU Ledge/Planter | all | 0 | bench (378 times) |

---

## Experiment 2: Sun Glare Is Not a Model Size Problem

![Glare Analysis](https://drive.google.com/thumbnail?id=14fefm-M2h_YPaGsH39D2GDb1Sf8wz86f&sz=w1280)

This is the finding I did not expect.

In clear conditions the detector averaged 97.4% frame detection rate across 7 scenarios. In the sun glare scenario it dropped to 40.9%. A 56.5 percentage point collapse.

I then ran the same footage through YOLOv8x, the 68 million parameter version, 20 times larger than nano. It scored 40.9% as well. The nano model actually scored slightly higher at 42.2%.

The larger model performed worse under glare. More parameters did not help. The failure is in the training distribution, not the model capacity. KITTI was filmed in Germany under overcast European skies. Arizona direct sun at 10am is a different optical world. No scaling fixes a distribution mismatch.

```
Clear avg detection rate:   97.4%
Sun glare detection rate:   40.9%
Drop:                       56.5%

YOLOv8n on glare:           42.2%
YOLOv8x on glare:           40.9%   worse
```

---

## Experiment 3: Detection Rate vs Distance

![Distance Failure](https://drive.google.com/thumbnail?id=15VRS_e2iGvdUdvRfz729jAJjGRMjJDwJ&sz=w1280)

Across all 8 scenarios the detection rate holds strong up to 40m. The sun glare scenario is the clear outlier, sitting at 40.9% regardless of distance because the failure is optical not geometric.

---

## Experiment 4: Semantic Misclassification Map

![Misclassification Map](https://drive.google.com/thumbnail?id=1KPtlBV335msssiXTEAoZbe3XFmkh5I-j&sz=w1280)

The model is not failing to detect. It is detecting the wrong thing. Every misclassification has a clear cause.

Fire hydrant maps to bollard because both are short cylindrical objects at road level. Potted plant maps to desert vegetation because the shape profile is similar but the species are completely different. Train maps to ASU tram because tram is not in the vocabulary so the model picks the closest rail vehicle it knows.

---

## Experiment 5: Model Size Does Not Fix Domain Shift

![Model Comparison](https://drive.google.com/thumbnail?id=1hPXiHBkOzsl3OD8ImDAp5EGd7kWv65Iy&sz=w1280)

I tested both YOLOv8n (3.2M parameters) and YOLOv8x (68M parameters) across all 8 scenarios.

The large model improved significantly on scenarios where the nano model struggled with small or distant objects. Golf cart detection went from 68% to 97% because the larger model is better at recognizing partial views of small vehicles.

But on sun glare, the large model performed worse. This confirms what the domain shift literature says: you cannot scale your way out of a distribution mismatch. The model needs Arizona data, not more parameters.

| Scenario | YOLOv8n | YOLOv8x | Change |
|----------|---------|---------|--------|
| Golf Cart | 68.0% | 97.0% | +29.0% |
| Cycles and Cars | 100% | 100% | same |
| MU Multi People | 100% | 100% | same |
| Ped Slow Walking | 71.9% | 84.6% | +12.7% |
| Ped Cross Cars | 98.4% | 100% | +1.6% |
| Sun Glare | 42.2% | 40.9% | -1.3% |
| Truck and Bus | 99.7% | 100% | same |
| Vid with Tram | 99.1% | 100% | same |

---

## What I Filmed

8 videos filmed on ASU Tempe campus, June 8 2026, 9:52am to 10:18am.

```
CART AND SLOW WALK     61s  303 frames  golf cart scenario
CYCLES AND CARS        63s  316 frames  cyclists and traffic
MU MULTI PEOPLE        60s  302 frames  dense pedestrian crowd
PED AND SLOW WALKING   66s  331 frames  pedestrians at range
PED CROSS AND CARS     64s  319 frames  crossing behavior
SUN GLARE              61s  303 frames  direct Arizona sun
TRUCK AND BUS          71s  355 frames  large vehicles
VID WITH TRAM          67s  336 frames  ASU Valley Metro tram

Total: 8 videos, 856 MB, 2,565 frames at 5 FPS
```

---

## Detection Pipeline

```
Frame extraction:   5 FPS from 30 FPS footage
Model:              YOLOv8x (primary), YOLOv8n (comparison)
Confidence:         0.25 threshold
Distance est:       similar triangles
                    distance = (real_height x focal_length) / pixel_height
                    iPhone focal length 1050px estimated
Classes tracked:    80 COCO classes
Frames processed:   2,565 across 8 scenarios
```

---

## How This Connects to the Series

```
Day 9:  Domain shift KITTI to nuScenes, 58.4% detection drop
        Root cause was sensor difference, not scene difference

Day 11: Domain shift KITTI to ASU campus
        Root causes measured one by one:
        vocabulary gap, lighting distribution,
        infrastructure appearance, camera perspective
        The sensor was fine. Everything else changed.
```

---

## Run It Yourself

```bash
git clone https://github.com/GVK-Engine/day-011-asu-perception
cd day-011-asu-perception
pip install -r requirements.txt
```

Update paths in each script to your local KITTI and footage directories.

```bash
py -3.11 extract_frames.py
py -3.11 detect_asu.py
py -3.11 analyze_failures.py
py -3.11 visualize.py
```

---

## Project Structure

```
day-011-asu-perception/
├── extract_frames.py       5 FPS frame extraction from MOV files
├── detect_asu.py           YOLOv8 detection with distance estimation
├── analyze_failures.py     5 experiments, 5 charts
├── visualize.py            4 video and GIF outputs
├── requirements.txt
├── data/
│   ├── raw_footage/        8 MOV files from ASU campus
│   ├── frames/             2,565 extracted JPEGs
│   └── detections/         all_detections.json, summary.json
└── results/
    ├── detection_demo.mp4
    ├── failure_highlights.gif
    ├── germany_vs_asu.gif
    ├── failure_heatmap.mp4
    ├── exp1_vocabulary_gap.png
    ├── exp2_glare_analysis.png
    ├── exp3_distance_failure.png
    ├── exp4_misclassification.png
    └── exp5_model_comparison.png
```

---

## Stack

`Python 3.11` `PyTorch 2.6` `YOLOv8` `Ultralytics` `OpenCV` `NumPy` `Matplotlib` `imageio` `COCO` `KITTI`

---

## Series 1 Progress

| # | Project | Key Finding | Status |
|---|---------|-------------|--------|
| P1.1 | LiDAR Obstacle Detection | 0.4m voxel creates ghost detections | ✅ |
| P1.2 | Stereo Camera Depth Safety | Camera unsafe beyond 10m | ✅ |
| P1.3 | PointPillars 3D Detector | 98.9% loss reduction from scratch | ✅ |
| P1.4 | Multi-Camera BEV Perception | 178 objects from 6 cameras | ✅ |
| P1.5 | Multi-Object Tracking SORT | Detector is bottleneck not tracker | ✅ |
| P1.6 | Semantic Segmentation ROS2 | 52.6 FPS, warmup cost measured | ✅ |
| P1.7 | Adverse Weather Analysis | Fog unsafe below 75m visibility | ✅ |
| P1.8 | LiDAR-Camera Depth Completion | 44x MAE improvement 0-10m | ✅ |
| P1.9 | Domain Shift Analysis | 58.4% drop, sensor not scene | ✅ |
| P1.10 | Neural Occupancy Network | Unsafe planning boundary at 40m | ✅ |
| P1.11 | ASU Campus Perception | Model scaling cannot fix domain shift | ✅ |
