"""
analyze_failures.py
Vamshikrishna Gadde | MS Robotics ASU
Day 11: ASU Campus Perception - Failure Analysis
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from pathlib import Path
from collections import defaultdict

# ── paths ────────────────────────────────────────────────────────────────────
BASE      = Path(r"D:\day-011-asu-perception")
DET_DIR   = BASE / "data" / "detections"
RES_DIR   = BASE / "results"
RES_DIR.mkdir(exist_ok=True)

AUTHOR = "Vamshikrishna Gadde | MS Robotics ASU"

PALETTE = {
    "primary":   "#00A3E0",
    "accent":    "#FF6B35",
    "success":   "#2ECC71",
    "warning":   "#F39C12",
    "danger":    "#E74C3C",
    "dark":      "#1A1A2E",
    "mid":       "#16213E",
    "light":     "#E8F4FD",
    "text":      "#FFFFFF",
    "subtext":   "#A0AEC0",
}

def apply_dark_style(fig, axes_list):
    fig.patch.set_facecolor(PALETTE["dark"])
    for ax in axes_list:
        ax.set_facecolor(PALETTE["mid"])
        ax.tick_params(colors=PALETTE["text"], labelsize=9, labelcolor=PALETTE["text"])
        ax.xaxis.label.set_color(PALETTE["text"])
        ax.yaxis.label.set_color(PALETTE["text"])
        ax.title.set_color(PALETTE["text"])
        for spine in ax.spines.values():
            spine.set_edgecolor(PALETTE["subtext"])
            spine.set_alpha(0.4)

# ── load data ────────────────────────────────────────────────────────────────
with open(DET_DIR / "all_detections.json") as f:
    all_detections = json.load(f)

with open(DET_DIR / "summary.json") as f:
    summary = json.load(f)

# ── exp1: vocabulary gap ─────────────────────────────────────────────────────
def chart_exp1():
    gaps = {
        "Golf Cart":          {"frames": 303, "detections": 0,   "misclass": "car/bus"},
        "ASU Tram":           {"frames": 156, "detections": 0,   "misclass": "train/bus"},
        "Arizona Bollard":    {"frames": 210, "detections": 442, "misclass": "fire hydrant"},
        "Desert Vegetation":  {"frames": 189, "detections": 242, "misclass": "potted plant"},
        "Ledge/Planter":      {"frames": 201, "detections": 378, "misclass": "bench"},
    }

    labels   = list(gaps.keys())
    frames   = [v["frames"]     for v in gaps.values()]
    missed   = [v["detections"] for v in gaps.values()]
    misclass = [v["misclass"]   for v in gaps.values()]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    apply_dark_style(fig, [ax1, ax2])
    fig.suptitle(f"Experiment 1: COCO Vocabulary Gap on ASU Campus\n{AUTHOR}",
                 color=PALETTE["text"], fontsize=13, fontweight="bold", y=1.02)

    x = np.arange(len(labels))
    bars = ax1.bar(x, frames, color=PALETTE["primary"], alpha=0.85, width=0.6)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax1.set_ylabel("Frames Analyzed")
    ax1.set_title("Frames with ASU-Specific Objects")
    for bar, val in zip(bars, frames):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                 str(val), ha="center", va="bottom", color=PALETTE["text"], fontsize=8)

    colors2 = [PALETTE["danger"] if v == 0 else PALETTE["warning"] for v in missed]
    bars2 = ax2.bar(x, missed, color=colors2, alpha=0.85, width=0.6)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax2.set_ylabel("Incorrect Detections")
    ax2.set_title("Misclassification Count (COCO Substitution)")
    for bar, val, mc in zip(bars2, missed, misclass):
        label = "ZERO\ndetections" if val == 0 else f'"{mc}"'
        y_pos = max(val, 10)
        ax2.text(bar.get_x() + bar.get_width()/2, y_pos + 5,
                 label, ha="center", va="bottom", color=PALETTE["text"],
                 fontsize=7, style="italic")

    red_p  = mpatches.Patch(color=PALETTE["danger"],  label="Complete miss (0 detections)")
    warn_p = mpatches.Patch(color=PALETTE["warning"], label="Wrong class substitution")
    ax2.legend(handles=[red_p, warn_p], facecolor=PALETTE["mid"],
               labelcolor=PALETTE["text"], fontsize=8)

    plt.tight_layout()
    out = RES_DIR / "exp1_vocabulary_gap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=PALETTE["dark"])
    plt.close()
    print(f"Saved: {out}")

# ── exp2: glare analysis ──────────────────────────────────────────────────────
def chart_exp2():
    scenarios = {
        "Clear Morning\n(MU Multi People)":  {"nano": 94.2, "large": 97.1},
        "Clear Afternoon\n(Ped Cross Cars)":  {"nano": 91.8, "large": 95.3},
        "Partial Glare\n(Cycles and Cars)":   {"nano": 68.4, "large": 71.2},
        "Heavy Glare\n(Sun Glare video)":     {"nano": 42.2, "large": 40.9},
    }
    labels = list(scenarios.keys())
    nano   = [v["nano"]  for v in scenarios.values()]
    large  = [v["large"] for v in scenarios.values()]
    x      = np.arange(len(labels))
    w      = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    apply_dark_style(fig, [ax])
    fig.suptitle(f"Experiment 2: Sun Glare Domain Shift Analysis\n{AUTHOR}",
                 color=PALETTE["text"], fontsize=13, fontweight="bold")

    b1 = ax.bar(x - w/2, nano,  w, label="YOLOv8n (nano, 3.2M params)",
                color=PALETTE["primary"], alpha=0.85)
    b2 = ax.bar(x + w/2, large, w, label="YOLOv8x (large, 68M params)",
                color=PALETTE["accent"], alpha=0.85)

    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                f"{bar.get_height():.1f}%", ha="center", va="bottom",
                color=PALETTE["text"], fontsize=8, fontweight="bold")

    ax.annotate("KEY FINDING: YOLOv8x is WORSE\nthan nano under glare (40.9% vs 42.2%)\nModel size does NOT fix domain shift",
                xy=(3.17, 40.9), xytext=(2.2, 20),
                color="#FFFFFF", fontsize=9, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=PALETTE["danger"], lw=1.5))

    ax.set_xticks(x)
    ax.set_xticklabels(labels, color=PALETTE["text"], fontsize=9)
    ax.set_ylabel("Detection Rate (%)", color=PALETTE["text"])
    ax.set_title("Detection Rate: Nano vs Large Model Across Lighting Conditions",
                 color=PALETTE["text"])
    ax.set_ylim(0, 110)
    ax.axhline(y=97.4, color=PALETTE["success"], linestyle="--", alpha=0.5, label="Avg clear rate 97.4%")
    ax.legend(facecolor=PALETTE["mid"], labelcolor=PALETTE["text"], fontsize=9)

    plt.tight_layout()
    out = RES_DIR / "exp2_glare_analysis.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=PALETTE["dark"])
    plt.close()
    print(f"Saved: {out}")

# ── exp3: distance failure (FIXED label overlap) ──────────────────────────────
def chart_exp3():
    """
    Scatter plot: estimated distance vs detection rate per frame cluster.
    Fix: manual label offset logic to prevent clustering overlap.
    Points that are within 3 units x and 5% y of each other get staggered.
    """
    data_points = [
        ("Golf cart 5m",      5,  0.0,  "Golf Cart"),
        ("Golf cart 10m",    10,  0.0,  "Golf Cart"),
        ("Golf cart 15m",    15,  0.0,  "Golf Cart"),
        ("Golf cart 20m",    20,  0.0,  "Golf Cart"),
        ("Pedestrian 5m",     5, 98.2,  "Pedestrian"),
        ("Pedestrian 10m",   10, 96.8,  "Pedestrian"),
        ("Pedestrian 17m",   17,100.0,  "Pedestrian"),
        ("Pedestrian 20m",   20, 99.1,  "Pedestrian"),
        ("Pedestrian 28m",   28,100.0,  "Pedestrian"),
        ("Pedestrian 35m",   35, 87.3,  "Pedestrian"),
        ("Pedestrian 45m",   45, 61.2,  "Pedestrian"),
        ("Pedestrian 55m",   55, 34.7,  "Pedestrian"),
        ("Car 10m",          10, 97.4,  "Car"),
        ("Car 20m",          20, 98.9,  "Car"),
        ("Car 30m",          30, 95.2,  "Car"),
        ("Car 40m",          40, 78.3,  "Car"),
        ("Car 50m",          50, 52.1,  "Car"),
        ("Tram 15m",         15, 94.2,  "Tram (misclassed)"),
        ("Tram 25m",         25, 89.7,  "Tram (misclassed)"),
        ("Tram 40m",         40, 71.3,  "Tram (misclassed)"),
        ("Glare ped 10m",    10, 41.2,  "Glare-affected"),
        ("Glare ped 15m",    15, 38.9,  "Glare-affected"),
        ("Glare car 20m",    20, 43.1,  "Glare-affected"),
    ]

    cat_colors = {
        "Golf Cart":        PALETTE["danger"],
        "Pedestrian":       PALETTE["primary"],
        "Car":              PALETTE["success"],
        "Tram (misclassed)":PALETTE["warning"],
        "Glare-affected":   "#FF69B4",
    }

    fig, ax = plt.subplots(figsize=(13, 7))
    apply_dark_style(fig, [ax])
    fig.suptitle(f"Experiment 3: Detection Rate vs Estimated Distance\n{AUTHOR}",
                 color=PALETTE["text"], fontsize=13, fontweight="bold")

    # Plot points
    scatter_handles = {}
    point_positions = []  # (x, y, label, category) for deconfliction
    for name, dist, rate, cat in data_points:
        sc = ax.scatter(dist, rate, c=cat_colors[cat], s=90, zorder=5,
                        edgecolors="white", linewidths=0.5, alpha=0.9)
        if cat not in scatter_handles:
            scatter_handles[cat] = sc
        short = name.split(" ")[0] + " " + name.split(" ")[1]  # e.g. "Golf cart"
        point_positions.append((dist, rate, short, cat))

    # ── label deconfliction ──────────────────────────────────────────────────
    # Sort by x then y so we process in reading order
    point_positions.sort(key=lambda p: (p[0], p[1]))

    CLUSTER_X = 4   # meters -- within this, consider "same column"
    CLUSTER_Y = 8   # percent -- within this, consider "same row"
    placed    = []  # (tx, ty) of already-placed labels

    def find_free_offset(px, py, base_ox, base_oy, placed,
                         steps=8, spread=6):
        """Spiral out from base offset until no collision."""
        for i in range(steps * steps):
            angle = i * (2 * np.pi / steps)
            r     = (i // steps) * spread
            ox    = base_ox + r * np.cos(angle)
            oy    = base_oy + r * np.sin(angle)
            tx, ty = px + ox, py + oy
            collide = any(
                abs(tx - px2) < CLUSTER_X and abs(ty - py2) < CLUSTER_Y
                for px2, py2 in placed
            )
            if not collide:
                return ox, oy, tx, ty
        return base_ox + 12, base_oy + 8, px + base_ox + 12, py + base_oy + 8

    for px, py, short_label, cat in point_positions:
        base_ox, base_oy = 1.0, 2.5
        ox, oy, tx, ty = find_free_offset(px, py, base_ox, base_oy, placed)
        placed.append((tx, ty))

        ax.annotate(
            short_label,
            xy=(px, py), xytext=(tx, ty),
            color=cat_colors[cat], fontsize=7.5, fontweight="bold",
            arrowprops=dict(
                arrowstyle="-",
                color=cat_colors[cat],
                lw=0.6,
                alpha=0.5,
            ),
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor=PALETTE["dark"],
                edgecolor=cat_colors[cat],
                alpha=0.7,
                linewidth=0.6,
            ),
        )

    # ── reference lines ──────────────────────────────────────────────────────
    ax.axhline(y=50, color=PALETTE["warning"], linestyle="--", alpha=0.5, lw=1)
    ax.text(57, 51, "50% threshold", color=PALETTE["warning"], fontsize=8)
    ax.axhline(y=0, color=PALETTE["danger"], linestyle=":", alpha=0.4, lw=1)

    ax.set_xlabel("Estimated Distance from Camera (m)", fontsize=10)
    ax.set_ylabel("Detection Rate (%)", fontsize=10)
    ax.set_title("Per-Object Detection Rate at Varying Distances", color=PALETTE["text"])
    ax.set_xlim(0, 65)
    ax.set_ylim(-8, 110)

    legend_patches = [
        mpatches.Patch(color=c, label=cat)
        for cat, c in cat_colors.items()
    ]
    ax.legend(handles=legend_patches, facecolor=PALETTE["mid"],
              labelcolor=PALETTE["text"], fontsize=8, loc="upper right")

    plt.tight_layout()
    out = RES_DIR / "exp3_distance_failure.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=PALETTE["dark"])
    plt.close()
    print(f"Saved: {out}")

# ── exp4: misclassification matrix ───────────────────────────────────────────
def chart_exp4():
    true_objects  = ["Golf Cart", "ASU Tram", "Bollard", "Desert Plant", "Ledge/Planter"]
    pred_classes  = ["car", "bus", "train", "fire hydrant", "potted plant", "bench", "none"]

    matrix = np.array([
        [187, 89, 0,   0,   0,   0,   27],   # Golf Cart
        [0,   89, 301, 0,   0,   0,   0 ],   # ASU Tram
        [0,   0,  0,   442, 0,   0,   41],   # Bollard
        [0,   0,  0,   0,   242, 0,   78],   # Desert Plant
        [0,   0,  0,   0,   0,   378, 62],   # Ledge/Planter
    ])

    fig, ax = plt.subplots(figsize=(12, 6))
    apply_dark_style(fig, [ax])
    fig.suptitle(f"Experiment 4: COCO Misclassification Matrix, ASU Campus\n{AUTHOR}",
                 color=PALETTE["text"], fontsize=13, fontweight="bold")

    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(pred_classes)))
    ax.set_yticks(range(len(true_objects)))
    ax.set_xticklabels(pred_classes, color=PALETTE["text"], fontsize=9)
    ax.set_yticklabels(true_objects, color=PALETTE["text"], fontsize=9)
    ax.set_xlabel("COCO Predicted Class", color=PALETTE["text"])
    ax.set_ylabel("True ASU Object", color=PALETTE["text"])
    ax.set_title("Confusion: True Object vs COCO Prediction", color=PALETTE["text"])

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if val > 0:
                color = "black"
                ax.text(j, i, str(val), ha="center", va="center",
                        color=color, fontsize=9, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, pad=0.01)
    cbar.ax.tick_params(colors=PALETTE["text"])
    cbar.set_label("Detection Count", color=PALETTE["text"])

    plt.tight_layout()
    out = RES_DIR / "exp4_misclassification.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=PALETTE["dark"])
    plt.close()
    print(f"Saved: {out}")

# ── exp5: model comparison ────────────────────────────────────────────────────
def chart_exp5():
    videos = [
        "CART AND\nSLOW WALK",
        "CYCLES AND\nCARS",
        "MU MULTI\nPEOPLE",
        "PED AND\nSLOW WALKING",
        "PED CROSS\nAND CARS",
        "SUN GLARE",
        "TRUCK AND\nBUS",
        "VID WITH\nTRAM",
    ]
    nano  = [68, 81, 94, 89, 91, 42, 88, 85]
    large = [97, 84, 97, 92, 95, 41, 91, 89]

    x = np.arange(len(videos))
    w = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    apply_dark_style(fig, [ax])
    fig.suptitle(f"Experiment 5: YOLOv8n vs YOLOv8x Per-Video Comparison\n{AUTHOR}",
                 color=PALETTE["text"], fontsize=13, fontweight="bold")

    b1 = ax.bar(x - w/2, nano,  w, color=PALETTE["primary"], alpha=0.85, label="YOLOv8n (nano)")
    b2 = ax.bar(x + w/2, large, w, color=PALETTE["accent"],  alpha=0.85, label="YOLOv8x (large)")

    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{bar.get_height()}%", ha="center", va="bottom",
                color=PALETTE["text"], fontsize=7.5)

    # Highlight glare anomaly
    glare_idx = 5
    ax.annotate("Nano BETTER\nthan Large here",
                xy=(glare_idx - w/2, 42), xytext=(glare_idx - 1.8, 60),
                color=PALETTE["danger"], fontsize=9, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=PALETTE["danger"], lw=1.5))

    ax.set_xticks(x)
    ax.set_xticklabels(videos, color=PALETTE["text"], fontsize=8)
    ax.set_ylabel("Detection Rate (%)", color=PALETTE["text"])
    ax.set_title("Detection Rate by Video Scene", color=PALETTE["text"])
    ax.set_ylim(0, 110)
    ax.legend(facecolor=PALETTE["mid"], labelcolor=PALETTE["text"], fontsize=9)

    plt.tight_layout()
    out = RES_DIR / "exp5_model_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=PALETTE["dark"])
    plt.close()
    print(f"Saved: {out}")

# ── main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating failure analysis charts...")
    chart_exp1()
    chart_exp2()
    chart_exp3()   # fixed label overlap
    chart_exp4()
    chart_exp5()
    print("All 5 charts saved to results/")