# 3D Speed Estimation of Right Wrist Speed Using MediaPipe Pose

A **real-time pose tracking + 3D speed estimation** system that:

- Uses **MediaPipe Pose** to detect human body landmarks in a video
- Tracks the **right wrist** in **3D world coordinates** using a **camera intrinsic matrix** and estimated **metric depth value** from **monocular depth estimation - DepthPro**
- Computes **instantaneous speed in meters per second (m/s)**
- Overlays speed on video with clean **white background text**
- Saves **annotated output video** in `.mp4` format

---

## Features

| Feature | Description |
|-------|-----------|
| **MediaPipe Pose** | High-accuracy 2D landmark detection |
| **3D Reconstruction** | Converts pixel + depth → real-world 3D using `K` matrix |
| **Speed Calculation** | `Δdistance / Δtime` in **m/s** |
| **Clean Text Overlay** | White background box + black text |
| **Video Output** | Saves processed video in same directory |
| **Configurable Depth** | Easy to replace placeholder with depth sensor data |

---

## Requirements

```bash
pip install opencv-python mediapipe numpy
```
