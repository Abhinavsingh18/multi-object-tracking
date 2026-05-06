# 🏃 Multi-Object Detection & Tracking — Sports / Event Footage

> **Original Public Video Used:** [10 Minutes of Dusan Bulut Highlights | FIBA 3x3 Basketball](https://www.youtube.com/watch?v=fpnkLpCBBnU)

A complete computer vision pipeline that detects and persistently tracks every person in a sports video using **YOLOv8** (detection) and **ByteTrack** (tracking). Includes a full **Gradio web app** so anyone can upload their own video and get annotated results instantly.

---

## 📦 Project Structure

```
multi-object-tracking/
├── app.py                  # ← Gradio Web App (run this for the UI)
├── main.py                 # ← CLI Python Script (run this from terminal)
├── requirements.txt        # ← All dependencies
├── README.md
```

---

## 🚀 Quick Start (3 Steps)

### Step 1 — Clone & Setup Environment

```bash
git clone https://github.com/Abhinavsingh18/multi-object-tracking.git
cd multi-object-tracking

python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

pip install -r requirements.txt
```

> ⚠️ YOLOv8 model weights (`yolov8m.pt`) are downloaded **automatically** the first time you run the script. No manual download needed.

---

### Step 2 — Add Your Video

Place any sports/event video in the project folder and name it `input_video.mp4`.

**Or download the test video automatically using yt-dlp:**

```bash
yt-dlp "https://www.youtube.com/watch?v=fpnkLpCBBnU" -f "b[ext=mp4]" -o "input_video.mp4"
```

---

### Step 3 — Run (choose one)

#### Option A: 🌐 Web App (Recommended)
```bash
python app.py
```
Then open your browser at **http://localhost:7860** — upload any video and click **Run Tracking**.

#### Option B: 💻 CLI Script
```bash
python main.py
```
Reads `input_video.mp4`, writes `output_video.mp4`.

---

## ✅ Features

| Feature | Status |
|---|---|
| Person detection (YOLOv8m) | ✅ |
| Persistent ID assignment (ByteTrack) | ✅ |
| Bounding box + ID labels | ✅ |
| Movement trajectory traces | ✅ |
| Thermal heatmap overlay | ✅ |
| Live object count on video | ✅ |
| Gradio web app UI | ✅ |

---

## ⚙️ Assumptions

1. **Target subjects are people** — The pipeline filters for COCO class `0` (person). Balls and other objects are intentionally ignored.
2. **Camera is relatively stable** — ByteTrack handles moderate panning, but extreme camera shakes may reduce ID stability.
3. **Standard video format** — Expects standard MP4 at 25–60 FPS.

---

## ⚠️ Known Limitations

1. **Severe overlaps** — When two players are tightly overlapped for many frames, ByteTrack may swap IDs when they separate.
2. **Extreme motion blur** — Very fast camera panning can cause all detections to be missed for 1–2 frames. ByteTrack's buffer (30 frames) recovers most IDs automatically.
3. **Processing speed** — CPU-only processing runs at ~3–5× video duration. A CUDA GPU will run in near real-time.

---

## 🔬 Model & Tracker Choices

| Choice | Reason |
|---|---|
| **YOLOv8m** | Best speed/accuracy balance. Pre-trained on COCO — detects people out of the box without fine-tuning |
| **ByteTrack** | Tracks low-confidence detections (unlike DeepSORT) — essential for occluded/blurred players |
| **supervision** | Clean annotation API for boxes, labels, traces, and heatmaps |
