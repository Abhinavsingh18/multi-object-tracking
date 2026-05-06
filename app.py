import gradio as gr
import cv2
import numpy as np
import tempfile
import os
from ultralytics import YOLO
import supervision as sv

# ─── Load model once at startup ──────────────────────────────────────────────
print("Loading YOLOv8 model...")
model = YOLO("yolov8m.pt")
print("Model loaded!")

# ─── Core tracking function ───────────────────────────────────────────────────
def run_tracking(video_path: str, conf_threshold: float, show_heatmap: bool, show_trace: bool):
    """
    Takes an input video, runs YOLOv8 + ByteTrack detection/tracking,
    and returns the annotated output video path.
    """
    if video_path is None:
        return None, "⚠️ Please upload a video first."

    # Re-initialise tracker for each new run so IDs start fresh
    tracker = sv.ByteTrack()

    # Annotators
    box_annotator  = sv.BoundingBoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(
        text_thickness=2, text_scale=0.5, text_padding=8
    )
    trace_annotator   = sv.TraceAnnotator(thickness=2, trace_length=50)
    heatmap_annotator = sv.HeatMapAnnotator()

    cap = cv2.VideoCapture(video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Write to a temp file
    out_fd, out_path = tempfile.mkstemp(suffix=".mp4")
    os.close(out_fd)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    processed    = 0
    unique_ids   = set()   # track every ID ever seen across the whole video

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False, conf=conf_threshold, classes=[0])[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = tracker.update_with_detections(detections)

        if detections.tracker_id is not None and len(detections.tracker_id) > 0:
            unique_ids.update(detections.tracker_id.tolist())

        confs  = detections.confidence  if detections.confidence  is not None else []
        tids   = detections.tracker_id  if detections.tracker_id  is not None else []
        labels = [f"#{tid}  {conf:.2f}" for conf, tid in zip(confs, tids)]

        annotated = frame.copy()

        if show_heatmap:
            annotated = heatmap_annotator.annotate(scene=annotated, detections=detections)

        if show_trace:
            annotated = trace_annotator.annotate(scene=annotated, detections=detections)

        annotated = box_annotator.annotate(scene=annotated, detections=detections)
        annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

        # Live counter overlay
        cv2.putText(
            annotated,
            f"Live Count: {len(detections)}  |  Unique IDs so far: {len(unique_ids)}",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA,
        )

        writer.write(annotated)
        processed += 1

    cap.release()
    writer.release()

    progress_pct = round(processed / total * 100) if total > 0 else 100
    summary = (
        f"✅ Done! Processed {processed}/{total} frames ({progress_pct}%)\n"
        f"👥 Total unique IDs assigned: {len(unique_ids)}\n"
        f"💡 Tip: If IDs seem too high, increase the confidence threshold slider."
    )
    return out_path, summary


# ─── Gradio UI ────────────────────────────────────────────────────────────────
DESCRIPTION = """
# 🏃 Multi-Object Tracking — Sports & Event Footage
**Powered by YOLOv8 + ByteTrack**

### How to use:
1. **Upload** a sports/event video (MP4 recommended, keep it under 60 seconds for speed).
2. Adjust **confidence threshold** (lower → detects more people, higher → fewer false positives).
3. Toggle **heatmap** and **trajectory trace** overlays.
4. Click **Run Tracking** and wait for the annotated video.

### What this does:
- Detects every person in every frame using **YOLOv8m**.
- Assigns a **persistent unique ID** to each person using **ByteTrack**.
- Handles occlusion, fast motion, and similar-looking subjects.
- Draws bounding boxes, ID labels, movement traces, and a thermal heatmap.
"""

TIPS = """
### ⚠️ Tips & Limitations
- 🎥 Best results with **720p/1080p sports footage** (soccer, cricket, basketball).
- ⏱️ Processing time ≈ **2–5× video duration** on CPU. GPU will be much faster.
- 🔁 **ID swaps** can happen when two players overlap for extended frames — this is a known ByteTrack limitation.
- 📌 Only **people (class 0)** are tracked. Balls and other objects are ignored.
- 🌐 The original test video: [YouTube Soccer Highlights](https://www.youtube.com/watch?v=KtZyv-KGFa8)
"""

with gr.Blocks(title="Multi-Object Tracker") as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(label="📹 Upload Input Video", sources=["upload"])
            conf_slider = gr.Slider(
                minimum=0.1, maximum=0.9, value=0.5, step=0.05,
                label="Detection Confidence Threshold",
                info="⬆️ Higher = fewer false detections & more stable IDs. Recommended: 0.4–0.6 for sports."
            )
            with gr.Row():
                heatmap_toggle = gr.Checkbox(value=True,  label="🔥 Show Heatmap")
                trace_toggle   = gr.Checkbox(value=True,  label="🛤️  Show Trajectories")
            run_btn = gr.Button("🚀 Run Tracking", variant="primary", size="lg")

        with gr.Column(scale=1):
            video_output = gr.Video(label="🎬 Annotated Output Video")
            status_box   = gr.Textbox(label="📊 Processing Summary", lines=3, interactive=False)

    gr.Markdown(TIPS)

    run_btn.click(
        fn=run_tracking,
        inputs=[video_input, conf_slider, heatmap_toggle, trace_toggle],
        outputs=[video_output, status_box],
    )

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())
