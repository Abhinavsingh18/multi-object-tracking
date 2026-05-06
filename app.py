import gradio as gr
import cv2
import os
from ultralytics import YOLO
import supervision as sv

# Load model once at startup
print("Loading YOLOv8 model...")
model = YOLO("yolov8m.pt")
print("Model loaded!")


def run_tracking(video_path: str, conf_threshold: float, show_heatmap: bool, show_trace: bool):
    """
    Detect every subject in the video and assign each a unique sequential ID (#1, #2, #3 ...).
    No fixed limits — the number of IDs depends entirely on what is detected.
    """
    if video_path is None:
        return None, "Please upload a video first."

    # Fresh tracker per run
    tracker = sv.ByteTrack()

    # Annotators
    box_annotator     = sv.BoundingBoxAnnotator(thickness=2)
    lbl_annotator     = sv.LabelAnnotator(text_thickness=2, text_scale=0.6, text_padding=8)
    trace_annotator   = sv.TraceAnnotator(thickness=2, trace_length=50)
    heatmap_annotator = sv.HeatMapAnnotator()

    cap    = cv2.VideoCapture(video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Fixed output path — always overwritten so Gradio never serves stale video
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tracked_output.mp4")
    writer   = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    # Remap ByteTrack's internal (large) IDs → clean #1, #2, #3 ...
    id_map    = {}
    next_id   = 1
    processed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results    = model(frame, verbose=False, conf=conf_threshold, classes=[0])[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = tracker.update_with_detections(detections)

        # Build label for each detected subject
        labels = []
        tids  = detections.tracker_id if detections.tracker_id is not None else []
        confs = detections.confidence  if detections.confidence  is not None else []
        for raw_id, conf in zip(tids, confs):
            raw_id = int(raw_id)
            if raw_id not in id_map:   # first time we see this subject → give next clean ID
                id_map[raw_id] = next_id
                next_id += 1
            labels.append(f"#{id_map[raw_id]}  {conf:.2f}")

        # Draw annotations
        annotated = frame.copy()
        if show_heatmap:
            annotated = heatmap_annotator.annotate(scene=annotated, detections=detections)
        if show_trace:
            annotated = trace_annotator.annotate(scene=annotated, detections=detections)
        annotated = box_annotator.annotate(scene=annotated, detections=detections)
        annotated = lbl_annotator.annotate(scene=annotated, detections=detections, labels=labels)

        writer.write(annotated)
        processed += 1

    cap.release()
    writer.release()

    total_unique = next_id - 1
    progress_pct = round(processed / total * 100) if total > 0 else 100
    summary = (
        f"Done — {processed}/{total} frames processed ({progress_pct}%)\n"
        f"Unique subjects tracked: {total_unique}\n"
        f"IDs assigned on video: #1 to #{total_unique}"
    )
    return out_path, summary


# ── Gradio UI ─────────────────────────────────────────────────────────────────
DESCRIPTION = """
# 🏃 Multi-Object Tracking — Sports & Event Footage
**Powered by YOLOv8 + ByteTrack**

### How to use:
1. **Upload** a sports/event video (MP4, keep under 60 seconds for speed).
2. Adjust the **confidence threshold** — higher = fewer false detections, more stable IDs.
3. Toggle **heatmap** and **trajectory** overlays.
4. Click **Run Tracking** and wait.

### What happens:
- Every person detected gets a **unique sequential ID** (#1, #2, #3 …).
- IDs are persistent — same person keeps the same ID across frames.
- Number of IDs depends entirely on the video content. No limits.
"""

TIPS = """
### ⚠️ Tips & Limitations
- 🎥 Best with **720p/1080p sports footage** (soccer, cricket, basketball).
- ⏱️ Processing ≈ **2–5× video duration** on CPU.
- 🔁 **ID swaps** can happen when two subjects overlap for many frames — known ByteTrack limitation.
- 📌 Only **people** are tracked. Balls, vehicles ignored (configurable in code).
- 🌐 Test video used: [YouTube Soccer Highlights](https://www.youtube.com/watch?v=KtZyv-KGFa8)
"""

with gr.Blocks(title="Multi-Object Tracker") as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(label="📹 Upload Input Video", sources=["upload"])
            conf_slider = gr.Slider(
                minimum=0.1, maximum=0.9, value=0.5, step=0.05,
                label="Detection Confidence Threshold",
                info="Higher = cleaner detections & more stable IDs. Try 0.4–0.6 for sports."
            )
            with gr.Row():
                heatmap_toggle = gr.Checkbox(value=True, label="🔥 Show Heatmap")
                trace_toggle   = gr.Checkbox(value=True, label="🛤️  Show Trajectories")
            run_btn = gr.Button("🚀 Run Tracking", variant="primary", size="lg")

        with gr.Column(scale=1):
            video_output = gr.Video(label="🎬 Annotated Output Video")
            status_box   = gr.Textbox(label="📊 Result Summary", lines=3, interactive=False)

    gr.Markdown(TIPS)

    run_btn.click(
        fn=run_tracking,
        inputs=[video_input, conf_slider, heatmap_toggle, trace_toggle],
        outputs=[video_output, status_box],
    )

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())
