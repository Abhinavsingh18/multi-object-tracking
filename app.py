import gradio as gr
import cv2
import os
from ultralytics import YOLO
import supervision as sv

print("Loading YOLOv8 model...")
model = YOLO("yolov8m.pt")
print("Model loaded!")


def run_tracking(video_path: str, conf_threshold: float, show_heatmap: bool, show_trace: bool):
    if video_path is None:
        return None, "Please upload a video first."

    tracker           = sv.ByteTrack()
    box_annotator     = sv.BoundingBoxAnnotator(thickness=2)
    lbl_annotator     = sv.LabelAnnotator(text_thickness=2, text_scale=0.6, text_padding=8)
    trace_annotator   = sv.TraceAnnotator(thickness=2, trace_length=50)
    heatmap_annotator = sv.HeatMapAnnotator()

    cap    = cv2.VideoCapture(video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tracked_output.mp4")
    writer   = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    all_ids   = set()  # not used for display — ByteTrack IDs are shown directly on video
    processed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results    = model(frame, verbose=False, conf=conf_threshold, classes=[0])[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = tracker.update_with_detections(detections)

        # Track unique IDs (for internal use only — not shown in summary)
        if detections.tracker_id is not None:
            all_ids.update(detections.tracker_id.tolist())

        # Labels — just show the ID ByteTrack assigned, no remapping at all
        tids  = detections.tracker_id if detections.tracker_id is not None else []
        confs = detections.confidence  if detections.confidence  is not None else []
        labels = [f"#{int(tid)}  {conf:.2f}" for tid, conf in zip(tids, confs)]

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

    progress_pct = round(processed / total * 100) if total > 0 else 100
    summary = f"Done — {processed}/{total} frames processed ({progress_pct}%)"
    return out_path, summary


DESCRIPTION = """
# 🏃 Multi-Object Tracking — Sports & Event Footage
**Powered by YOLOv8 + ByteTrack**

### How to use:
1. **Upload** a sports/event video (MP4, keep under 60 seconds for speed).
2. Set **confidence threshold** — higher = fewer false detections, more stable IDs.
3. Toggle **heatmap** and **trajectory** overlays.
4. Click **Run Tracking**.

### What happens:
- Every detected person gets a **unique persistent ID** automatically.
- ByteTrack assigns IDs — however many subjects are in the video, that many IDs appear.
- No limits, no fixed numbers.
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
            status_box   = gr.Textbox(label="📊 Result Summary", lines=2, interactive=False)
    run_btn.click(
        fn=run_tracking,
        inputs=[video_input, conf_slider, heatmap_toggle, trace_toggle],
        outputs=[video_output, status_box],
    )

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 7860))
    demo.launch(share=False, server_name="0.0.0.0", server_port=port, theme=gr.themes.Soft())

