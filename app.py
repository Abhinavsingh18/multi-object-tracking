import streamlit as st
import cv2
import os
import tempfile
import subprocess
from ultralytics import YOLO
import supervision as sv

st.set_page_config(page_title="Multi-Object Tracker", page_icon="🏃", layout="wide")

@st.cache_resource
def load_model():
    print("Loading YOLOv8 model...")
    model = YOLO("yolov8m.pt")
    print("Model loaded!")
    return model

model = load_model()

def run_tracking(video_path: str, conf_threshold: float, show_heatmap: bool, show_trace: bool, progress_bar, status_text):
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

    processed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results    = model(frame, verbose=False, conf=conf_threshold, classes=[0])[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = tracker.update_with_detections(detections)

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
        
        # Update progress
        if total > 0:
            progress_pct = int((processed / total) * 100)
            progress_bar.progress(progress_pct)
            status_text.text(f"Processing frame {processed}/{total} ({progress_pct}%)")

    cap.release()
    writer.release()
    
    # Convert video to H264 for web browser compatibility using ffmpeg
    out_web_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tracked_output_web.mp4")
    if os.path.exists(out_web_path):
        os.remove(out_web_path)
    
    status_text.text("Converting video format for web viewing...")
    subprocess.run(["ffmpeg", "-y", "-i", out_path, "-vcodec", "libx264", "-f", "mp4", out_web_path], capture_output=True)
    
    progress_bar.empty()
    status_text.text(f"Done — {processed}/{total} frames processed (100%)")
    
    if os.path.exists(out_web_path):
        return out_web_path
    return out_path

st.markdown("""
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
---
""")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📹 Upload Input Video")
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "mov", "avi"])
    
    conf_slider = st.slider(
        "Detection Confidence Threshold",
        min_value=0.1, max_value=0.9, value=0.5, step=0.05,
        help="Higher = cleaner detections & more stable IDs. Try 0.4–0.6 for sports."
    )
    
    col1_a, col1_b = st.columns(2)
    with col1_a:
        heatmap_toggle = st.checkbox("🔥 Show Heatmap", value=True)
    with col1_b:
        trace_toggle = st.checkbox("🛤️  Show Trajectories", value=True)
        
    run_btn = st.button("🚀 Run Tracking", type="primary", use_container_width=True)

with col2:
    st.subheader("🎬 Annotated Output Video")
    output_container = st.empty()
    status_text = st.empty()
    progress_bar = st.empty()

if run_btn:
    if uploaded_file is None:
        st.error("Please upload a video first.")
    else:
        # Save uploaded file to temp
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") 
        tfile.write(uploaded_file.read())
        
        # Run tracking
        with st.spinner('Processing video...'):
            final_video_path = run_tracking(
                tfile.name, 
                conf_slider, 
                heatmap_toggle, 
                trace_toggle,
                progress_bar,
                status_text
            )
            
            # Display result
            with open(final_video_path, 'rb') as video_file:
                video_bytes = video_file.read()
                output_container.video(video_bytes)
                
            st.success("Tracking complete!")
