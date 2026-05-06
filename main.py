import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np

def main():
    # Configuration
    SOURCE_VIDEO_PATH = "input_video.mp4"
    TARGET_VIDEO_PATH = "output_video.mp4"
    MODEL_PATH = "yolov8m.pt"

    # Initialize YOLOv8 model (this will automatically download the weights if not present)
    print("Loading YOLOv8 model...")
    model = YOLO(MODEL_PATH)

    # Initialize video info using supervision
    try:
        video_info = sv.VideoInfo.from_video_path(video_path=SOURCE_VIDEO_PATH)
    except FileNotFoundError:
        print(f"Error: Could not find {SOURCE_VIDEO_PATH}")
        return

    # Initialize ByteTrack
    # Note: ByteTrack is integrated within ultralytics tracker, we can use model.track() directly.
    # supervision also has a ByteTrack implementation which gives us more control.
    print("Initializing ByteTrack...")
    tracker = sv.ByteTrack()

    # Initialize Annotators
    box_annotator = sv.BoxAnnotator(
        thickness=2
    )
    
    heatmap_annotator = sv.HeatMapAnnotator()
    
    label_annotator = sv.LabelAnnotator(
        text_thickness=2,
        text_scale=0.5,
        text_padding=10
    )

    trace_annotator = sv.TraceAnnotator(
        thickness=2,
        trace_length=60,
        position=sv.Position.BOTTOM_CENTER
    )

    print("Processing video...")
    # Define a callback function to process each frame
    def process_frame(frame: np.ndarray, index: int) -> np.ndarray:
        # Run YOLO inference
        # classes=[0] filters for 'person' class only
        results = model(frame, verbose=False, classes=[0])[0]
        
        # Convert ultralytics results to supervision Detections
        detections = sv.Detections.from_ultralytics(results)
        
        # Update tracker with detections
        detections = tracker.update_with_detections(detections)

        # Build labels containing ID and Confidence
        labels = [
            f"#{tracker_id} {confidence:0.2f}"
            for confidence, tracker_id
            in zip(detections.confidence, detections.tracker_id)
        ]

        # Annotate the frame
        annotated_frame = frame.copy()
        
        # Add Heatmap overlay
        annotated_frame = heatmap_annotator.annotate(
            scene=annotated_frame,
            detections=detections
        )
        
        annotated_frame = trace_annotator.annotate(
            scene=annotated_frame, 
            detections=detections
        )
        annotated_frame = box_annotator.annotate(
            scene=annotated_frame, 
            detections=detections
        )
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame, 
            detections=detections, 
            labels=labels
        )

        # Add Object Count Over Time
        count_text = f"Live Object Count: {len(detections)}"
        cv2.putText(
            annotated_frame,
            count_text,
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

        return annotated_frame

    # Process video using supervision's VideoSink
    sv.process_video(
        source_path=SOURCE_VIDEO_PATH,
        target_path=TARGET_VIDEO_PATH,
        callback=process_frame
    )
    
    print(f"Finished processing. Saved to {TARGET_VIDEO_PATH}")

if __name__ == "__main__":
    main()
