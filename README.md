# Multi-Object Detection and Tracking in Sports Footage

> **Original Public Video Link**: The video used for testing and output generation is a public Soccer Highlights clip available at: [https://www.youtube.com/watch?v=KtZyv-KGFa8](https://www.youtube.com/watch?v=KtZyv-KGFa8)

This project implements a robust computer vision pipeline for multi-object detection and persistent ID tracking in sports/event footage. It utilizes **YOLOv8** for state-of-the-art object detection and **ByteTrack** for handling multi-object tracking, especially in scenarios with occlusion and rapid movement.

## Features
- **Object Detection**: Identifies players and individuals (Class 0: `person`) using YOLOv8.
- **Persistent Tracking**: Assigns unique IDs to detected subjects across frames utilizing ByteTrack.
- **Annotation & Visualization**: Draws bounding boxes, unique IDs, and recent movement trajectories using the `supervision` library.

## Dependencies

- Python 3.10+
- `ultralytics` (YOLOv8)
- `supervision`
- `opencv-python`
- `yt-dlp` (for downloading the sample video)

## Installation Steps

1. Clone or download this repository.
2. Create and activate a Python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run the Pipeline

1. **Provide a Video**: Ensure you have an `input_video.mp4` file in the project directory. If you don't, you can use `yt-dlp` to download a short public sports clip.
   ```bash
   yt-dlp "https://www.youtube.com/watch?v=kR2vKqjF2k8" -f "best[ext=mp4]" -o "input_video.mp4"
   ```
2. **Run the Script**: Execute the main pipeline script.
   ```bash
   python main.py
   ```
3. **Check the Output**: The processed video will be saved as `output_video.mp4` in the same directory.

## Assumptions Taken

1. **Camera Position**: Assumes a relatively stable camera view, though ByteTrack handles panning reasonably well.
2. **Target Class**: Assumes the main subjects of interest are humans (`class 0`). The script is filtered to only track people, filtering out irrelevant objects like cars or background elements.
3. **Video Format**: Assumes standard 1080p or 720p 30/60 FPS video input.

## Limitations

1. **Severe Occlusion**: While ByteTrack is robust, prolonged overlapping of subjects (e.g., players hugging or dogpiling) may still cause an ID swap upon separation.
2. **Extreme Blur**: Fast camera whips causing severe motion blur may drop detections for a few frames. ByteTrack's buffer helps recover IDs, but extremely long blur periods might assign a new ID.
3. **Computational Load**: Running YOLOv8m (medium) requires a decent CPU/GPU. For real-time performance on lower-end hardware, switching to `yolov8n.pt` (nano) is recommended.

## Model and Tracker Choices

- **Model: YOLOv8m**: Chosen for its excellent balance of inference speed and detection accuracy, crucial for picking up smaller subjects in wide-angle sports shots.
- **Tracker: ByteTrack**: Chosen over DeepSORT because ByteTrack utilizes low-confidence detections instead of discarding them. This makes it highly effective at maintaining tracks when subjects are partially occluded or blurred.
