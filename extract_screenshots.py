import cv2
import os

def extract_screenshots(video_path="output_video.mp4", num_screenshots=3, output_dir="screenshots"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        print("Video has no frames.")
        return
        
    # Extract frames at equal intervals
    intervals = [total_frames // (num_screenshots + 1) * i for i in range(1, num_screenshots + 1)]
    
    for i, frame_idx in enumerate(intervals):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            output_path = os.path.join(output_dir, f"screenshot_{i+1}.jpg")
            cv2.imwrite(output_path, frame)
            print(f"Saved screenshot: {output_path}")
        else:
            print(f"Failed to extract frame at index {frame_idx}")

    cap.release()
    print("Screenshot extraction complete.")

if __name__ == "__main__":
    extract_screenshots()
