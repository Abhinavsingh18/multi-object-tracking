# Technical Report: Multi-Object Detection and Persistent ID Tracking

## 1. Model and Detector Selection
For this computer vision pipeline, **YOLOv8** (specifically `yolov8m.pt`) was selected as the core object detector. YOLOv8 is a state-of-the-art, single-stage object detector that provides an optimal balance between inference speed and bounding box accuracy. 
- **Why YOLOv8:** In sports and event footage, subjects (players) can range significantly in size depending on camera zoom, and rapid movements are common. YOLOv8 excels at detecting small objects and operates fast enough to process video in near real-time, which is crucial for scalable video analytics. The model is pre-trained on the COCO dataset, making it natively capable of detecting the `person` class with high confidence without fine-tuning.

## 2. Tracking Algorithm Selection
The chosen tracking algorithm is **ByteTrack**. 
- **Why ByteTrack:** Traditional trackers (like DeepSORT) typically filter out bounding boxes with low confidence scores. In sports videos, a player might temporarily have a low confidence score due to motion blur, partial occlusion, or irregular posture. ByteTrack innovates by associating almost every detection box (even those with low scores) to tracklets. This drastically reduces the number of lost tracks and fragmented IDs. ByteTrack uses Kalman Filters to predict the next location of a tracklet, comparing it with incoming detections using Intersection over Union (IoU) matching.

## 3. Maintaining ID Consistency
ID consistency is a primary challenge in multi-object tracking. The pipeline maintains persistent IDs through the following mechanisms:
- **Low-Confidence Matching:** By utilizing ByteTrack, if a player is partially occluded (confidence drops), the tracker still associates the low-confidence detection with the existing ID based on spatial overlap (IoU), rather than dropping the ID and assigning a new one once the player is fully visible again.
- **Lost Track Buffer:** The tracker is configured with a buffer (e.g., 30 frames). If a player completely disappears behind another object, their ID is held in memory for 30 frames. When they reappear in the predicted area, the original ID is restored.

## 4. Challenges Faced and Failure Cases
While robust, the pipeline is not entirely immune to complex real-world challenges:
- **Severe Occlusions (ID Swapping):** When two or more players converge tightly (e.g., a tackle in football or a celebratory hug), their bounding boxes merge entirely. Upon separating, the Kalman filter predictions might overlap, leading to an **ID swap** where Player A is assigned Player B's ID.
- **Extreme Camera Whips:** Sudden and fast camera panning can cause significant motion blur across all subjects simultaneously. In some instances, the displacement between consecutive frames is so large that spatial IoU matching fails, causing the tracker to assign new IDs to existing subjects.

## 5. Possible Improvements
To further enhance the pipeline's robustness, several improvements could be implemented:
- **Re-Identification (ReID) Model:** Integrating an appearance-based ReID model (like BoT-SORT or DeepSORT) would extract visual feature embeddings (such as jersey color/number). This would help correct ID swaps during severe occlusions by matching subjects based on how they look, not just where they are.
- **Camera Motion Compensation:** Implementing algorithms to estimate and compensate for global camera motion (using techniques like Optical Flow or ECC) would improve Kalman filter predictions during heavy camera panning.
- **Role/Team Clustering:** Using K-Means clustering on the dominant color of the bounding boxes could automatically classify subjects into teams (e.g., Team A vs Team B), adding valuable contextual analytics.
