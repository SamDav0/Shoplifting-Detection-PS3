import os
import cv2
import json
from ultralytics import YOLO

# Load YOLOv11-pose model
model = YOLO("yolo11n-pose.pt")

input_dir = "Dataset/Shoplifting"
output_base = "Keypoints/Shoplifting"

# Create output directories
bag_output_dir = os.path.join(output_base, "Bag")
pocket_output_dir = os.path.join(output_base, "Pocket")
os.makedirs(bag_output_dir, exist_ok=True)
os.makedirs(pocket_output_dir, exist_ok=True)

def extract_keypoints(video_path, output_json):
    cap = cv2.VideoCapture(video_path)
    frame_keypoints = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        keypoints = results[0].keypoints

        if keypoints is not None and len(keypoints.xy) > 0:
            kps = keypoints.xy[0].tolist()  # (17, 2)
        else:
            kps = [[0, 0] for _ in range(17)]

        frame_keypoints.append(kps)

    cap.release()

    with open(output_json, 'w') as f:
        json.dump(frame_keypoints, f)

# Process Bag videos
for filename in sorted(os.listdir(input_dir)):
    if filename.startswith("Bag") and filename.endswith(".mp4"):
        video_path = os.path.join(input_dir, filename)
        json_path = os.path.join(bag_output_dir, filename.replace(".mp4", ".json"))
        print(f"Processing Bag video {filename}...")
        extract_keypoints(video_path, json_path)

# Process Pocket videos
for filename in sorted(os.listdir(input_dir)):
    if filename.startswith("Pocket") and filename.endswith(".mp4"):
        video_path = os.path.join(input_dir, filename)
        json_path = os.path.join(pocket_output_dir, filename.replace(".mp4", ".json"))
        print(f"Processing Pocket video {filename}...")
        extract_keypoints(video_path, json_path)

print("✅ All shoplifting keypoints extracted to Keypoints/Shoplifting/{Bag,Pocket}/")
