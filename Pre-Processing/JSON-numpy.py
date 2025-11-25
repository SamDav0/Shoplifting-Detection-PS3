import json
import numpy as np
import os

# --- 1. Configuration ---
SEQUENCE_LENGTH = 240   # 8 seconds @ 30fps

base_path = '../'   # Parent directory
output_folder_path = os.path.join(base_path, 'Processed-Dataset-Numpy')
os.makedirs(output_folder_path, exist_ok=True)
print(f"Output will be saved in: {output_folder_path}")

# Dataset paths (relative to parent folder)
data_paths = {
    "normal": os.path.join(base_path, 'Dataset-JSON/Normal-JSON/'),
    "shoplifting_bag": os.path.join(base_path, 'Dataset-JSON/Shoplifting-JSON/Bag/'),
    "shoplifting_pocket": os.path.join(base_path, 'Dataset-JSON/Shoplifting-JSON/Pocket/')
}

# --- 2. Define new label map (2-class) ---
label_map = {
    "normal": 0,
    "shoplifting_bag": 1,
    "shoplifting_pocket": 1
}

# --- 3. Load and process data ---
all_data = []
all_labels = []

print("\nLoading and processing files (2-class, no sliding window)...")

for category, folder_path in data_paths.items():
    label = label_map[category]
    if not os.path.isdir(folder_path):
        print(f"Skipping {category}, folder not found: {folder_path}")
        continue

    for file_name in os.listdir(folder_path):
        if not file_name.endswith('.json'):
            continue

        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r') as f:
            raw_data = json.load(f)
            keypoints = np.array([
                frame for frame in raw_data 
                if isinstance(frame, list) and len(frame) == 17
            ])

            if len(keypoints) == 0:
                print(f"Skipping {file_name}, no valid frames found.")
                continue

            # Pad or truncate to fixed SEQUENCE_LENGTH
            num_frames = len(keypoints)
            padded_keypoints = np.zeros((SEQUENCE_LENGTH, 17, 2))
            copy_len = min(num_frames, SEQUENCE_LENGTH)
            padded_keypoints[:copy_len] = keypoints[:copy_len]

            all_data.append(padded_keypoints)
            all_labels.append(label)

# --- 4. Save dataset ---
X_final = np.array(all_data, dtype=np.float32)
y_final = np.array(all_labels, dtype=np.int32)

print("\n--- Final Dataset Shapes (2-Class, No-Sliding) ---")
print(f"Total samples created: {len(X_final)}")
print(f"X_final shape: {X_final.shape}")
print(f"y_final shape: {y_final.shape}")

# Count per class
unique, counts = np.unique(y_final, return_counts=True)
class_counts = dict(zip(unique, counts))
print(f"\nClass distribution: {class_counts}")

np.save(os.path.join(output_folder_path, 'X_data_2class.npy'), X_final)
np.save(os.path.join(output_folder_path, 'y_labels_2class.npy'), y_final)

print(f"\nSuccessfully saved 2-class (no-slide) dataset to {output_folder_path}")
