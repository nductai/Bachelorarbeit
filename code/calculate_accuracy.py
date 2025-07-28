import os
import json
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIG ---
base_dir = r"D:\TU\7_Semester\Bachelorarbeit\code\Pose-Estimation-ToF\testing\remove\005914"
gt_keypoints_dir = os.path.join(base_dir, "threshold_1_0", "images", "keypoints")

def extract_keypoints(kps_flat):
    keypoints = np.array(kps_flat).reshape(-1, 3)
    coords = keypoints[:, :2]
    mask = keypoints[:, 2] > 0
    return coords, mask

def compute_l2_distance(pred_coords, gt_coords, gt_mask, keypoint_idx=None):
    visible = np.where(gt_mask)[0]
    if len(visible) == 0:
        print("Warning: No visible keypoints in ground truth!")
        return float('nan')

    if keypoint_idx is not None:
        diff = pred_coords[keypoint_idx] - gt_coords[keypoint_idx]
        dist = np.linalg.norm(diff)
        return dist
    else:
        diffs = pred_coords[visible] - gt_coords[visible]
        dists = np.linalg.norm(diffs, axis=1)
        return dists

# --- LOAD GROUND TRUTH ---
gt_json_files = [f for f in os.listdir(gt_keypoints_dir) if f.endswith(".json")]
assert len(gt_json_files) > 0, f"No JSON files found in {gt_keypoints_dir}"
original_file = os.path.join(gt_keypoints_dir, gt_json_files[0])
print(f"Using ground truth JSON file: {gt_json_files[0]}")

with open(original_file) as f:
    gt_data = json.load(f)

gt_coords, gt_mask = extract_keypoints(gt_data["keypoints"])
print(f"Ground truth visible keypoints count: {np.sum(gt_mask)}")
print(f"Ground truth keypoints visible score: {np.array(gt_data['keypoints'])[2::3]}")

# --- LOOP THROUGH ALL THRESHOLD FOLDERS ---
all_results = {}

for subdir in os.listdir(base_dir):
    pred_keypoints_dir = os.path.join(base_dir, subdir, "images", "keypoints")
    if not os.path.isdir(pred_keypoints_dir):
        continue

    print(f"\n--- Evaluating folder: {subdir} ---")
    total_l2_error = 0.0
    file_count = 0

    for filename in os.listdir(pred_keypoints_dir):
        if filename.endswith(".json") and filename.startswith("005914_"):
            pred_path = os.path.join(pred_keypoints_dir, filename)
            with open(pred_path) as f:
                pred_data = json.load(f)
            pred_coords, _ = extract_keypoints(pred_data["keypoints"])

            print(f"\nProcessing {filename}:")
            l2_error = compute_l2_distance(pred_coords, gt_coords, gt_mask, keypoint_idx=0)  # keypoint_idx = None if we want to calculate all keypoints
            print(f"L2 error = {l2_error}")

            if isinstance(l2_error, np.ndarray):
                file_l2_sum = np.sum(l2_error)
            else:
                file_l2_sum = l2_error

            if not np.isnan(file_l2_sum):
                total_l2_error += file_l2_sum
                file_count += 1

    avg_l2_error = total_l2_error / file_count if file_count > 0 else float('nan')
    all_results[subdir] = avg_l2_error
    print(f"Processed {file_count} files in '{subdir}'. Average L2 error = {avg_l2_error}")

print("\n=== L2 Error Summary Across All Threshold Folders ===")
for folder, error in all_results.items():
    print(f"{folder}: Average L2 Error = {error}")

# --- ACCURACY ALGORITHM ---
r = 20.0
def compute_accuracy(l2_error, r):
    if np.isnan(l2_error):
        return float('nan')
    elif l2_error >= r:
        return 0.0
    else:
        return 1.0 - (l2_error / r)

print("\n=== L2 Error & Accuracy Summary Across All Threshold Folders ===")
for folder, error in all_results.items():
    accuracy = compute_accuracy(error, r)
    print(f"{folder}: Avg L2 = {error:.3f} | Accuracy (r={r}) = {accuracy:.3f}")

# --- PLOT ACCURACY VS THRESHOLD ---
thresholds = []
accuracies = []

for folder, error in all_results.items():
    try:
        parts = folder.split("_")
        if len(parts) == 3:
            t_str = f"{parts[1]}.{parts[2]}"
            threshold = float(t_str)
        elif len(parts) == 2:
            threshold = float(parts[1])
        else:
            continue

        accuracy = compute_accuracy(error, r)
        thresholds.append(threshold)
        accuracies.append(accuracy)
    except Exception as e:
        print(f"Skipping folder {folder}: {e}")

# Sort for proper plotting
thresholds, accuracies = zip(*sorted(zip(thresholds, accuracies)))

# --- PLOT ---
plt.figure(figsize=(8, 5))
plt.plot(thresholds, accuracies, marker='o', linestyle='-', color='blue')

for x, y in zip(thresholds, accuracies):
    plt.text(x, y + 0.02, f"{y:.2f}", ha='center', va='bottom', fontsize=8)

plt.title(f"Accuracy vs Threshold (r = {r})")
plt.xlabel("Threshold")
plt.ylabel("Accuracy")
plt.grid(True)
plt.xticks(thresholds, rotation=45)
plt.ylim(0, 1.1)
plt.tight_layout()

plot_save_path = os.path.join(base_dir, f"accuracy_vs_threshold.png")
plt.savefig(plot_save_path, dpi=300)
print(f"\nPlot saved to: {plot_save_path}")

plt.show()








