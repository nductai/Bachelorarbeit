import os
import json
import numpy as np
import matplotlib.pyplot as plt
from mmeval import PCKAccuracy

# --- CONFIG ---
base_dir = "D:/TU/7_Semester/Bachelorarbeit/code/Pose-Estimation-ToF/testing/remove/005914"
keypoints_dir = os.path.join(base_dir, "keypoints")
heatmap_dir = os.path.join(base_dir, "heat_map")
count_dir = os.path.join(base_dir, "count")
original_file = os.path.join(keypoints_dir, "005914_original.json")

# --- FUNCTIONS ---
def extract_keypoints(kps_flat):
    keypoints = np.array(kps_flat).reshape(-1, 3)
    coords = keypoints[:, :2]
    mask = keypoints[:, 2] > 0
    return coords, mask

# --- LOAD GROUND TRUTH ---
with open(original_file) as f:
    gt_data = json.load(f)

gt_coords, gt_mask = extract_keypoints(gt_data["keypoints"])
bbox = gt_data["bbox"]
bbox_size = np.array([[bbox[2] - bbox[0], bbox[3] - bbox[1]]])

# --- COMPUTE PCK SCORES ---
pck_metric = PCKAccuracy(thr=0.05, norm_item='bbox')
results = {}

for filename in os.listdir(keypoints_dir):
    if filename.endswith(".json") and filename != "005914_original.json":
        pred_path = os.path.join(keypoints_dir, filename)

        with open(pred_path) as f:
            pred_data = json.load(f)

        pred_coords, _ = extract_keypoints(pred_data["keypoints"])

        predictions = [{'coords': pred_coords[np.newaxis, :, :]}]
        groundtruths = [{
            'coords': gt_coords[np.newaxis, :, :],
            'mask': gt_mask[np.newaxis, :],
            'bbox_size': bbox_size
        }]

        result = pck_metric(predictions, groundtruths)
        results[filename.replace(".json", "")] = result['PCK@0.05']

# --- UPDATE HEATMAP FILES ---
for filename in os.listdir(heatmap_dir):
    if filename.endswith(".npy") and filename.startswith("005914_heatmap_"):
        heatmap_path = os.path.join(heatmap_dir, filename)
        heatmap = np.load(heatmap_path)

        iteration_id = filename.replace("005914_heatmap_", "").replace(".npy", "")
        json_basename = f"005914_{iteration_id}"

        if json_basename in results:
            pck_score = results[json_basename]
            heatmap_replaced = np.where(heatmap == 1, pck_score, heatmap)
            np.save(heatmap_path, heatmap_replaced)
            print(f"Updated {filename} with PCK@0.05 = {pck_score:.4f}")
        else:
            print(f"Warning: No PCK score for {json_basename}")

# --- SUM HEATMAPS ---
heatmap_final = None
for filename in os.listdir(heatmap_dir):
    if filename.endswith(".npy") and filename != "005914_heatmap_original.npy":
        data = np.load(os.path.join(heatmap_dir, filename))
        if heatmap_final is None:
            heatmap_final = np.zeros_like(data, dtype=np.float32)
        heatmap_final += data

final_heatmap_path = os.path.join(heatmap_dir, "005914_heatmap_final.npy")
np.save(final_heatmap_path, heatmap_final)
print(f"Saved final heatmap sum to {final_heatmap_path}")

# --- SUM COUNT MAPS ---
count_final = None
for filename in os.listdir(count_dir):
    if filename.endswith(".npy") and filename.startswith("005914_"):
        data = np.load(os.path.join(count_dir, filename))
        if count_final is None:
            count_final = np.zeros_like(data, dtype=np.float32)
        count_final += data

final_count_path = os.path.join(count_dir, "005914_final.npy")
np.save(final_count_path, count_final)
print(f"Saved final count sum to {final_count_path}")

# --- CALCULATE AVERAGE HEATMAP ---
with np.errstate(divide='ignore', invalid='ignore'):
    avg_map = np.true_divide(heatmap_final, count_final)
    avg_map[~np.isfinite(avg_map)] = 0

avg_map_path = os.path.join(base_dir, "005914_avg_map.npy")
np.save(avg_map_path, avg_map)
print(f"Saved averaged map to {avg_map_path}")

# --- SAVE AVERAGE MAP AS IMAGE ---
plt.figure(figsize=(6, 6))
plt.imshow(avg_map, cmap='RdYlBu_r', interpolation='nearest')
plt.colorbar(label='Average PCK@0.05')
plt.title('Average PCK Heatmap')
plt.tight_layout()

avg_img_path = os.path.join(base_dir, "005914_avg_map.png")
plt.savefig(avg_img_path)
plt.close()
print(f"Saved heatmap visualization to {avg_img_path}")


