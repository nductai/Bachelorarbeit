import os
import json
import numpy as np
from mmeval import PCKAccuracy

# Paths
base_dir = "D:/TU/7_Semester/Bachelorarbeit/code/Pose-Estimation-ToF/testing/remove/005914"
keypoints_dir = os.path.join(base_dir, "keypoints")
heatmap_dir = os.path.join(base_dir, "heat_map")

# Original GT
original_file = os.path.join(keypoints_dir, "005914_original.json")

def extract_keypoints(kps_flat):
    keypoints = np.array(kps_flat).reshape(-1, 3) # reshapes the flat 51-element list into shape (17, 3)
    coords = keypoints[:, :2] # shape: (17, 2)
    mask = keypoints[:, 2] > 0  # shape: (17, ) → give the visibility mask (whether each point is valid)
    return coords, mask # [x1, y1, v1, x2, y2, v2, ..., x17, y17, v17]

# Load GT
with open(original_file) as f:
    gt_data = json.load(f)

gt_coords, gt_mask = extract_keypoints(gt_data["keypoints"])
bbox = gt_data["bbox"]
bbox_size = np.array([[bbox[2] - bbox[0], bbox[3] - bbox[1]]])

# Init PCK
pck_metric = PCKAccuracy(thr=0.5, norm_item='bbox')
results = {}

# Compute PCK for each prediction
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
        results[filename.replace(".json", "")] = result['PCK@0.5']

# replace values in heatmap npy files
for filename in os.listdir(heatmap_dir):
    if filename.endswith(".npy") and filename.startswith("005914_heatmap_"):
        heatmap_path = os.path.join(heatmap_dir, filename)
        heatmap = np.load(heatmap_path)

        # Extract iteration id (e.g., '0001' from '005914_heatmap_0001.npy')
        iteration_id = filename.replace("005914_heatmap_", "").replace(".npy", "")
        json_basename = f"005914_{iteration_id}"

        if json_basename in results:
            pck_score = results[json_basename]
            heatmap_replaced = np.where(heatmap == 1, pck_score, heatmap)
            np.save(heatmap_path, heatmap_replaced)
            print(f"Updated {filename} with PCK@0.5 = {pck_score:.4f}")
        else:
            print(f"Warning: No PCK score for {json_basename}")





