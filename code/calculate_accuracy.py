import os
import json
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# --- CONFIG ---
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parent.parent  # -> Bachelorarbeit

base_dir = REPO_ROOT / "code" / "Pose-Estimation-ToF" / "testing" / "remove" / "005914"
gt_keypoints_dir = base_dir / "threshold_1_0" / "images" / "keypoints"

r = 20.0

# --- LOGGING SETUP ---
log_path = os.path.join(base_dir, "debug_log.txt")
log_file = open(log_path, "w")

def log_print(*args, **kwargs):
    print(*args, **kwargs)
    print(*args, **kwargs, file=log_file)

# --- ACCURACY ALGORITHM ---
def compute_accuracy(l2_error, r):
    if np.isnan(l2_error):
        return 0.0
    elif l2_error >= r:
        return 0.0
    else:
        return 1.0 - (l2_error / r)

def extract_keypoints(kps_flat):
    keypoints = np.array(kps_flat).reshape(-1, 3)
    coords = keypoints[:, :2]
    mask = keypoints[:, 2] > 0
    return coords, mask

def compute_l2_distance(pred_coords, gt_coords, gt_mask, pred_mask, keypoint_idx=None):
    visible = np.where(gt_mask & pred_mask)[0]

    if len(visible) == 0:
        return float('nan')

    if keypoint_idx is not None:
        if not (gt_mask[keypoint_idx] and pred_mask[keypoint_idx]):
            return float('nan')  # skip if either invisible
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
log_print(f"Using ground truth JSON file: {gt_json_files[0]}")

with open(original_file) as f:
    gt_data = json.load(f)

gt_coords, gt_mask = extract_keypoints(gt_data["keypoints"])
log_print(f"Ground truth visible keypoints count: {np.sum(gt_mask)}")
log_print(f"Ground truth keypoints visible score: {np.array(gt_data['keypoints'])[2::3]}")

# --- LOOP THROUGH ALL THRESHOLD FOLDERS ---
all_results = {}

for subdir in os.listdir(base_dir):

    #if subdir != "threshold_0_4":  # TODO: remove this line to run all thresholds
    #    continue

    pred_keypoints_dir = os.path.join(base_dir, subdir, "images", "keypoints")
    mask_dir = os.path.join(base_dir, subdir, "masks")
    if not os.path.isdir(pred_keypoints_dir):
        continue

    # LOOP THROUGH ALL KEYPOINTS
    for keypoint_idx in range(17):

        total_l2_error = 0.0
        file_count = 0

        # Create subfolder for this keypoint
        keypoint_subdir = os.path.join(mask_dir, f"keypoint_{keypoint_idx}")
        os.makedirs(keypoint_subdir, exist_ok=True)

        for filename in os.listdir(pred_keypoints_dir):
            if filename.endswith(".json") and filename.startswith("005914_"):
                pred_path = os.path.join(pred_keypoints_dir, filename)
                with open(pred_path) as f:
                    pred_data = json.load(f)
                pred_coords, pred_mask = extract_keypoints(pred_data["keypoints"])

                iteration_number = filename.rsplit('_', 1)[-1].replace(".json", "")
                #log_print(f"\nProcessing {filename} (iteration {iteration_number}) | keypoint {keypoint_idx}")

                # Calculate error for this keypoint
                l2_error = compute_l2_distance(pred_coords, gt_coords, gt_mask, pred_mask, keypoint_idx=keypoint_idx)
                accuracy = compute_accuracy(l2_error, r)

                if not np.isnan(l2_error):
                    log_print(f"  GT coords[{keypoint_idx}] = {gt_coords[keypoint_idx]} (mask={gt_mask[keypoint_idx]})")
                    log_print(f"  Pred coords[{keypoint_idx}] = {pred_coords[keypoint_idx]} (mask={pred_mask[keypoint_idx]})")
                    log_print(f"  L2 distance = {l2_error:.4f}")
                    log_print(f"  Accuracy (r={r}) = {accuracy:.4f}")
                else:
                    log_print(f"  Skipped keypoint {keypoint_idx} (not visible in GT or prediction)")

                # Replace heatmap values with accuracy
                heatmap_path = os.path.join(mask_dir, f"heatmap_{iteration_number}.npy")
                if os.path.exists(heatmap_path):
                    heatmap = np.load(heatmap_path)
                    heatmap_accuracy = np.where(heatmap == 1, accuracy, heatmap)
                    save_path = os.path.join(keypoint_subdir, f"heatmap_accuracy_{iteration_number}.npy")
                    np.save(save_path, heatmap_accuracy)
                    log_print(f"  Updated heatmap: {heatmap_path} -> {save_path}")
                else:
                    log_print(f"  Warning: No heatmap found for iteration {iteration_number}")

                if not np.isnan(l2_error):
                    total_l2_error += l2_error
                    file_count += 1

        # --- Sum all heatmap_accuracy_xxxx.npy files for this keypoint ---
        heatmap_acc_files = [
            f for f in os.listdir(keypoint_subdir)
            if f.startswith("heatmap_accuracy_") and f.endswith(".npy") and "final" not in f]

        summed_heatmap = None
        for f in heatmap_acc_files:
            arr = np.load(os.path.join(keypoint_subdir, f))
            if summed_heatmap is None:
                summed_heatmap = np.zeros_like(arr, dtype=float)
            summed_heatmap += arr
        if summed_heatmap is not None:
            np.save(os.path.join(keypoint_subdir, "heatmap_accuracy_final.npy"), summed_heatmap)

        # --- Sum all count_xxxx.npy files ---
        count_files = [f for f in os.listdir(mask_dir)
                       if f.startswith("count_") and f.endswith(".npy")and "final" not in f]
        summed_count = None
        for f in count_files:
            arr = np.load(os.path.join(mask_dir, f))
            if summed_count is None:
                summed_count = np.zeros_like(arr, dtype=float)
            summed_count += arr
        if summed_count is not None:
            np.save(os.path.join(keypoint_subdir, "count_final.npy"), summed_count)

        # --- Divide to get final heatmap ---
        if summed_heatmap is not None and summed_count is not None:
            with np.errstate(divide='ignore', invalid='ignore'):
                final_heatmap = np.true_divide(summed_heatmap, summed_count)
                final_heatmap[~np.isfinite(final_heatmap)] = 0
            np.save(os.path.join(keypoint_subdir, "final_heatmap.npy"), final_heatmap)

            # --- Plot final heatmap ---
            plt.figure(figsize=(8, 6))
            plt.imshow(final_heatmap, cmap='jet_r', interpolation='nearest')
            plt.colorbar(label='Accuracy')
            plt.title(f'Final Heatmap | Keypoint {keypoint_idx} | Threshold {subdir}')
            plt.axis('off')
            plt.savefig(os.path.join(keypoint_subdir, "final_heatmap.png"), dpi=300, bbox_inches='tight')
            plt.close()

        avg_l2_error = total_l2_error / file_count if file_count > 0 else float('nan')
        all_results[f"{subdir}_kp{keypoint_idx}"] = avg_l2_error
        log_print(f"Processed {file_count} files for keypoint {keypoint_idx} in '{subdir}'. "
                  f"Average L2 error = {avg_l2_error}")

log_print("\n=== L2 Error Summary Across All Threshold Folders ===")
for folder, error in all_results.items():
    log_print(f"{folder}: Average L2 Error = {error}")

log_print("\n=== L2 Error & Accuracy Summary Across All Threshold Folders ===")
for folder, error in all_results.items():
    accuracy = compute_accuracy(error, r)
    log_print(f"{folder}: Avg L2 = {error} | Accuracy (r={r}) = {accuracy}")

# --- PLOT ACCURACY VS THRESHOLD FOR ALL 17 KEYPOINTS ---
def plot_accuracy_vs_threshold_all_kps(results_dict, radius, save_dir):
    keypoints = sorted(set(int(k.split("_kp")[-1]) for k in results_dict.keys()))

    for kp in keypoints:
        thresholds = []
        accuracies = []

        for folder, error in results_dict.items():
            if folder.endswith(f"kp{kp}"):
                try:
                    parts = folder.split("_")
                    if len(parts) >= 3 and parts[0] == "threshold":
                        t_str = f"{parts[1]}.{parts[2]}" if parts[2].isdigit() else parts[1]
                        threshold = float(t_str)
                    else:
                        threshold = float(parts[1])

                    thresholds.append(threshold)
                    accuracies.append(compute_accuracy(error, radius))
                except Exception as e:
                    log_print(f"Skipping {folder}: {e}")

        if thresholds:
            thresholds, accuracies = zip(*sorted(zip(thresholds, accuracies)))
            plt.figure(figsize=(8, 5))
            plt.plot(thresholds, accuracies, marker='o', linestyle='-', color='blue')
            for x, y in zip(thresholds, accuracies):
                plt.text(x, y + 0.02, f"{y:.2f}", ha='center', va='bottom', fontsize=8)

            plt.title(f"Accuracy vs Threshold | Keypoint {kp} (r = {radius})")
            plt.xlabel("Threshold")
            plt.ylabel("Accuracy")
            plt.grid(True)
            plt.ylim(0, 1.1)
            plt.tight_layout()

            plot_save_path = os.path.join(save_dir, f"accuracy_vs_threshold_kp{kp}.png")
            plt.savefig(plot_save_path, dpi=300)
            plt.close()
            log_print(f"Saved plot for keypoint {kp} to: {plot_save_path}")

#plot_accuracy_vs_threshold_all_kps(all_results, r, base_dir)


