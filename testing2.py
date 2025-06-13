import os
import numpy as np
import matplotlib.pyplot as plt

# Paths
base_path = "D:/TU/7_Semester/Bachelorarbeit/code/Pose-Estimation-ToF/testing/remove/005914"
heatmap_dir = os.path.join(base_path, "heat_map")
count_dir = os.path.join(base_path, "count")

# Final accumulation arrays
heatmap_final = None
count_final = None

# --- Sum HEATMAPS ---
for filename in os.listdir(heatmap_dir):
    if filename.endswith(".npy") and filename != "005914_heatmap_original.npy":
        filepath = os.path.join(heatmap_dir, filename)
        data = np.load(filepath)

        if heatmap_final is None:
            heatmap_final = np.zeros_like(data, dtype=np.float32)

        heatmap_final += data

# Save final heatmap sum
final_heatmap_path = os.path.join(heatmap_dir, "005914_heatmap_final.npy")
np.save(final_heatmap_path, heatmap_final)
print(f"Saved final heatmap sum to {final_heatmap_path}")

# --- Sum COUNT MAPS ---
for filename in os.listdir(count_dir):
    if filename.endswith(".npy") and filename.startswith("005914_"):
        filepath = os.path.join(count_dir, filename)
        data = np.load(filepath)

        if count_final is None:
            count_final = np.zeros_like(data, dtype=np.float32)

        count_final += data

# Save final count sum
final_count_path = os.path.join(count_dir, "005914_final.npy")
np.save(final_count_path, count_final)
print(f"Saved final count sum to {final_count_path}")

# --- Calculate Average Heatmap (Heatmap / Count) ---
with np.errstate(divide='ignore', invalid='ignore'):
    avg_map = np.true_divide(heatmap_final, count_final)
    avg_map[~np.isfinite(avg_map)] = 0  # Replace NaN and inf with 0

# Save averaged map as .npy
avg_map_path = os.path.join(base_path, "005914_avg_map.npy")
np.save(avg_map_path, avg_map)
print(f"Saved averaged map to {avg_map_path}")

# --- Save as Heatmap Image ---
plt.figure(figsize=(6, 6))
plt.imshow(avg_map, cmap='RdYlBu_r', interpolation='nearest')
plt.colorbar(label='Average PCK@0.5')
plt.title('Average PCK Heatmap')
plt.tight_layout()

avg_img_path = os.path.join(base_path, "005914_avg_map.png")
plt.savefig(avg_img_path)
plt.close()
print(f"Saved heatmap visualization to {avg_img_path}")
