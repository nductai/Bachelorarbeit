import cv2
import numpy as np
import os
import random
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parent.parent  # -> Bachelorarbeit

input_dir = REPO_ROOT / "code" / "Pose-Estimation-ToF" / "testing"
output_base_dir = input_dir / "remove"  # Base output folder

grid_size = 5             # Size of one "big pixel"
num_iterations = 1000

# Target image file
target_file = '005914.png'
image_path = input_dir / target_file

if not image_path.exists():
    print(f"{target_file} does not exist in the directory.")
    exit()

original_image = cv2.imread(str(image_path))
if original_image is None:
    print(f"Failed to read {image_path}")
    exit()

image_name, ext = os.path.splitext(target_file)
height, width, _ = original_image.shape
grid_rows = height // grid_size
grid_cols = width // grid_size

# Output directory for this image
image_output_dir = output_base_dir / image_name
os.makedirs(str(image_output_dir), exist_ok=True)

# Save the original image
#original_save_path = os.path.join(image_output_dir, f'{image_name}_original{ext}')
#cv2.imwrite(original_save_path, original_image)

# threshold values: 0.0 to 1.0 inclusive, step 0.1
threshold_values = np.arange(0.0, 1.01, 0.1)

for threshold in threshold_values:
    thresh_str = f'{threshold:.1f}'.replace('.', '_')  # e.g., 0.3 → "0_3"

    # create subfolders for this threshold ===
    threshold_dir = image_output_dir / f"threshold_{thresh_str}"
    image_dir = threshold_dir / "images"
    mask_dir = threshold_dir / "masks"
    os.makedirs(str(image_dir), exist_ok=True)
    os.makedirs(str(mask_dir), exist_ok=True)

    # only 1 iteration if threshold = 0.0 or 1.0
    iterations = 1 if threshold in [0.0, 1.0] else num_iterations

    for iteration in range(1, iterations + 1):
        image = original_image.copy()
        mask = np.zeros((grid_rows, grid_cols), dtype=np.uint8)
        removed_grids = []

        # create and shuffle grid list
        all_grids = [(row, col) for row in range(grid_rows) for col in range(grid_cols)]
        random.shuffle(all_grids)

        # process each grid in shuffled order
        for row, col in all_grids:
            rand_thresh = random.uniform(0, 1)
            if rand_thresh > threshold:
                top_left_x = col * grid_size
                top_left_y = row * grid_size
                image[top_left_y:top_left_y + grid_size, top_left_x:top_left_x + grid_size] = [0, 0, 0]
                mask[row, col] = 1  # set to 1 if that pixel got removed
                removed_grids.append((row, col))

        # save outputs
        output_image_path = image_dir / f"{image_name}_thresh_{thresh_str}_{iteration:04d}{ext}"
        heatmap_path = mask_dir / f"heatmap_{iteration:04d}.npy"
        count_path = mask_dir / f"count_{iteration:04d}.npy"

        cv2.imwrite(str(output_image_path), image)
        np.save(str(heatmap_path), mask)
        np.save(str(count_path), mask)

        print(f"Threshold {threshold:.1f} - Iteration {iteration}: Removed {len(removed_grids)} grid(s)")

















