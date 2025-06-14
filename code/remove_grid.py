import cv2
import numpy as np
import os

# Path settings
input_dir = r'D:\TU\7_Semester\Bachelorarbeit\code\Pose-Estimation-ToF\testing'
output_base_dir = os.path.join(input_dir, 'remove')  # Base output folder

grid_size = 10            # Size of one "big pixel"
removal_prob = 0.1        # Probability to remove a cell
num_iterations = 1000      # Number of modified images per original

# List all image files in the folder
# image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

target_file = '005914.png'
image_path = os.path.join(input_dir, target_file)

if not os.path.exists(image_path):
    print(f"{target_file} does not exist in the directory.")
    exit()

image_files = [target_file]

for image_file in image_files:
    image_path = os.path.join(input_dir, image_file)
    image_name, ext = os.path.splitext(image_file)

    # Output directory for this image
    image_output_dir = os.path.join(output_base_dir, image_name)
    os.makedirs(image_output_dir, exist_ok=True)

    # Load image
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Failed to read {image_path}")
        continue

    height, width, _ = original_image.shape
    grid_rows = height // grid_size
    grid_cols = width // grid_size

    # Save the original image
    original_save_path = os.path.join(image_output_dir, f'{image_name}_original{ext}')
    cv2.imwrite(original_save_path, original_image)

    # Create heat_map and count directories
    heat_map_dir = os.path.join(image_output_dir, 'heat_map')
    count_dir = os.path.join(image_output_dir, 'count')
    os.makedirs(heat_map_dir, exist_ok=True)
    os.makedirs(count_dir, exist_ok=True)

    for iteration in range(1, num_iterations + 1):
        image = original_image.copy()

        # Generate mask
        mask = np.random.choice([0, 1], size=(grid_rows, grid_cols), p=[removal_prob, 1 - removal_prob])

        # Save mask in heat_map folder
        heatmap_path = os.path.join(heat_map_dir, f'{image_name}_heatmap_{iteration:04d}.npy')
        np.save(heatmap_path, mask)

        # Save same mask in count folder
        count_path = os.path.join(count_dir, f'{image_name}_{iteration:04d}.npy')
        np.save(count_path, mask)

        removed_grids = []
        for row in range(grid_rows):
            for col in range(grid_cols):
                if mask[row, col] == 0:
                    top_left_x = col * grid_size
                    top_left_y = row * grid_size
                    image[top_left_y:top_left_y + grid_size, top_left_x:top_left_x + grid_size] = [0, 0, 0]
                    removed_grids.append((top_left_x, top_left_y))

        output_image_path = os.path.join(image_output_dir, f'{image_name}_{iteration:04d}{ext}')
        cv2.imwrite(output_image_path, image)

        # Print required info
        print(f'{image_file} - Iteration {iteration}: Removed {len(removed_grids)} grid(s)')
        print(f'Grid locations: {removed_grids}')
        print(f'Modified image saved to: {output_image_path}\n')













