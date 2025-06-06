import cv2
import numpy as np
import os

image_path = r'D:\TU\7_Semester\Bachelorarbeit\sample\005985.png'
output_dir = r'D:\TU\7_Semester\Bachelorarbeit\sample\removed'
os.makedirs(output_dir, exist_ok=True)

grid_size = 10  # Size of one "big pixel"
removal_prob = 0.1  # Probability to remove a cell

original_image = cv2.imread(image_path)
height, width, _ = original_image.shape

# Compute grid dimensions
grid_rows = height // grid_size
grid_cols = width // grid_size

for iteration in range(1, 1001):
    image = original_image.copy()

    # Create binary mask with 0 (remove) and 1 (keep)
    mask = np.random.choice([0, 1], size=(grid_rows, grid_cols), p=[removal_prob, 1 - removal_prob])

    removed_grids = []

    for row in range(grid_rows):
        for col in range(grid_cols):
            if mask[row, col] == 0:
                top_left_x = col * grid_size
                top_left_y = row * grid_size

                image[top_left_y:top_left_y + grid_size, top_left_x:top_left_x + grid_size] = [0, 0, 0]

                removed_grids.append((top_left_x, top_left_y))

    output_image_path = os.path.join(output_dir, f'modified_output_{iteration:02d}.png')
    cv2.imwrite(output_image_path, image)

    print(f'Iteration {iteration}: Removed {len(removed_grids)} grid(s)- big pixel(s)')
    print(f'Grid locations: {removed_grids}')
    print(f'Modified image saved to: {output_image_path}\n')












