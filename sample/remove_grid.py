import cv2
import numpy as np
import random
import os

image_path = r'D:\TU\7_Semester\Bachelorarbeit\sample\005985.png'
output_dir = r'D:\TU\7_Semester\Bachelorarbeit\sample\removed'

os.makedirs(output_dir, exist_ok=True)

for iteration in range(1, 11):
    image = cv2.imread(image_path)

    height, width, _ = image.shape

    grid_size = 10 #TODO: grid size

    grid_rows = height // grid_size
    grid_cols = width // grid_size

    grids_removed = set()
    pixels_to_remove = 10
    removed_grids = []

    while len(grids_removed) < pixels_to_remove:
        # Choose a random grid
        random_row = random.randint(0, grid_rows - 1)
        random_col = random.randint(0, grid_cols - 1)

        grid_index = (random_row, random_col)

        if grid_index in grids_removed:
            continue  # skip if this grid already had a pixel removed

        top_left_x = random_col * grid_size
        top_left_y = random_row * grid_size

        # count how many non-black pixels are in the grid
        non_black_pixel_count = 0
        for i in range(top_left_y, min(top_left_y + grid_size, height)):
            for j in range(top_left_x, min(top_left_x + grid_size, width)):
                if not np.array_equal(image[i, j], [0, 0, 0]):  # If the pixel is not black
                    non_black_pixel_count += 1

        # if more than half of the pixels in the grid are not black
        if non_black_pixel_count > (grid_size * grid_size) // 2:

            grids_removed.add(grid_index)

            for i in range(top_left_y, min(top_left_y + grid_size, height)):
                for j in range(top_left_x, min(top_left_x + grid_size, width)):
                    image[i, j] = [0, 0, 0]  #TODO: remove grid

            removed_grids.append((top_left_x, top_left_y))

    output_image_path = os.path.join(output_dir, f'modified_output_{iteration:02d}.png')
    cv2.imwrite(output_image_path, image)

    print(f'Iteration {iteration}: Grids removed {removed_grids}')
    print(f'Modified image saved to {output_image_path}')










