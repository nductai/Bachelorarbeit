import numpy as np
import os

file_path = r"D:\TU\7_Semester\Bachelorarbeit\code\Pose-Estimation-ToF\testing\remove\005914\count\005914_0235.npy"

# Load the data
data = np.load(file_path)

# Save as CSV
np.savetxt("005914_0235.csv", data, delimiter=",", fmt="%.3f")

# Count number of removed grids (value == 1)
removed_count = np.sum(data == 1.0)

# Find the positions (row, col) of removed grids
removed_positions = np.argwhere(data == 1.0)

print(f"Total number of removed (1.000) values: {removed_count}")
print("Positions of removed values (row, col):")
print(removed_positions)

