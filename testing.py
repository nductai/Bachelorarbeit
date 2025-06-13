import numpy as np

file_path = r"D:\TU\7_Semester\Bachelorarbeit\code\Pose-Estimation-ToF\testing\remove\005914\count\005914_0055.npy"
file_path2 = r"D:/TU/7_Semester/Bachelorarbeit/code/Pose-Estimation-ToF/testing/remove/005914\heat_map\005914_heatmap_0055.npy"

data = np.load(file_path)
data2 = np.load(file_path2)
np.set_printoptions(precision=3, suppress=True, linewidth=120)

print(data)

print(data2)