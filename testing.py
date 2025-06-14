import numpy as np

file_path = r"D:\TU\7_Semester\Bachelorarbeit\code\Pose-Estimation-ToF\testing\remove\005914\count\005914_final.npy"
file_path2 = r"D:/TU/7_Semester/Bachelorarbeit/code/Pose-Estimation-ToF/testing/remove/005914\heat_map\005914_heatmap_final.npy"
file_path3 = r"D:/TU/7_Semester/Bachelorarbeit/code/Pose-Estimation-ToF/testing/remove/005914\005914_avg_map.npy"

data = np.load(file_path)
data2 = np.load(file_path2)
data3 = np.load(file_path3)

np.savetxt("005914_final.csv", data, delimiter=",", fmt="%.3f")
np.savetxt("005914_heatmap_final.csv", data2, delimiter=",", fmt="%.3f")
np.savetxt("005914_avg_map.csv", data3, delimiter=",", fmt="%.3f")

print("CSV files saved successfully.")
