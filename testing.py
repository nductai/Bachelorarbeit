import numpy as np

file_path = r"D:\TU\7_Semester\Bachelorarbeit\code\Pose-Estimation-ToF\testing\remove\005914\threshold_0_4\masks\heatmap_0001.npy"
file_path2 = r"D:\TU\7_Semester\Bachelorarbeit\code\Pose-Estimation-ToF\testing\remove\005914\threshold_0_4\masks\keypoint_0\heatmap_accuracy_0001.npy"
#file_path3 = r"D:/TU/7_Semester/Bachelorarbeit/code/Pose-Estimation-ToF/testing/remove/005914\005914_avg_map.npy"

data = np.load(file_path)
data2 = np.load(file_path2)
#data3 = np.load(file_path3)

np.savetxt("heatmap_0001.csv", data, delimiter=",", fmt="%.3f")
np.savetxt("heatmap_accuracy_0001.csv", data2, delimiter=",", fmt="%.3f")
#np.savetxt("005914_avg_map.csv", data3, delimiter=",", fmt="%.3f")

print("CSV files saved successfully.")
