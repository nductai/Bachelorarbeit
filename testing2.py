import cv2
import matplotlib.pyplot as plt
import numpy as np

image_path = r"D:\TU\7_Semester\Bachelorarbeit\code\Pose-Estimation-ToF\testing\remove\005914\005914_0001.png"
csv_path = r"D:\TU\7_Semester\Bachelorarbeit\code\Pose-Estimation-ToF\testing\remove\005914\005914_0001_values.csv"

depth_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if depth_image is None:
    print("Failed to load image. Check the path.")
else:
    plt.imshow(depth_image, cmap='gray')
    plt.title("Depth Image Grayscale Visualization")
    plt.axis('off')
    plt.show()

    np.savetxt(csv_path, depth_image, fmt='%d', delimiter=',')
    print(f"Pixel values saved to {csv_path}")

    max_value = np.max(depth_image)
    print(f"Highest grayscale value in the image: {max_value}")
