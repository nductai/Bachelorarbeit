import os
import sys
from ultralytics import YOLO
import cv2

model = YOLO("yolov8s.pt")

color_data_path = "Pose-Estimation-ToF/recording-tof-20221024-0918/kinect/color/0/"
depth_data_path = "Pose-Estimation-ToF/recording-tof-20221024-0918/kinect/depth/0/"

save_dir = "Pose-Estimation-ToF/recording-tof-20221024-0918/cropped/"
os.makedirs(save_dir, exist_ok=True)

color_files = sorted(os.listdir(color_data_path))
depth_files = sorted(os.listdir(depth_data_path))

if len(color_files) != len(depth_files):
    print("Number of files and directories in color_files:", len(color_files))
    print("Number of files and directories in depth_files:", len(depth_files))
    raise ValueError("NUMBER OF FILES ARE NOT THE SAME!")

# for file in files:
#     file_path = os.path.join(color_data_path, file)
#
#     if file.endswith(('.jpg', '.jpeg', '.png')):
#         img = cv2.imread(file_path)
#         result = model(source=file_path, show=False, conf=0.4, save=True, save_crop=True, project=save_dir, name="0",
#                        exist_ok=True)

# -------------DIFFERENT VERSION OF CROPPING (WITH COORDINATE)------------------

# loop through files
for i in range(len(color_files)):
    color_file = color_files[i]
    depth_file = depth_files[i]

    color_file_path = os.path.join(color_data_path, color_file)
    depth_file_path = os.path.join(depth_data_path, depth_file)

    if color_file.endswith(('.jpg', '.jpeg', '.png')):
        color_img = cv2.imread(color_file_path)
        depth_img = cv2.imread(depth_file_path, cv2.IMREAD_UNCHANGED)  # maybe IMREAD_UNCHANGED because of depth images

        result = model(source=color_file_path, show=False, conf=0.4)

        # Loop through detected objects in the color image
        for j, det in enumerate(result[0].boxes.xyxy):
            print("Detect ", det)
            sys.exit()
            x1, y1, x2, y2 = map(int, det[:4])
            crop_depth_img = depth_img[y1:y2, x1:x2]

            crop_path = os.path.join(save_dir, f"depth_cropped_{i}_{j}.png")  # Save with index and detection number
            cv2.imwrite(crop_path, crop_depth_img)

