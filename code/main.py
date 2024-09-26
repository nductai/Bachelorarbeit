import os
from ultralytics import YOLO
import cv2

model = YOLO("yolov8s.pt")

data_path = "Pose-Estimation-ToF/recording-tof-20221024-0918/kinect/color/0/"

save_dir = "Pose-Estimation-ToF/recording-tof-20221024-0918/cropped/color/"

os.makedirs(save_dir, exist_ok=True)

files = os.listdir(data_path)

for file in files:
    file_path = os.path.join(data_path, file)

    if file.endswith(('.jpg', '.jpeg', '.png')):
        img = cv2.imread(file_path)
        result = model(source=file_path, show=False, conf=0.4, save=True, save_crop=True, project=save_dir, name="0",
                       exist_ok=True)

# -------------DIFFERENT VERSION OF CROPPING------------------
# for file in files:
#     file_path = os.path.join(data_path, file)
#
#     if file.endswith(('.jpg', '.jpeg', '.png')):
#         img = cv2.imread(file_path)
#         result = model(source=file_path, show=False, conf=0.4)
#
#         for i, det in enumerate(result[0].boxes.xyxy):
#             x1, y1, x2, y2 = map(int, det[:4])
#             crop_img = img[y1:y2, x1:x2]
#
#             # save cropped image
#             crop_path = os.path.join(save_dir, file)
#             cv2.imwrite(crop_path, crop_img)
