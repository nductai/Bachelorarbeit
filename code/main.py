import os
import glob
import shutil
import cv2
from ultralytics import YOLO

model = YOLO("yolov8s.pt")
base_dir = "Pose-Estimation-ToF/"


def clean_save_directory(directory):
    files = glob.glob(os.path.join(directory, '*'))
    for file in files:
        try:
            if os.path.isfile(file):
                os.remove(file)  # remove file
            elif os.path.isdir(file):
                shutil.rmtree(file)  # remove directory and its contents
        except Exception as e:
            print(f"Error removing {file}: {e}")


def process_folders(base_dir):
    # loop through each recording folder
    for recording_folder in os.listdir(base_dir):
        recording_path = os.path.join(base_dir, recording_folder)
        if os.path.isdir(recording_path):  # Ensure it's a directory
            print(f"Processing folder: {recording_folder}")

            # paths of color and depth images
            color_data_path = os.path.join(recording_path, "kinect/color")
            depth_data_path = os.path.join(recording_path, "kinect/depth")

            # corresponding save directories for cropped color and depth images
            save_dir_color = os.path.join(recording_path, "cropped/color")
            save_dir_depth = os.path.join(recording_path, "cropped/depth")

            os.makedirs(save_dir_color, exist_ok=True)
            clean_save_directory(save_dir_color)
            os.makedirs(save_dir_depth, exist_ok=True)
            clean_save_directory(save_dir_depth)

            # process subfolders within the color and depth directories
            process_color_depth_subfolders(color_data_path, depth_data_path, save_dir_color, save_dir_depth)


def process_color_depth_subfolders(color_data_path, depth_data_path, save_dir_color, save_dir_depth):
    # get all subfolders
    color_subfolders = sorted(os.listdir(color_data_path))
    depth_subfolders = sorted(os.listdir(depth_data_path))

    if len(color_subfolders) != len(depth_subfolders):
        print(f"Mismatch in number of color and depth subfolders in {color_data_path}")
        return

    # loop through each subfolder
    for color_subfolder, depth_subfolder in zip(color_subfolders, depth_subfolders):
        color_subfolder_path = os.path.join(color_data_path, color_subfolder)
        depth_subfolder_path = os.path.join(depth_data_path, depth_subfolder)

        # save directories for cropped images
        save_subfolder_color = os.path.join(save_dir_color, color_subfolder)
        save_subfolder_depth = os.path.join(save_dir_depth, depth_subfolder)

        os.makedirs(save_subfolder_color, exist_ok=True)
        os.makedirs(save_subfolder_depth, exist_ok=True)

        # process data
        process_color_depth_pairs(color_subfolder_path, depth_subfolder_path, save_subfolder_color,
                                  save_subfolder_depth)


def process_color_depth_pairs(color_data_path, depth_data_path, save_dir_color, save_dir_depth):
    color_files = sorted(os.listdir(color_data_path))
    depth_files = sorted(os.listdir(depth_data_path))

    if len(color_files) != len(depth_files):
        print(f"Mismatch in number of color and depth files in {color_data_path}")
        return

    # loop through each color image and corresponding depth image
    for i in range(len(color_files)):
        color_file = color_files[i]
        depth_file = depth_files[i]

        color_file_path = os.path.join(color_data_path, color_file)
        depth_file_path = os.path.join(depth_data_path, depth_file)

        if color_file.endswith(('.jpg', '.jpeg', '.png')):
            color_img = cv2.imread(color_file_path)
            depth_img = cv2.imread(depth_file_path, cv2.IMREAD_UNCHANGED)  # Depth images may have special formats

            result = model(source=color_file_path, show=False, conf=0.4)

            # loop through detected objects in the color image
            for j, (box, conf, cls) in enumerate(zip(result[0].boxes.xyxy, result[0].boxes.conf, result[0].boxes.cls)):
                label = result[0].names[int(cls)]
                if label == "person":
                    # print(f"Label: {label}, Box: {box.tolist()}, Confidence: {conf.item()}")

                    x1, y1, x2, y2 = map(int, box[:4])
                    # print(f"x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}")

                    # crop the color and depth images using box coordinates
                    crop_color_img = color_img[y1:y2, x1:x2]
                    crop_depth_img = depth_img[y1:y2, x1:x2]

                    # save the cropped color image
                    crop_color_path = os.path.join(save_dir_color, f"{label}_color_cropped_{i}_{j}.png")
                    cv2.imwrite(crop_color_path, crop_color_img)

                    # save the cropped depth image
                    crop_depth_path = os.path.join(save_dir_depth, f"{label}_depth_cropped_{i}_{j}.png")
                    cv2.imwrite(crop_depth_path, crop_depth_img)

    # cropped_color_files = sorted(os.listdir(save_dir_color))
    # cropped_depth_files = sorted(os.listdir(save_dir_depth))
    # print(f"Number of cropped color files saved: {len(cropped_color_files)}")
    # print(f"Number of cropped depth files saved: {len(cropped_depth_files)}")


process_folders(base_dir)
