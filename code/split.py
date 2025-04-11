import os
import shutil
import json

def copy_images(base_script_dir, target_subfolder="Pose-Estimation-ToF", validation_folders=None, testing_folders=None):

    if validation_folders is None:
        validation_folders = set()
    if testing_folders is None:
        testing_folders = set()

    base_dir = os.path.join(base_script_dir, target_subfolder)

    training_dest = os.path.join(base_dir, "training", "images")
    validation_dest = os.path.join(base_dir, "validation")
    testing_dest = os.path.join(base_dir, "testing")
    os.makedirs(training_dest, exist_ok=True)
    os.makedirs(validation_dest, exist_ok=True)
    os.makedirs(testing_dest, exist_ok=True)

    # Traverse all directories
    for root, dirs, files in os.walk(base_dir):
        if "cropped" in root and "depth" in root:
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.isfile(file_path):
                    # Copy all images to training/images
                    shutil.copy(file_path, training_dest)

                    #TODO: remove block below because we dont need it

                    # Copy images to validation if the folder matches
                    if any(folder in root for folder in validation_folders):
                        shutil.copy(file_path, validation_dest)

                    # Copy images to testing if the folder matches
                    if any(folder in root for folder in testing_folders):
                        shutil.copy(file_path, testing_dest)

    print(f"All images have been copied to Training ({training_dest}), Validation ({validation_dest}), and Testing ({testing_dest})!")


def copy_json_to_folders(base_script_dir, target_subfolder="Pose-Estimation-ToF", training_folders=None, validation_folders=None, testing_folders=None):
    base_dir = os.path.join(base_script_dir, target_subfolder)

    training_dir = os.path.join(base_dir, "training", "training")
    validation_dir = os.path.join(base_dir, "training", "validation")
    testing_dir = os.path.join(base_dir, "training", "testing")
    os.makedirs(training_dir, exist_ok=True)
    os.makedirs(validation_dir, exist_ok=True)
    os.makedirs(testing_dir, exist_ok=True)

    if not training_folders or not validation_folders or not testing_folders:
        raise ValueError("All of 'training_folders', 'validation_folders', and 'testing_folders' must be provided!")

    for root, dirs, files in os.walk(base_dir):
        for folder_name in training_folders.union(validation_folders).union(testing_folders):
            if folder_name in root and "keypoints" in root:
                for file in files:
                    file_path = os.path.join(root, file)
                    if os.path.isfile(file_path) and file.endswith(".json"):
                        if folder_name in training_folders:
                            dest_folder = training_dir
                        elif folder_name in validation_folders:
                            dest_folder = validation_dir
                        elif folder_name in testing_folders:
                            dest_folder = testing_dir
                        shutil.copy(file_path, dest_folder)

    print(f"All JSON files have been copied to respective folders!")


def combine_json_files(base_script_dir, target_subfolder="Pose-Estimation-ToF"):
    base_dir = os.path.join(base_script_dir, target_subfolder, "training")

    training_dir = os.path.join(base_dir, "training")
    validation_dir = os.path.join(base_dir, "validation")
    testing_dir = os.path.join(base_dir, "testing")

    train_output_file = os.path.join(base_dir, "train.json")
    val_output_file = os.path.join(base_dir, "val.json")
    test_output_file = os.path.join(base_dir, "test.json")

    def process_folder(input_folder, output_file):
        combined_data = []
        counter = 0
        for root, dirs, files in os.walk(input_folder):
            for file in files:
                if file.endswith(".json"):
                    file_path = os.path.join(root, file)
                    with open(file_path, "r") as f:
                        try:
                            json_data = json.load(f)
                            combined_data.append({
                                "image_file": json_data["image_file"],
                                "image_size": json_data["image_size"],
                                "bbox": json_data["bbox"],
                                "keypoints": json_data["keypoints"]
                            })
                            counter += 1
                        except json.JSONDecodeError as e:
                            print(f"Error reading {file_path}: {e}")

        with open(output_file, "w") as f:
            json.dump(combined_data, f, indent=4)
        print(f"Combined {counter} JSON files into {output_file}")

    process_folder(training_dir, train_output_file)
    process_folder(validation_dir, val_output_file)
    process_folder(testing_dir, test_output_file)


if __name__ == "__main__":
    base_script_dir = os.path.dirname(os.path.abspath(__file__))

    training_folders = {
        "recording-tof-20221024-0918",
        "recording-tof-20221024-0929",
        "recording-tof-20221024-0940",
        "recording-tof-20221024-0950",
        "recording-tof-20221024-1000",
        "recording-tof-20221024-1012",
    }

    testing_folders = {
        "recording-tof-20221024-1045",
        "recording-tof-20221024-1104",
    }

    validation_folders = {
        "recording-tof-20221024-1020",
        "recording-tof-20221024-1031",
    }

    copy_images(base_script_dir, validation_folders=validation_folders, testing_folders=testing_folders)
    copy_json_to_folders(base_script_dir, training_folders=training_folders, validation_folders=validation_folders, testing_folders=testing_folders)
    combine_json_files(base_script_dir)

