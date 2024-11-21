import os
import shutil
import json

def copy_images_to_training(base_script_dir, target_subfolder="Pose-Estimation-ToF"):
    base_dir = os.path.join(base_script_dir, target_subfolder)

    destination_dir = os.path.join(base_dir, "training", "images")

    os.makedirs(destination_dir, exist_ok=True)

    for root, dirs, files in os.walk(base_dir):
        if "cropped" in root and "depth" in root:
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.isfile(file_path):
                    shutil.copy(file_path, destination_dir)

    print(f"All images have been copied to {destination_dir}!")

def copy_json_to_folders(base_script_dir, target_subfolder="Pose-Estimation-ToF"):
    base_dir = os.path.join(base_script_dir, target_subfolder)

    # Define training and validation folder destinations
    training_dir = os.path.join(base_dir, "training", "training")
    validation_dir = os.path.join(base_dir, "training", "validation")
    os.makedirs(training_dir, exist_ok=True)
    os.makedirs(validation_dir, exist_ok=True)

    training_folders = {
        "recording-tof-20221024-0918",
        "recording-tof-20221024-0929",
        "recording-tof-20221024-0940",
        "recording-tof-20221024-0950",
        "recording-tof-20221024-1000",
        "recording-tof-20221024-1012",
        "recording-tof-20221024-1020",
        "recording-tof-20221024-1031",
    }
    validation_folders = {
        "recording-tof-20221024-1045",
        "recording-tof-20221024-1104",
    }

    for root, dirs, files in os.walk(base_dir):
        for folder_name in training_folders.union(validation_folders):
            if folder_name in root and "keypoints" in root:
                for file in files:
                    file_path = os.path.join(root, file)
                    if os.path.isfile(file_path) and file.endswith(".json"):
                        if folder_name in training_folders:
                            dest_folder = training_dir
                        elif folder_name in validation_folders:
                            dest_folder = validation_dir
                        shutil.copy(file_path, dest_folder)

    print(f"All JSON files have been copied to respective folders!")

def combine_json_files(source_dir, output_file):
    combined_data = []

    # Traverse the source directory
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                with open(file_path, "r") as f:
                    json_data = json.load(f)

                # Reformat the JSON data
                entry = {
                    "image_file": json_data.get("image_file", ""),
                    "image_size": json_data.get("image_size", [0, 0]),
                    "bbox": json_data.get("bbox", [0, 0, 0, 0]),
                    "keypoints": json_data.get("keypoints", [0] * 51)
                }
                combined_data.append(entry)

    # Write the combined data to the output file
    with open(output_file, "w") as f:
        json.dump(combined_data, f, indent=4)

    print(f"Combined JSON saved to {output_file}")

def combine_json_files(base_script_dir, target_subfolder="Pose-Estimation-ToF"):
    base_dir = os.path.join(base_script_dir, target_subfolder, "training")

    # Paths for JSON files
    training_dir = os.path.join(base_dir, "training")
    validation_dir = os.path.join(base_dir, "validation")
    train_output_file = os.path.join(base_dir, "train.json")
    val_output_file = os.path.join(base_dir, "val.json")

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
                            # Transform the JSON structure to the required format
                            combined_data.append({
                                "image_file": json_data["image_file"],
                                "image_size": json_data["image_size"],
                                "bbox": json_data["bbox"],
                                "keypoints": json_data["keypoints"]
                            })
                            counter += 1
                        except json.JSONDecodeError as e:
                            print(f"Error reading {file_path}: {e}")
        # Save combined JSON
        with open(output_file, "w") as f:
            json.dump(combined_data, f, indent=4)
        print(f"Combined {counter} JSON files into {output_file}")

    # Process training and validation folders
    process_folder(training_dir, train_output_file)
    process_folder(validation_dir, val_output_file)

if __name__ == "__main__":
    base_script_dir = os.path.dirname(os.path.abspath(__file__))
    #copy_images_to_training(base_script_dir)
    #copy_json_to_folders(base_script_dir)
    combine_json_files(base_script_dir)
