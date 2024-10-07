import os
import json
import shutil
from mmcv.image import imread
from mmpose.apis import inference_topdown, init_model
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples

model_cfg = r'D:\TU\7_Semester\Bachelorarbeit\mmpose\models\td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'
ckpt = r'D:\TU\7_Semester\Bachelorarbeit\mmpose\models\td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth'
device = 'cuda'

# Initialize the model
model = init_model(model_cfg, ckpt, device=device)

# Define COCO keypoint labels
coco_keypoint_labels = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
]

# Base directory
base_dir = r'D:\TU\7_Semester\Bachelorarbeit\code\Pose-Estimation-ToF'

# Initialize visualizer
visualizer = VISUALIZERS.build(model.cfg.visualizer)
visualizer.set_dataset_meta(model.dataset_meta)


def clean_save_directories():
    """Remove the `keypoints` and `visualized` directories if they exist."""
    for recording_folder in os.listdir(base_dir):
        recording_path = os.path.join(base_dir, recording_folder)
        if os.path.isdir(recording_path):
            keypoints_dir = os.path.join(recording_path, 'cropped/keypoints')
            visualized_dir = os.path.join(recording_path, 'cropped/visualized')

            # Remove keypoints and visualized directories if they exist
            if os.path.exists(keypoints_dir):
                shutil.rmtree(keypoints_dir)
                print(f"Deleted directory: {keypoints_dir}")
            if os.path.exists(visualized_dir):
                shutil.rmtree(visualized_dir)
                print(f"Deleted directory: {visualized_dir}")


def process_folders(base_dir):
    """ Process each recording folder and its cropped color subfolders """
    for recording_folder in os.listdir(base_dir):
        recording_path = os.path.join(base_dir, recording_folder)
        if os.path.isdir(recording_path):
            color_data_path = os.path.join(recording_path, 'cropped/color')
            if os.path.exists(color_data_path):
                print(f"Processing folder: {recording_folder}")
                process_color_subfolders(color_data_path, recording_folder)


def process_color_subfolders(color_data_path, recording_folder):
    """Process each subfolder inside the cropped/color folder."""
    for subfolder in sorted(os.listdir(color_data_path)):
        subfolder_path = os.path.join(color_data_path, subfolder)
        if os.path.isdir(subfolder_path):
            print(f"Processing subfolder: {subfolder}")
            process_images(subfolder_path, recording_folder, subfolder)


def process_images(image_folder, recording_folder, subfolder):
    """Process each image, perform inference, save keypoints, and save visualized output."""
    for img_file in sorted(os.listdir(image_folder)):
        if img_file.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(image_folder, img_file)
            keypoints, scores, batch_results = infer_keypoints(img_path)

            if keypoints is not None and scores is not None:
                # Save keypoints and scores to JSON
                save_results_to_json(img_file, keypoints, scores, recording_folder, subfolder)

                # Save the visualized keypoints image
                save_visualized_image(img_path, batch_results, recording_folder, subfolder, img_file)


def infer_keypoints(img_path):
    """Perform pose estimation inference on a single image."""
    try:
        # Inference on the image
        batch_results = inference_topdown(model, img_path)
        if len(batch_results) > 0:
            result = batch_results[0]
            keypoints = result.pred_instances.keypoints  # Get keypoints
            scores = result.pred_instances.keypoint_scores  # Get confidence scores

            return keypoints, scores, batch_results
        else:
            print(f"No people detected in {img_path}")
            return None, None, None
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None, None, None


def save_results_to_json(image_name, keypoints, scores, recording_folder, subfolder):
    """Save the keypoints and confidence scores to a JSON file."""
    results = []

    # Loop through keypoints and scores, format them
    for i in range(len(keypoints)):
        keypoint_data = {
            'keypoint_label': coco_keypoint_labels[i],
            'keypoint': keypoints[i].tolist(),
            'confidence_score': scores[i].tolist()
        }
        results.append(keypoint_data)

    # Define the JSON file path
    json_file_name = f"{os.path.splitext(image_name)[0]}_keypoints.json"
    json_dir = os.path.join(base_dir, recording_folder, 'cropped/keypoints', subfolder)
    os.makedirs(json_dir, exist_ok=True)
    json_file_path = os.path.join(json_dir, json_file_name)

    # Save the results to the JSON file
    with open(json_file_path, 'w') as json_file:
        json.dump(results, json_file, indent=4)

    #print(f"Saved keypoints to {json_file_path}")


def save_visualized_image(img_path, batch_results, recording_folder, subfolder, img_file):
    """Save the visualized image with keypoints to a new directory."""
    # Merge the results into a single data sample
    results = merge_data_samples(batch_results)

    # Load the image
    img = imread(img_path, channel_order='rgb')

    # Define the output directory and file path for the visualized image
    vis_dir = os.path.join(base_dir, recording_folder, 'cropped/visualized', subfolder)
    os.makedirs(vis_dir, exist_ok=True)
    output_image_path = os.path.join(vis_dir, f"{os.path.splitext(img_file)[0]}_visualized.png")

    # Visualize and save the results to the output image
    visualizer.add_datasample(
        'result',
        img,
        data_sample=results,
        draw_gt=False,  # Ensure only the predicted output is drawn
        show=False,     # Do not display the image interactively
        out_file=output_image_path  # Save the output image
    )

    #print(f"Saved visualized keypoints to {output_image_path}")


# Clean directories at the beginning of the run
clean_save_directories()

# Start processing folders
process_folders(base_dir)