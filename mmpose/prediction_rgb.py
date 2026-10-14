import os
import json
import shutil
from pathlib import Path

from mmcv.image import imread
from mmpose.apis import inference_topdown, init_model
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples

device = 'cuda'

# ------------------PATH SETUP------------------
# prediction_rgb.py is located at: Bachelorarbeit/mmpose/prediction_rgb.py
THIS_FILE = Path(__file__).resolve()
MMPOSE_DIR = THIS_FILE.parent
REPO_ROOT = MMPOSE_DIR.parent

# ------------------MODEL FOR COLOR IMAGES------------------
model_cfg = (
    MMPOSE_DIR
    / "models"
    / "td-hm_hrnet-w48_8xb32-210e_coco-256x192.py"
)

ckpt = (
    MMPOSE_DIR
    / "models"
    / "td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth"
)

model = init_model(str(model_cfg), str(ckpt), device=device)

# ------------------DIRECTORIES------------------
base_dir = (
    REPO_ROOT
    / "code"
    / "Pose-Estimation-ToF"
)

# initialize visualizer
visualizer = VISUALIZERS.build(model.cfg.visualizer)
visualizer.set_dataset_meta(model.dataset_meta)


def clean_save_directories():
    for recording_folder in os.listdir(base_dir):
        recording_path = base_dir / recording_folder

        if recording_path.is_dir():
            keypoints_dir = recording_path / 'keypoints'
            visualized_dir = recording_path / 'visualized'

            if keypoints_dir.exists():
                shutil.rmtree(keypoints_dir)
                print(f"Deleted directory: {keypoints_dir}")

            if visualized_dir.exists():
                shutil.rmtree(visualized_dir)
                print(f"Deleted directory: {visualized_dir}")


def process_folders(base_dir):
    for recording_folder in os.listdir(base_dir):
        recording_path = base_dir / recording_folder

        if recording_path.is_dir():
            color_data_path = recording_path / 'cropped' / 'color'

            if color_data_path.exists():
                print(f"Processing folder: {recording_folder}")
                process_color_subfolders(color_data_path, recording_folder)


# process each subfolder inside cropped/color
def process_color_subfolders(color_data_path, recording_folder):
    for subfolder in sorted(os.listdir(color_data_path)):
        subfolder_path = color_data_path / subfolder

        if subfolder_path.is_dir():
            print(f"Processing subfolder: {subfolder}")
            process_images(subfolder_path, recording_folder, subfolder)


# process each image in a folder
def process_images(image_folder, recording_folder, subfolder):
    for img_file in sorted(os.listdir(image_folder)):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = image_folder / img_file
            keypoints, scores, batch_results = infer_keypoints(img_path)

            if keypoints is not None and scores is not None:
                save_results_to_json(
                    img_file, keypoints, scores, recording_folder, subfolder
                )

                save_visualized_image(
                    img_path, batch_results, recording_folder, subfolder, img_file
                )


# run pose estimation
def infer_keypoints(img_path):
    try:
        batch_results = inference_topdown(model, str(img_path))

        if len(batch_results) > 0:
            result = batch_results[0]
            keypoints = result.pred_instances.keypoints
            scores = result.pred_instances.keypoint_scores

            return keypoints, scores, batch_results

        else:
            print(f"No people detected in {img_path}")
            return None, None, None

    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None, None, None


# save keypoints and confidence scores to JSON
def save_results_to_json(image_name, keypoints, scores, recording_folder, subfolder):
    img_path = (
        base_dir
        / recording_folder
        / 'cropped'
        / 'color'
        / subfolder
        / image_name
    )

    img = imread(str(img_path))
    image_height, image_width = img.shape[:2]

    results = {
        'image_file': image_name,
        'image_size': [image_width, image_height],
        'bbox': [],
        'keypoints': []
    }

    batch_results = inference_topdown(model, str(img_path))

    if batch_results and 'bboxes' in batch_results[0].pred_instances:
        x_min, y_min, x_max, y_max = (
            batch_results[0].pred_instances.bboxes[0].tolist()
        )

        width = x_max - x_min
        height = y_max - y_min

        results['bbox'] = [x_min, y_min, width, height]

    # Visibility:
    # 0: Not labeled
    # 1: Labeled but not visible
    # 2: Labeled and visible
    for i, keypoint in enumerate(keypoints):
        for j, coord in enumerate(keypoint):
            visibility = (
                2 if scores[i][j] > 0.5
                else (1 if scores[i][j] > 0 else 0)
            )

            results['keypoints'].extend([
                float(coord[0]),
                float(coord[1]),
                visibility
            ])

    json_file_name = f"{Path(image_name).stem}.json"

    json_dir = (
        base_dir
        / recording_folder
        / 'keypoints'
        / subfolder
    )
    json_dir.mkdir(parents=True, exist_ok=True)

    json_file_path = json_dir / json_file_name

    with open(json_file_path, 'w') as json_file:
        json.dump(results, json_file, indent=4)


# save the visualized image with keypoints
def save_visualized_image(img_path, batch_results, recording_folder, subfolder, img_file):
    results = merge_data_samples(batch_results)

    img = imread(str(img_path), channel_order='rgb')

    vis_dir = (
        base_dir
        / recording_folder
        / 'visualized'
        / subfolder
    )
    vis_dir.mkdir(parents=True, exist_ok=True)

    output_image_path = vis_dir / f"{Path(img_file).stem}.png"

    visualizer.add_datasample(
        'result',
        img,
        data_sample=results,
        draw_gt=False,
        show=False,
        out_file=str(output_image_path)
    )


clean_save_directories()
process_folders(base_dir)