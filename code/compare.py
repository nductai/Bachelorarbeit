import os
import json
from pathlib import Path
from mmcv.image import imread
from mmpose.apis import inference_topdown, init_model
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples

device = 'cuda'

# ------------------PATH SETUP------------------
# script is inside the Bachelorarbeit repo (e.g. Bachelorarbeit/code/...)
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parent.parent  # -> Bachelorarbeit

# ------------------MODEL FOR COLOR IMAGES------------------
color_model_cfg = (
    REPO_ROOT
    / "mmpose"
    / "models"
    / "td-hm_hrnet-w48_8xb32-210e_coco-256x192.py"
)
color_ckpt = (
    REPO_ROOT
    / "mmpose"
    / "models"
    / "td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth"
)

# ------------------MODEL FOR DEPTH IMAGES------------------
depth_model_cfg = (
    REPO_ROOT
    / "mmpose"
    / "work_dirs"
    / "td-hm_hrnet"
    / "td-hm_hrnet-w48_8xb32-210e_coco-256x192.py"
)
depth_ckpt = (
    REPO_ROOT
    / "mmpose"
    / "work_dirs"
    / "td-hm_hrnet"
    / "best_PCK_epoch_86.pth"
)

color_model = init_model(str(color_model_cfg), str(color_ckpt), device=device)
depth_model = init_model(str(depth_model_cfg), str(depth_ckpt), device=device)

color_visualizer = VISUALIZERS.build(color_model.cfg.visualizer)
depth_visualizer = VISUALIZERS.build(depth_model.cfg.visualizer)

color_visualizer.set_dataset_meta(color_model.dataset_meta)
depth_visualizer.set_dataset_meta(depth_model.dataset_meta)

# ------------------ SINGLE IMAGE PATH ------------------
base_dir = REPO_ROOT / "code" / "Pose-Estimation-ToF"
img_file = '005914.png'
img_path = base_dir / 'compare' / img_file


def infer_keypoints(model, img_path):
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


def save_results_to_json_validation(image_name, keypoints, scores, recording_folder, subfolder, model):
    img_path = base_dir / recording_folder / image_name
    img = imread(str(img_path))
    image_height, image_width = img.shape[:2]

    results = {
        'image_file': image_name,
        'image_size': [image_width, image_height],
        'bbox': [],
        'keypoints': []
    }

    batch_results = inference_topdown(model, img)
    if batch_results and 'bboxes' in batch_results[0].pred_instances:
        x_min, y_min, x_max, y_max = batch_results[0].pred_instances.bboxes[0].tolist()
        width = x_max - x_min
        height = y_max - y_min
        results['bbox'] = [x_min, y_min, width, height]

    for i, keypoint in enumerate(keypoints):
        for j, coord in enumerate(keypoint):
            visibility = 2 if scores[i][j] > 0.5 else (1 if scores[i][j] > 0 else 0)
            results['keypoints'].extend([float(coord[0]), float(coord[1]), visibility])

    json_file_name = f"{os.path.splitext(image_name)[0]}.json"
    json_dir = base_dir / recording_folder / 'keypoints' / subfolder
    os.makedirs(str(json_dir), exist_ok=True)
    json_file_path = json_dir / json_file_name

    with open(json_file_path, 'w') as json_file:
        json.dump(results, json_file, indent=4)


def save_visualized_image(img_path, batch_results, recording_folder, subfolder, img_file, visualizer):
    results = merge_data_samples(batch_results)
    img = imread(str(img_path), channel_order='rgb')

    vis_dir = base_dir / recording_folder / 'visualized' / subfolder
    os.makedirs(str(vis_dir), exist_ok=True)
    output_image_path = vis_dir / f"{os.path.splitext(img_file)[0]}.png"

    visualizer.add_datasample(
        'result',
        img,
        data_sample=results,
        draw_gt=False,
        show=False,
        out_file=str(output_image_path)
    )


# ------------------ RUN ON SINGLE IMAGE ------------------

# Process with default (color) model
keypoints_color, scores_color, batch_results_color = infer_keypoints(color_model, img_path)
if keypoints_color is not None and scores_color is not None:
    save_results_to_json_validation(img_file, keypoints_color, scores_color, 'compare', 'keypoints_default', color_model)
    save_visualized_image(img_path, batch_results_color, 'compare', 'visualized_default', img_file, color_visualizer)

# Process with custom (depth) model
keypoints_depth, scores_depth, batch_results_depth = infer_keypoints(depth_model, img_path)
if keypoints_depth is not None and scores_depth is not None:
    save_results_to_json_validation(img_file, keypoints_depth, scores_depth, 'compare', 'keypoints_custom', depth_model)
    save_visualized_image(img_path, batch_results_depth, 'compare', 'visualized_custom', img_file, depth_visualizer)



