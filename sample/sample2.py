import os
import json
from mmcv.image import imread
from mmpose.apis import inference_topdown, init_model
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples

device = 'cuda'

# ------------------MODEL CONFIGURATION------------------
model_cfg = r'D:\TU\7_Semester\Bachelorarbeit\mmpose\work_dirs\td-hm_hrnet\td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'
ckpt = r'D:\TU\7_Semester\Bachelorarbeit\mmpose\work_dirs\td-hm_hrnet\best_PCK_epoch_86.pth'
model = init_model(model_cfg, ckpt, device=device)

# ------------------DIRECTORIES------------------
image_dir = r'D:\TU\7_Semester\Bachelorarbeit\sample\removed'
keypoints_dir = os.path.join(image_dir, 'keypoints')
visualized_dir = os.path.join(image_dir, 'visualized')
os.makedirs(keypoints_dir, exist_ok=True)
os.makedirs(visualized_dir, exist_ok=True)

# ------------------VISUALIZER------------------
visualizer = VISUALIZERS.build(model.cfg.visualizer)
visualizer.set_dataset_meta(model.dataset_meta)

# ------------------HELPERS------------------

def process_image(img_path):
    try:
        batch_results = inference_topdown(model, img_path)
        if not batch_results:
            print(f"No people detected in {img_path}")
            return

        result = batch_results[0]
        keypoints = result.pred_instances.keypoints
        scores = result.pred_instances.keypoint_scores

        save_results_to_json(img_path, keypoints, scores)
        save_visualized_image(img_path, batch_results)

    except Exception as e:
        print(f"Error processing {img_path}: {e}")

def save_results_to_json(img_path, keypoints, scores):
    img = imread(img_path)
    h, w = img.shape[:2]

    filename = os.path.basename(img_path)
    result = {
        'image_file': filename,
        'image_size': [w, h],
        'bbox': [],
        'keypoints': []
    }

    batch_results = inference_topdown(model, img_path)
    if batch_results and 'bboxes' in batch_results[0].pred_instances:
        x_min, y_min, x_max, y_max = batch_results[0].pred_instances.bboxes[0].tolist()
        result['bbox'] = [x_min, y_min, x_max - x_min, y_max - y_min]

    for i, kp in enumerate(keypoints):
        for j, coord in enumerate(kp):
            vis = 2 if scores[i][j] > 0.5 else (1 if scores[i][j] > 0 else 0)
            result['keypoints'].extend([float(coord[0]), float(coord[1]), vis])

    json_name = os.path.splitext(filename)[0] + '.json'
    json_path = os.path.join(keypoints_dir, json_name)
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=4)
    print(f"Saved: {json_path}")

def save_visualized_image(img_path, batch_results):
    merged_result = merge_data_samples(batch_results)
    img = imread(img_path, channel_order='rgb')

    filename = os.path.basename(img_path)
    output_path = os.path.join(visualized_dir, os.path.splitext(filename)[0] + '.png')

    visualizer.add_datasample(
        'result',
        img,
        data_sample=merged_result,
        draw_gt=False,
        show=False,
        out_file=output_path
    )
    print(f"Saved visualization: {output_path}")

# ------------------PROCESS ALL IMAGES IN DIRECTORY------------------
for file in sorted(os.listdir(image_dir)):
    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
        process_image(os.path.join(image_dir, file))
