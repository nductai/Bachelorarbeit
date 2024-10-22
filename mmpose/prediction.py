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


model = init_model(model_cfg, ckpt, device=device)

coco_keypoint_labels = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
]

base_dir = r'D:\TU\7_Semester\Bachelorarbeit\code\Pose-Estimation-ToF'

# Initialize visualizer
visualizer = VISUALIZERS.build(model.cfg.visualizer)
visualizer.set_dataset_meta(model.dataset_meta)


def clean_save_directories():
    for recording_folder in os.listdir(base_dir):
        recording_path = os.path.join(base_dir, recording_folder)
        if os.path.isdir(recording_path):
            keypoints_dir = os.path.join(recording_path, 'cropped/keypoints')
            visualized_dir = os.path.join(recording_path, 'cropped/visualized')

            # remove keypoints and visualized directories if they exist
            if os.path.exists(keypoints_dir):
                shutil.rmtree(keypoints_dir)
                print(f"Deleted directory: {keypoints_dir}")
            if os.path.exists(visualized_dir):
                shutil.rmtree(visualized_dir)
                print(f"Deleted directory: {visualized_dir}")


def process_folders(base_dir):
    for recording_folder in os.listdir(base_dir):
        recording_path = os.path.join(base_dir, recording_folder)
        if os.path.isdir(recording_path):
            color_data_path = os.path.join(recording_path, 'cropped/color')
            if os.path.exists(color_data_path):
                print(f"Processing folder: {recording_folder}")
                process_color_subfolders(color_data_path, recording_folder)

# process each subfolder inside the cropped/color folder
def process_color_subfolders(color_data_path, recording_folder):
    for subfolder in sorted(os.listdir(color_data_path)):
        subfolder_path = os.path.join(color_data_path, subfolder)
        if os.path.isdir(subfolder_path):
            print(f"Processing subfolder: {subfolder}")
            process_images(subfolder_path, recording_folder, subfolder)

#process each image in folder
def process_images(image_folder, recording_folder, subfolder):
    for img_file in sorted(os.listdir(image_folder)):
        if img_file.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(image_folder, img_file)
            keypoints, scores, batch_results = infer_keypoints(img_path)

            if keypoints is not None and scores is not None:
                # save keypoints and scores to JSON
                save_results_to_json(img_file, keypoints, scores, recording_folder, subfolder)

                # save visualized keypoints image
                save_visualized_image(img_path, batch_results, recording_folder, subfolder, img_file)


# run pose estimation here
def infer_keypoints(img_path):
    try:
        batch_results = inference_topdown(model, img_path)
        if len(batch_results) > 0:
            result = batch_results[0]
            keypoints = result.pred_instances.keypoints  # get keypoints
            scores = result.pred_instances.keypoint_scores  # get confidence scores

            return keypoints, scores, batch_results
        else:
            print(f"No people detected in {img_path}")
            return None, None, None
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None, None, None


# save keypoints and confidence scores to a JSON file
def save_results_to_json(image_name, keypoints, scores, recording_folder, subfolder):
    img_path = os.path.join(base_dir, recording_folder, 'cropped/color', subfolder, image_name)
    img = imread(img_path)
    image_height, image_width = img.shape[:2]


    results = {
        'image_file': image_name,
        'image_size': [image_width, image_height],
        'bbox': [],  # [x top left corner, y top right corner, width, height]
        'keypoints': []
    }

    batch_results = inference_topdown(model, img_path)
    if batch_results and 'bboxes' in batch_results[0].pred_instances:

        x_min, y_min, x_max, y_max = batch_results[0].pred_instances.bboxes[0].tolist()

        width = x_max - x_min
        height = y_max - y_min

        results['bbox'] = [x_min, y_min, width, height]

    for keypoint in keypoints:
        results['keypoints'].append(keypoint.tolist())

    json_file_name = f"{os.path.splitext(image_name)[0]}_keypoints.json"
    json_dir = os.path.join(base_dir, recording_folder, 'keypoints', subfolder)
    os.makedirs(json_dir, exist_ok=True)
    json_file_path = os.path.join(json_dir, json_file_name)

    with open(json_file_path, 'w') as json_file:
        json.dump(results, json_file, indent=4)

    # print(f"Saved keypoints to {json_file_path}")

# save the visualized image with keypoints
def save_visualized_image(img_path, batch_results, recording_folder, subfolder, img_file):
    results = merge_data_samples(batch_results)

    img = imread(img_path, channel_order='rgb')

    vis_dir = os.path.join(base_dir, recording_folder, 'visualized', subfolder)
    os.makedirs(vis_dir, exist_ok=True)
    output_image_path = os.path.join(vis_dir, f"{os.path.splitext(img_file)[0]}_visualized.png")

    visualizer.add_datasample(
        'result',
        img,
        data_sample=results,
        draw_gt=False,  # only the predicted output is drawn
        show=False,
        out_file=output_image_path
    )

    #print(f"Saved visualized keypoints to {output_image_path}")


clean_save_directories()

process_folders(base_dir)