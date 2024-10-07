from mmcv.image import imread
from mmpose.apis import inference_topdown, init_model
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples
import numpy as np

model_cfg = r'D:\TU\7_Semester\Bachelorarbeit\mmpose\models\td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'
ckpt = r'D:\TU\7_Semester\Bachelorarbeit\mmpose\models\td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth'
device = 'cuda'

# init model
model = init_model(model_cfg, ckpt, device=device)
img_path = 'demo.png'

# COCO keypoint labels
coco_keypoint_labels = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
]

# inference on a single image
batch_results = inference_topdown(model, img_path)

# for i, result in enumerate(batch_results): IF THERE IS MANY PEOPLE IN IMAGE
result = batch_results[0]
keypoints = result.pred_instances.keypoints  # Get the keypoints
scores = result.pred_instances.keypoint_scores  # Get the confidence scores

print("Person:")
print("Keypoints:")
print(keypoints)
print("Confidence Scores:")
print(scores)


# ---------------FOR OUTPUTTING IMAGE--------------------------
# merge results as a single data sample
results = merge_data_samples(batch_results)

# build the visualizer
visualizer = VISUALIZERS.build(model.cfg.visualizer)

# set skeleton, colormap and joint connection rule
visualizer.set_dataset_meta(model.dataset_meta)

img = imread(img_path, channel_order='rgb')

# visualize the results
visualizer.add_datasample(
    'result',
    img,
    data_sample=results,
    show=True)
