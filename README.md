# Depth-Based Human Pose Estimation with Occlusion-Based Saliency Analysis

## Overview

This repository contains the implementation of a depth-based human pose estimation pipeline and an occlusion-based saliency analysis. The goal is to investigate whether a human pose estimation model can be trained directly on depth images using keypoint pseudo-labels generated from aligned RGB images.

The pipeline uses YOLOv8 for person detection and cropping, MMPose with HRNet-W48 to generate keypoint pseudo-labels from RGB images, and HRNet-W48 to train a pose estimation model on depth images.

In addition to model training and evaluation, an occlusion-based saliency analysis investigates how masking different regions of the depth image affects keypoint localization accuracy.

The pipeline covers:

1. Person detection and synchronized cropping of RGB and depth images using YOLOv8.
2. Keypoint pseudo-label generation from RGB images using MMPose.
3. Dataset preparation and splitting into training, validation, and test sets.
4. Training an HRNet-W48 pose estimation model on depth images.
5. Model evaluation using Percentage of Correct Keypoints (PCK) and comparison with an RGB-based baseline applied to depth images.
6. Occlusion-based saliency analysis using random grid masking and keypoint accuracy heatmaps.

---

## Repository Structure

```text
├── code/
│   ├── main.py                   # Person detection and image cropping
│   ├── split.py                  # Dataset preparation and splitting
│   ├── plot_wandb.py             # Training metrics visualization
│   ├── compare.py                # RGB baseline and depth model comparison
│   ├── remove_grid.py            # Random grid occlusion
│   ├── calculate_accuracy.py     # Keypoint accuracy and saliency heatmaps
│   └── Pose-Estimation-ToF/      # Processed data, annotations, and experiments
│
└── mmpose/
    ├── prediction.py             # RGB-based keypoint pseudo-label generation
    ├── train.py                  # Depth-based pose model training
    ├── test.py                   # Model evaluation using PCK
    └── configs/                  # HRNet-W48 model configurations
```

---

## Workflow and Pipeline

### 1. Person Detection and Image Cropping

YOLOv8 is used to detect the person in each RGB image. The resulting bounding box is applied to the corresponding RGB and depth images so that both image streams use the same crop.

Run the script:

```bash
python code/main.py
```

**Input:** Synchronized RGB and ToF depth images.

**Output:** Cropped RGB and depth images saved in the following directories:

```text
code/Pose-Estimation-ToF/<recording>/cropped/color/
code/Pose-Estimation-ToF/<recording>/cropped/depth/
```

### 2. Keypoint Pseudo-Label Generation

A pre-trained MMPose model with HRNet-W48 is used to predict 17 COCO keypoints from the cropped RGB images. These keypoints represent the nose, eyes, ears, shoulders, elbows, wrists, hips, knees, and ankles.

The predicted keypoints are transferred to the corresponding depth images and used as pseudo-labels for training the depth-based pose estimation model. They are automatically generated and are not manually annotated depth ground truth.

Run the script:

```bash
python mmpose/prediction.py
```

The script generates keypoint annotations in COCO/TinyCOCO-compatible JSON format.

Example annotation:

```json
{
  "image_file": "000693.png",
  "image_size": [120, 345],
  "bbox": [0.0, 0.0, 120.0, 345.0],
  "keypoints": [71.79, 32.68, 2, "..."]
}
```

The annotations contain the keypoint coordinates and their visibility values.

**Output directories:**

- Keypoint annotations: `code/Pose-Estimation-ToF/<recording>/keypoints/`
- Pose visualizations: `code/Pose-Estimation-ToF/<recording>/visualized/`

### 3. Dataset Preparation

The depth images and their corresponding keypoint pseudo-labels are split into training, validation, and test sets using a 60:20:20 ratio.

Run the script:

```bash
python code/split.py
```

The script copies the depth images and combines their annotations into JSON files for the three dataset partitions.

**Output structure:**

```text
code/Pose-Estimation-ToF/
└── training/
    ├── images/
    ├── train.json
    ├── val.json
    └── test.json
```

The training set is used to train the model, the validation set is used during training to monitor performance, and the test set is used for the final evaluation.

### 4. Training the Depth-Based Pose Estimation Model

The project uses a top-down HRNet-W48 model with a heatmap-based keypoint representation. The model is trained on depth images using pseudo-labels generated from the aligned RGB images.

Run the training script:

```bash
python mmpose/train.py mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py
```

The training configuration includes the dataset settings, model architecture, optimizer, and evaluation settings.

**Output:** Model checkpoints, training logs, and evaluation results are saved in:

```text
mmpose/work_dirs/td-hm_hrnet/
```

Example checkpoint names:

```text
best_PCK_epoch_*.pth
```

### 5. Training Metrics Visualization

Weights & Biases (W&B) is used to track training metrics and visualize the training process, including loss curves and PCK-related metrics where logged.

Run the visualization script:

```bash
python code/plot_wandb.py
```

The generated plots help monitor model convergence and training performance.

### 6. Model Evaluation and Comparison

#### 6.1. PCK Evaluation

The trained depth-based model is evaluated using Percentage of Correct Keypoints (PCK). This metric measures the proportion of keypoints whose localization errors fall below a specified threshold relative to the reference keypoints.

From the MMPose directory, run:

```bash
cd mmpose

python test.py \
    work_dirs/td-hm_hrnet/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py \
    work_dirs/td-hm_hrnet/best_PCK_epoch_*.pth \
    --out pck_results.json
```

The evaluation results are saved to `pck_results.json`.

Replace the checkpoint wildcard with the specific checkpoint filename if the script does not support wildcard expansion.

#### 6.2. RGB Baseline and Depth Model Comparison

The RGB-based pose estimation model is applied directly to depth images and compared with the model trained on depth images.

Run the comparison script:

```bash
python code/compare.py
```

This comparison investigates the difference between applying an RGB-trained model directly to depth data and training a model specifically on depth images.

### 7. Occlusion-Based Saliency Analysis

The occlusion-based saliency analysis investigates which spatial regions of a depth image influence the model's keypoint predictions. It does this by masking image regions and measuring the resulting changes in keypoint localization accuracy.

#### 7.1. Random Grid Masking (`remove_grid.py`)

The depth image is divided into a grid of patches. Random subsets of these patches are masked, and the modified images are passed to the depth-based pose estimation model.

The process is repeated over 1,000 masking iterations using different masking patterns. The resulting predictions are used to analyze how the removal of information from different image regions affects keypoint localization.

#### 7.2. Depth-Based Pose Prediction on Occluded Images (`prediction_depth.py`)

After generating the occluded images, the depth-based pose estimation model is run again to predict the 17 COCO keypoints for each occluded image at every masking threshold.

The script `mmpose/prediction_depth.py` loads the trained HRNet-W48 depth model using its configuration file and checkpoint. It processes the occluded images generated in the previous step for all masking thresholds, from `threshold_0_0` to `threshold_1_0`, and saves the predicted keypoints as JSON files. Pose visualizations are also generated to inspect the predictions.

The predictions and visualizations are saved in separate directories for each image and masking threshold. For example, for the image `005914`, the output structure is:

```text
code/Pose-Estimation-ToF/testing/remove/005914/
├── threshold_0_0/
│   └── images/
│       ├── keypoints/
│       └── visualized/
├── threshold_0_1/
│   └── images/
│       ├── keypoints/
│       └── visualized/
├── ...
└── threshold_1_0/
    └── images/
        ├── keypoints/
        └── visualized/
```

The `keypoints/` directory contains the predicted keypoint annotations in JSON format, while the `visualized/` directory contains the corresponding pose visualizations. This structure is repeated for each masking threshold, allowing the predictions to be compared under different levels of occlusion.

These predictions are then used in the next step to calculate the keypoint localization accuracy and generate the occlusion-based saliency heatmaps.

#### 7.3. Keypoint Accuracy and Saliency Heatmaps (`calculate_accuracy.py`)

The predicted keypoints from the masked images are compared with their reference positions using the L2 localization error.

The continuous keypoint accuracy score is defined as:

$$
\operatorname{Accuracy}(e_{L2}, r) =
\begin{cases}
1 - \dfrac{e_{L2}}{r}, & e_{L2} < r, \\[6pt]
0, & e_{L2} \geq r \text{ or } e_{L2} \text{ is NaN}.
\end{cases}
$$

where:

- $e_{L2}$ is the Euclidean distance between the predicted keypoint and its reference position.
- $r$ is the localization error threshold. In the occlusion analysis, $r = 20$ pixels.

The score decreases linearly as the localization error increases. For example, an error of 5 pixels gives a score of 0.75, while an error of 20 pixels or more gives a score of 0.

The accuracy scores from the different masking iterations are accumulated to generate spatial heatmaps for individual keypoints. These heatmaps show how sensitive each keypoint prediction is to the removal of information from different image regions.

The analysis measures the model's sensitivity to occlusion; it does not directly reveal the model's internal reasoning.

---

## Summary

This repository implements a pipeline for depth-based human pose estimation using RGB-derived keypoint pseudo-labels. The pipeline includes synchronized RGB and depth preprocessing, pseudo-label generation, depth model training, PCK evaluation, and comparison with an RGB-based baseline.

The occlusion-based saliency analysis complements the model evaluation by investigating how masking different spatial regions affects keypoint localization accuracy. Together, these experiments examine the feasibility of depth-based pose estimation and the model's sensitivity to spatial information in depth images.
