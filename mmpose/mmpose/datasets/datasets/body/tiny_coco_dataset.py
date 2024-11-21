# Copyright (c) OpenMMLab. All rights reserved.
import json
import os.path as osp
from typing import Callable, List, Optional, Sequence, Union

import numpy as np
from mmengine.utils import check_file_exist

from mmpose.registry import DATASETS
from mmpose.datasets.datasets.base import BaseCocoStyleDataset


@DATASETS.register_module()
class TinyCocoDataset(BaseCocoStyleDataset):
    METAINFO: dict = dict(from_file='configs/_base_/datasets/coco.py')

    def _load_annotations(self) -> List[dict]:
        """Load data from annotations in MPII format."""

        check_file_exist(self.ann_file)
        with open(self.ann_file) as anno_file:
            anns = json.load(anno_file)

        data_list = []
        ann_id = 0

        for idx, ann in enumerate(anns):
            img_h, img_w = ann['image_size']

            # get bbox in shape [1, 4], formatted as xywh
            x, y, w, h = ann['bbox']
            x1 = np.clip(x, 0, img_w - 1)
            y1 = np.clip(y, 0, img_h - 1)
            x2 = np.clip(x + w, 0, img_w - 1)
            y2 = np.clip(y + h, 0, img_h - 1)

            bbox = np.array([x1, y1, x2, y2], dtype=np.float32).reshape(1, 4)

            # load keypoints in shape [1, K, 2] and keypoints_visible in [1, K]
            joints_3d = np.array(ann['keypoints']).reshape(1, -1, 3)
            num_joints = joints_3d.shape[1]
            keypoints = np.zeros((1, num_joints, 2), dtype=np.float32)
            keypoints[:, :, :2] = joints_3d[:, :, :2]
            keypoints_visible = np.minimum(1, joints_3d[:, :, 2:3])
            keypoints_visible = keypoints_visible.reshape(1, -1)

            data_info = {
                'id': ann_id,
                'img_id': int(ann['image_file'].split('.')[0]),
                'img_path': osp.join(self.data_prefix['img'], ann['image_file']),
                'bbox': bbox,
                'bbox_score': np.ones(1, dtype=np.float32),
                'keypoints': keypoints,
                'keypoints_visible': keypoints_visible,
            }

            data_list.append(data_info)
            ann_id += 1

        return data_list, None