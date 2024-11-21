from mmengine.config import Config
from mmengine.runner import Runner

from mmpose.datasets.datasets.body.tiny_coco_dataset import TinyCocoDataset


if __name__ == '__main__':
    # Load the configuration file
    cfg = Config.fromfile(r'D:\TU\7_Semester\Bachelorarbeit\mmpose\configs\body_2d_keypoint\topdown_heatmap\coco\td-hm_hrnet-w48_8xb32-210e_coco-256x192.py')

    # Set preprocess configs to model
    cfg.model.setdefault('data_preprocessor', cfg.get('preprocess_cfg', {}))

    # Build the runner from config
    runner = Runner.from_cfg(cfg)

    # Start training
    runner.train()