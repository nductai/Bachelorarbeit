import wandb
import json
from pathlib import Path

wandb.init(project="pose-estimation", name="json-plot")

# ------------------PATH SETUP------------------
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parent.parent  # -> Bachelorarbeit

file_path = (
    REPO_ROOT
    / "mmpose"
    / "work_dirs"
    / "td-hm_hrnet"
    / "20250606_203408"
    / "vis_data"
    / "20250606_203408.json"
)

with open(file_path, "r") as file:
    lines = file.readlines()

for line in lines:
    record = json.loads(line.strip())
    if "loss" in record:
        # training metrics
        wandb.log({
            "loss": record["loss"],
            "accuracy": record["acc_pose"],
            "learning_rate": record["lr"],
            "epoch": record["epoch"],
            "step": record["step"]
        })
    if "PCK" in record:
        # validation metrics
        wandb.log({
            "validation_PCK": record["PCK"],
            "validation_step": record["step"]
        })

