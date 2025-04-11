import wandb
import json

wandb.init(project="pose-estimation", name="json-plot")

file_path = r"D:\TU\7_Semester\Bachelorarbeit\mmpose\work_dirs\td-hm_hrnet\20241126_094436\vis_data\20241126_094436.json"

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


