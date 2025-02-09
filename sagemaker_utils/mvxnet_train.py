import os
import shutil
import subprocess
import boto3

# Paths inside the SageMaker container
S3_MODEL_WEIGHTS_DIR = "/opt/ml/input/data/model_weights"
CHECKPOINTS_DIR = "/opt/ml/checkpoints/"
OUTPUT_DIR = "/opt/ml/output/data/"
LOCAL_CONFIG_PATH = "/opt/ml/code/configs/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class.py"
LOCAL_TRAIN_SCRIPT = "/opt/ml/code/tools/train.py"

# Ensure directories exist
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Copy model weights to checkpoints
if os.path.exists(S3_MODEL_WEIGHTS_DIR):
    for file in os.listdir(S3_MODEL_WEIGHTS_DIR):
        shutil.copy(os.path.join(S3_MODEL_WEIGHTS_DIR, file), CHECKPOINTS_DIR)

# Verify that the config file and train script exist
if not os.path.exists(LOCAL_CONFIG_PATH):
    raise FileNotFoundError(f"❌ Config file missing: {LOCAL_CONFIG_PATH}")

if not os.path.exists(LOCAL_TRAIN_SCRIPT):
    raise FileNotFoundError(f"❌ Train script missing: {LOCAL_TRAIN_SCRIPT}")

# Define training command
TRAINING_COMMAND = [
    "python3", LOCAL_TRAIN_SCRIPT,
    LOCAL_CONFIG_PATH,  # Use the verified path
    "--work-dir=/opt/ml/output/data/",
    "--resume=auto",
    "--launcher=none"
]

print("Starting training...")
subprocess.run(TRAINING_COMMAND, check=True)
