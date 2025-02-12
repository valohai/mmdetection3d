import os
import shutil
import subprocess

import os
print("🔹 Checking files in /opt/ml/code/:")
print(os.listdir("/opt/ml/code/"))

# Ensure boto3 is installed
try:
    import boto3
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "boto3"])
    import boto3  # Retry import

# S3 bucket details
S3_BUCKET = "dd-sample-bucket-north"
S3_TRAIN_SCRIPT_PATH = "mmdetection3d/scripts/tools/train.py"
S3_CONFIG_PATH = "mmdetection3d/scripts/configs/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class.py"

# Local paths inside SageMaker container
LOCAL_TRAIN_SCRIPT = "/opt/ml/code/tools/train.py"
LOCAL_CONFIG_PATH = "/opt/ml/code/configs/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class.py"

# Other directories
CHECKPOINTS_DIR = "/opt/ml/checkpoints/"
OUTPUT_DIR = "/opt/ml/output/data/"

# Ensure directories exist
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOCAL_TRAIN_SCRIPT), exist_ok=True)  # Ensure /opt/ml/code/tools/
os.makedirs(os.path.dirname(LOCAL_CONFIG_PATH), exist_ok=True)  # Ensure /opt/ml/code/configs/

# Initialize S3 client
s3_client = boto3.client("s3")

# Download train.py from S3
print(f"Downloading training script from S3: s3://{S3_BUCKET}/{S3_TRAIN_SCRIPT_PATH}")
s3_client.download_file(S3_BUCKET, S3_TRAIN_SCRIPT_PATH, LOCAL_TRAIN_SCRIPT)

# Download config file from S3
print(f"Downloading config file from S3: s3://{S3_BUCKET}/{S3_CONFIG_PATH}")
s3_client.download_file(S3_BUCKET, S3_CONFIG_PATH, LOCAL_CONFIG_PATH)

# Verify that both files exist
if not os.path.exists(LOCAL_TRAIN_SCRIPT):
    raise FileNotFoundError(f"❌ Train script missing: {LOCAL_TRAIN_SCRIPT}")

if not os.path.exists(LOCAL_CONFIG_PATH):
    raise FileNotFoundError(f"❌ Config file missing: {LOCAL_CONFIG_PATH}")

print("✅ Both train.py and config file are successfully downloaded!")

# Copy model weights to checkpoints
S3_MODEL_WEIGHTS_DIR = "/opt/ml/input/data/model_weights"
if os.path.exists(S3_MODEL_WEIGHTS_DIR):
    for file in os.listdir(S3_MODEL_WEIGHTS_DIR):
        shutil.copy(os.path.join(S3_MODEL_WEIGHTS_DIR, file), CHECKPOINTS_DIR)

# Define training command
TRAINING_COMMAND = [
    "python3", LOCAL_TRAIN_SCRIPT,
    LOCAL_CONFIG_PATH,  # Use the verified config file path
    "--work-dir=/opt/ml/output/data/",
    "--resume=auto",
    "--launcher=none"
]

print("Starting training...")
subprocess.run(TRAINING_COMMAND, check=True)
