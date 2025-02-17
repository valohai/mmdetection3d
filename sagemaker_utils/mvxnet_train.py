import os
import sys
import shutil
import subprocess

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
S3_BASE_DIR = "mmdetection3d/scripts/configs/_base_/"

# Local paths inside SageMaker container
LOCAL_TRAIN_SCRIPT = "/opt/ml/code/tools/train.py"
LOCAL_CONFIG_PATH = "/opt/ml/code/configs/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class.py"
LOCAL_BASE_DIR = "/opt/ml/code/_base_/"

# Other directories
CHECKPOINTS_DIR = "/opt/ml/checkpoints/"
OUTPUT_DIR = "/opt/ml/output/data/"

# Ensure directories exist
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOCAL_TRAIN_SCRIPT), exist_ok=True)  # Ensure /opt/ml/code/tools/
os.makedirs(os.path.dirname(LOCAL_CONFIG_PATH), exist_ok=True)  # Ensure /opt/ml/code/configs/
os.makedirs(LOCAL_BASE_DIR, exist_ok=True)  # Ensure /opt/ml/code/_base_/

# Initialize S3 client
s3_client = boto3.client("s3")

# Download train.py from S3
print(f"Downloading training script from S3: s3://{S3_BUCKET}/{S3_TRAIN_SCRIPT_PATH}")
s3_client.download_file(S3_BUCKET, S3_TRAIN_SCRIPT_PATH, LOCAL_TRAIN_SCRIPT)

# Download config file from S3
print(f"Downloading config file from S3: s3://{S3_BUCKET}/{S3_CONFIG_PATH}")
s3_client.download_file(S3_BUCKET, S3_CONFIG_PATH, LOCAL_CONFIG_PATH)

# Download the entire _base_ directory from S3
print(f"Downloading _base_ config files from S3: s3://{S3_BUCKET}/{S3_BASE_DIR}")
response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=S3_BASE_DIR)
if 'Contents' in response:
    for obj in response['Contents']:
        file_key = obj['Key']
        file_name = file_key.split('/')[-1]
        local_file_path = os.path.join(LOCAL_BASE_DIR, file_name)
        print(f"Downloading {file_key} -> {local_file_path}")
        s3_client.download_file(S3_BUCKET, file_key, local_file_path)
else:
    print(f"No files found in S3 directory: {S3_BASE_DIR}")

# Verify that all required files exist
if not os.path.exists(LOCAL_TRAIN_SCRIPT):
    raise FileNotFoundError(f"❌ Train script missing: {LOCAL_TRAIN_SCRIPT}")
if not os.path.exists(LOCAL_CONFIG_PATH):
    raise FileNotFoundError(f"❌ Config file missing: {LOCAL_CONFIG_PATH}")
if not os.listdir(LOCAL_BASE_DIR):
    raise FileNotFoundError(f"❌ No files found in _base_ directory: {LOCAL_BASE_DIR}")

print("✅ All required files are successfully downloaded!")

# Copy model weights to checkpoints
S3_MODEL_WEIGHTS_DIR = "/opt/ml/input/data/model_weights"
if os.path.exists(S3_MODEL_WEIGHTS_DIR):
    for file in os.listdir(S3_MODEL_WEIGHTS_DIR):
        shutil.copy(os.path.join(S3_MODEL_WEIGHTS_DIR, file), CHECKPOINTS_DIR)

print("🔹 Checking files in /opt/ml/code/:")
print(os.listdir("/opt/ml/code/"))

# Define training command
TRAINING_COMMAND = [
    "python3", LOCAL_TRAIN_SCRIPT,
    LOCAL_CONFIG_PATH,  # Use the verified config file path
    "--work-dir=/opt/ml/output/data",
    "--resume=auto",
    "--launcher=none"
]

print("Starting training...")
subprocess.run(TRAINING_COMMAND, check=True, capture_output=True, text=True)
