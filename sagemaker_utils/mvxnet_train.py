import os
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

# Base config files
S3_BASE_FILES = {
    "default_runtime.py": ("mmdetection3d/scripts/configs/_base_/default_runtime.py", "/opt/ml/code/_base_/"),
    "cosine.py": ("mmdetection3d/scripts/configs/_base_/schedules/cosine.py", "/opt/ml/code/_base_/schedules/")
}

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

# Download required _base_ files to the correct subdirectories
for file_name, (s3_path, local_dir) in S3_BASE_FILES.items():
    os.makedirs(local_dir, exist_ok=True)  # Ensure subdirectory exists
    local_path = os.path.join(local_dir, file_name)
    print(f"Downloading {s3_path} -> {local_path}")
    s3_client.download_file(S3_BUCKET, s3_path, local_path)

# Verify that all required files exist
required_files = [LOCAL_TRAIN_SCRIPT, LOCAL_CONFIG_PATH] + [os.path.join(dir_path, f) for f, (_, dir_path) in S3_BASE_FILES.items()]
for file_path in required_files:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ Missing required file: {file_path}")

print("✅ All required files are successfully downloaded!")

# Copy model weights to checkpoints
S3_MODEL_WEIGHTS_DIR = "/opt/ml/input/data/model_weights"
if os.path.exists(S3_MODEL_WEIGHTS_DIR):
    for file in os.listdir(S3_MODEL_WEIGHTS_DIR):
        shutil.copy(os.path.join(S3_MODEL_WEIGHTS_DIR, file), CHECKPOINTS_DIR)

import os
print("🔹 Checking files in /opt/ml/code/:")
print(os.listdir("/opt/ml/code/"))


CONFIG_PATH = "/opt/ml/code/configs/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class.py"

with open(CONFIG_PATH, "r") as f:
    config_content = f.read()

# Replace dataset path dynamically
config_content = config_content.replace("data/kitti/", "/opt/ml/input/data/kitti/")

with open(CONFIG_PATH, "w") as f:
    f.write(config_content)

print("✅ Updated dataset path in config file:", CONFIG_PATH)

print("🔹 Checking dataset in /opt/ml/input/data:")
print(os.listdir("/opt/ml/input/data/"))
print(os.listdir("/opt/ml/input/data/kitti"))

# Define training command
TRAINING_COMMAND = [
    "python3", LOCAL_TRAIN_SCRIPT,
    LOCAL_CONFIG_PATH,  # Use the verified config file path
    "--work-dir=/opt/ml/output/data",
    "--resume=auto",
    "--launcher=none"
]

print("Starting training...")
result = subprocess.run(TRAINING_COMMAND, text=True, capture_output=True)

# Print STDOUT and STDERR to get more details
print("STDOUT:", result.stdout)
print("STDERR:", result.stderr)

# Raise an error if the process failed
if result.returncode != 0:
    raise RuntimeError(f"Training failed with exit code {result.returncode}")

