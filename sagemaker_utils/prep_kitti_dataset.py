import os
import subprocess
from urllib.request import urlretrieve
import sys

# Ensure boto3 is installed
try:
    import boto3
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "boto3"])
    import boto3  # Retry import

# Define Paths
DATA_URLS = [
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip"
]

SPLIT_FILES = {
    "test.txt": "https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/test.txt",
    "train.txt": "https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/train.txt",
    "val.txt": "https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/val.txt",
    "trainval.txt": "https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/trainval.txt"
}

S3_BUCKET = "dd-sample-bucket-north"  # Change this to your S3 bucket
S3_OUTPUT_PREFIX = "mmdetection3d/datasets/kitti_processed"
S3_SCRIPT_PATH = "mmdetection3d/scripts/tools/create_data.py"

LOCAL_DOWNLOAD_DIR = "/opt/ml/processing/input"
LOCAL_UNPACK_DIR = "/opt/ml/processing/output/kitti"
LOCAL_SCRIPT_PATH = "/opt/ml/processing/code/create_data.py"
IMAGESETS_DIR = os.path.join(LOCAL_UNPACK_DIR, "ImageSets")

# Ensure directories exist
os.makedirs(LOCAL_DOWNLOAD_DIR, exist_ok=True)
os.makedirs(LOCAL_UNPACK_DIR, exist_ok=True)
os.makedirs(IMAGESETS_DIR, exist_ok=True)

# Step 1: Download KITTI dataset
print("Downloading KITTI dataset...")
for url in DATA_URLS:
    filename = os.path.join(LOCAL_DOWNLOAD_DIR, os.path.basename(url))
    print(f"Downloading {url} -> {filename}")
    urlretrieve(url, filename)

# Step 2: Unzip dataset
import zipfile

def extract_and_delete(zip_path, output_dir):
    """Extracts a zip file and deletes it after extraction."""
    print(f"Extracting {zip_path} to {output_dir}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    os.remove(zip_path)  # Free up space immediately

# Extract one file at a time and delete
for zip_file in os.listdir(LOCAL_DOWNLOAD_DIR):
    if zip_file.endswith(".zip"):
        zip_path = os.path.join(LOCAL_DOWNLOAD_DIR, zip_file)
        extract_and_delete(zip_path, LOCAL_UNPACK_DIR)

print("Unzipping completed!")

# Step 3: Download missing ImageSets split files
print("Downloading missing ImageSets split files...")
for filename, url in SPLIT_FILES.items():
    file_path = os.path.join(IMAGESETS_DIR, filename)

    if not os.path.exists(file_path):  # Only download if missing
        print(f"Downloading {url} -> {file_path}")
        urlretrieve(url, file_path)
    else:
        print(f"File already exists: {file_path}")

# Verify that all files exist before proceeding
missing_files = [f for f in SPLIT_FILES.keys() if not os.path.exists(os.path.join(IMAGESETS_DIR, f))]
if missing_files:
    raise FileNotFoundError(f"Missing required ImageSets files: {missing_files}")

# Step 4: Download `create_data.py` from S3
os.makedirs("/opt/ml/processing/code/", exist_ok=True)
s3_client = boto3.client("s3")
s3_client.download_file(S3_BUCKET, S3_SCRIPT_PATH, LOCAL_SCRIPT_PATH)

# Step 5: Run create_data.py
print("Running create_data.py...")
subprocess.run([
    "python", LOCAL_SCRIPT_PATH,
    "kitti", "--root-path", LOCAL_UNPACK_DIR,
    "--out-dir", LOCAL_UNPACK_DIR, "--extra-tag", "kitti"
], check=True)

# Step 6: Upload processed dataset **without tarring** (folder structure is preserved)
def upload_directory_to_s3(local_dir, bucket, s3_prefix):
    """Uploads an entire directory to S3 recursively."""
    for root, _, files in os.walk(local_dir):
        for filename in files:
            local_path = os.path.join(root, filename)
            relative_path = os.path.relpath(local_path, local_dir)  # Preserve folder structure
            s3_path = f"{s3_prefix}/{relative_path}"

            print(f"Uploading {local_path} -> s3://{bucket}/{s3_path}")
            s3_client.upload_file(local_path, bucket, s3_path)

print(f"Uploading processed dataset to S3: s3://{S3_BUCKET}/{S3_OUTPUT_PREFIX}/")
upload_directory_to_s3(LOCAL_UNPACK_DIR, S3_BUCKET, S3_OUTPUT_PREFIX)

print("Dataset preprocessing complete. All files are now in S3.")
