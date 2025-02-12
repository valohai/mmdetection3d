from sagemaker.pytorch import PyTorch
import boto3
import os

# Define S3 paths
s3_scripts = "s3://dd-sample-bucket-north/mmdetection3d/scripts/"
s3_processed_data = "s3://dd-sample-bucket-north/mmdetection3d/datasets/kitti_processed/"
s3_model_weights = "s3://dd-sample-bucket-north/mmdetection3d/model_weights/"
s3_output_model = "s3://dd-sample-bucket-north/mmdetection3d/mmdetection3d-train-output/"

image_uri = "910181886844.dkr.ecr.eu-north-1.amazonaws.com/mmdetection3d:latest"
iam_role = "arn:aws:iam::910181886844:role/AmazonSageMaker-ExecutionRole"

# Create a local directory to store scripts
LOCAL_SCRIPTS_DIR = "/Users/sofiacharnota/Dev/valohai/mmdetection3d_fork/sagemaker_utils"
os.makedirs(LOCAL_SCRIPTS_DIR, exist_ok=True)

# # Download the training script from S3
# s3_client = boto3.client("s3")
# s3_client.download_file("dd-sample-bucket-north", "mmdetection3d/scripts/mvxnet_train.py", f"{LOCAL_SCRIPTS_DIR}/mvxnet_train.py")

# Define SageMaker Estimator
estimator = PyTorch(
    entry_point="sagemaker_utils/mvxnet_train.py",
    # source_dir="sagemaker_utils",
    role=iam_role,
    instance_count=1,
    instance_type="ml.g4dn.2xlarge",
    image_uri=image_uri,
    hyperparameters={"epochs": 10},
    output_path=s3_output_model,
    input_mode="File",  # SageMaker automatically downloads S3 data to /opt/ml/input/data/
    volume_size_in_gb=300
)

# Start Training Job
estimator.fit({
    "dataset": s3_processed_data,
})
