from sagemaker.pytorch import PyTorch

MLFLOW_TRACKING_URI = "http://ec2-16-171-110-153.eu-north-1.compute.amazonaws.com:5000/"


# Define S3 paths
s3_processed_data = "s3://dd-sample-bucket-north/mmdetection3d/datasets/kitti_processed/ImageSets/"
s3_model_weights = "s3://dd-sample-bucket-north/mmdetection3d/model_weights/"
s3_output_model = "s3://dd-sample-bucket-north/mmdetection3d/mmdetection3d-train-output/"

# AWS IAM Role & Custom Docker Image
image_uri = "910181886844.dkr.ecr.eu-north-1.amazonaws.com/mmdetection3d:with-sagemaker-training-mlflow"
iam_role = "arn:aws:iam::910181886844:role/AmazonSageMaker-ExecutionRole"

# ✅ Git configuration (auto-pulls latest scripts from your repo)
git_config = {
    'repo': 'https://github.com/SofiaChar/mmdetection3d.git',
    'branch': 'sagemaker-integration'
}

# ✅ Define training arguments
hyperparameters = {
    "config_file": "configs/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class.py",
    "work_dir": "/opt/ml/output/data",
    "resume": "auto",
    "launcher": "none",
    "mlflow_tracking_uri": MLFLOW_TRACKING_URI,
    "mlflow_experiment_name": "mmdet_training_experiment"
}


# ✅ SageMaker Estimator for Training
estimator = PyTorch(
    entry_point="sg_train.py",  # Directly run train.py
    git_config=git_config,
    source_dir=".",  # Make sure all scripts from the repo are available
    dependencies=["configs"],  # Ensure configs folder is included
    role=iam_role,
    instance_count=1,
    instance_type="ml.g4dn.2xlarge",
    image_uri=image_uri,
    output_path=s3_output_model,
    input_mode="File",
    volume_size_in_gb=300,
    hyperparameters=hyperparameters,  # ✅ Pass arguments dynamically
    environment = {
        "MLFLOW_TRACKING_URI": MLFLOW_TRACKING_URI
    }
)

# ✅ Start Training Job
estimator.fit({
    "kitti": s3_processed_data,
    "model_weights": s3_model_weights
})
