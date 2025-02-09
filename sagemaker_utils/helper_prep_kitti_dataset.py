import sagemaker
from sagemaker.processing import ScriptProcessor
from sagemaker.s3 import S3Uploader

# Define S3 paths
s3_script_path = "s3://dd-sample-bucket-north/mmdetection3d/scripts/prep_kitti_dataset.py"
s3_output_data = "s3://dd-sample-bucket-north/mmdetection3d/datasets/kitti_processed/"

image_uri = "910181886844.dkr.ecr.eu-north-1.amazonaws.com/mmdetection3d:latest"
iam_role = "arn:aws:iam::910181886844:role/AmazonSageMaker-ExecutionRole"

processor = ScriptProcessor(
    role=iam_role,
    image_uri=image_uri,
    command=["python3"],
    instance_count=1,
    instance_type="ml.m5.xlarge",
    volume_size_in_gb=100
)

# Run processing job
processor.run(
    code=s3_script_path,
    outputs=[sagemaker.processing.ProcessingOutput(source="/opt/ml/processing/output", destination=s3_output_data)]
)
