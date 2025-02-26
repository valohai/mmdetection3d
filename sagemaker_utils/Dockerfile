# ✅ Use the existing image as a base
FROM valohai/mmdetection3d:latest

# ✅ Install sagemaker-training inside the container
RUN pip install --no-cache-dir sagemaker-training

RUN pip install --upgrade pip setuptools wheel
RUN pip install mlflow boto3 packaging

# ✅ Verify installation (optional)
RUN python3 -c "import sagemaker_training; print('✅ sagemaker-training installed successfully')"