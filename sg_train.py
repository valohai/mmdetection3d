# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import os.path as osp
import subprocess

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
from mmdet3d.utils import replace_ceph_backend

import os
import sys
import argparse
import logging
import datetime
from mmengine.config import Config
from mmengine.runner import Runner

print('***************THE CODE')
print(os.listdir('/opt/ml/code/'))

import os
os.system("pip install protobuf==3.8.0 --no-cache-dir")

try:
    import boto3
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "boto3"])
    import boto3  # Retry import

# Install pytorch_lightning if not already installed
try:
    import pytorch_lightning
except ModuleNotFoundError:
    subprocess.check_call(["pip", "install", "pytorch_lightning==1.5.10"])
    import pytorch_lightning  # Retry import

try:
    import mlflow
    import mlflow.sagemaker
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--force-reinstall", "packaging<21.0"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "mlflow==1.24.0"])
    import mlflow
    import mlflow.sagemaker
    if mlflow.active_run():
        mlflow.end_run()

# SageMaker MLflow tracking URI
MLFLOW_TRACKING_URI = "http://ec2-16-171-110-153.eu-north-1.compute.amazonaws.com:5000/"

# Set MLflow tracking server
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Start an MLflow experiment
mlflow.set_experiment("mmdet_training_experiment")

# Auto-log all parameters and metrics
mlflow.pytorch.autolog()


def parse_args():
    """Parse command-line arguments, supporting SageMaker integration."""
    parser = argparse.ArgumentParser(description="Train a 3D detector")

    # Standard arguments
    parser.add_argument("--config_file", help="Train config file path")
    parser.add_argument("--work_dir", help="The dir to save logs and models")
    parser.add_argument("--amp", action="store_true", help="Enable AMP training")
    parser.add_argument("--sync_bn", choices=["none", "torch", "mmcv"], default="none",
                        help="Convert BatchNorm layers to SyncBatchNorm")
    parser.add_argument("--auto-scale-lr", action="store_true",
                        help="Enable automatic LR scaling")
    parser.add_argument("--resume", nargs="?", type=str, const="auto",
                        help="Resume training from a checkpoint")
    parser.add_argument("--ceph", action="store_true", help="Use Ceph as storage backend")
    parser.add_argument("--cfg_options", nargs="+", action=DictAction,
                        help="Override config settings")
    parser.add_argument("--launcher", choices=["none", "pytorch", "slurm", "mpi"],
                        default="none", help="Job launcher")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--mlflow_experiment_name", type=str, required=False, help="MLflow experiment name")
    parser.add_argument("--mlflow_tracking_uri", type=str, required=False, help="MLflow tracking server URI")

    # SageMaker-specific parameters
    parser.add_argument("--s3-bucket", type=str, default="dd-sample-bucket-north",
                        help="S3 bucket for model and configs")
    parser.add_argument("--s3_config_dir", type=str, default="mmdetection3d/scripts",
                        help="S3 directory where config files are stored")

    args = parser.parse_args()

    # Set local rank in environment (needed for distributed training)
    os.environ["LOCAL_RANK"] = str(args.local_rank)

    return args


def ensure_sagemaker_setup(args):
    """Detects and configures SageMaker environment."""
    is_sagemaker = "SM_TRAINING_ENV" in os.environ

    if is_sagemaker:
        print("✅ Running inside SageMaker. Auto-configuring paths...")

        # Load SageMaker input directories
        sm_input_dir = os.getenv("SM_INPUT_DIR", "/opt/ml/input")
        sm_output_dir = os.getenv("SM_OUTPUT_DIR", "/opt/ml/output")
        sm_data_dir = os.getenv("SM_CHANNEL_KITTI", "/opt/ml/input/data/kitti")

        # Adjust work directory
        args.work_dir = os.path.join(sm_output_dir, "data")
        print("/opt/ml/input/", os.listdir('/opt/ml/input/'))
        print("/opt/ml/input/data", os.listdir('/opt/ml/input/data'))
        print("/opt/ml/input/data/kitti", os.listdir('/opt/ml/input/data/kitti'))
        # Ensure dataset paths are correct
        args.config_file = args.config_file.replace("data/kitti/", sm_data_dir + "/")


        # Ensure necessary directories exist
        os.makedirs(args.work_dir, exist_ok=True)

    return is_sagemaker


def download_s3_config_files(args):
    """Downloads missing config files from S3 if needed."""
    s3_client = boto3.client("s3")
    local_config_path = f"/opt/ml/code/{args.config_file}"

    # Ensure directory exists
    os.makedirs(os.path.dirname(local_config_path), exist_ok=True)

    # Download config file if missing
    if not os.path.exists(local_config_path):
        s3_config_path = f"{args.s3_config_dir}/{args.config_file}"
        print(f"Downloading config file from S3: s3://{args.s3_bucket}/{s3_config_path}")
        s3_client.download_file(args.s3_bucket, s3_config_path, local_config_path)

    return local_config_path


def main():
    args = parse_args()

    # ✅ Ensure SageMaker setup
    is_sagemaker = ensure_sagemaker_setup(args)

    # ✅ Download missing config files from S3 if running on SageMaker
    local_config_path = download_s3_config_files(args)

    print('os.listdir code', os.listdir('/opt/ml/code/'))

    print('os.listdir code/configs', os.listdir('/opt/ml/code/configs/'))

    print('os.listdir code/configs/_base_', os.listdir('/opt/ml/code/configs/_base_/'))
    # ✅ Load training config
    cfg = Config.fromfile('/opt/ml/code/configs/mvxnet/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class.py')

    if args.ceph:
        cfg = replace_ceph_backend(cfg)

    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # ✅ Adjust work directory based on environment
    if args.work_dir:
        cfg.work_dir = args.work_dir
    elif cfg.get("work_dir") is None:
        cfg.work_dir = osp.join("./work_dirs", osp.splitext(osp.basename(args.config_file))[0])

    # ✅ Handle AMP training
    if args.amp:
        if cfg.optim_wrapper.type == "OptimWrapper":
            cfg.optim_wrapper.type = "AmpOptimWrapper"
            cfg.optim_wrapper.loss_scale = "dynamic"
        else:
            print_log("AMP already enabled in config", logger="current", level=logging.WARNING)

    # ✅ Convert BatchNorm layers if requested
    if args.sync_bn != "none":
        cfg.sync_bn = args.sync_bn

    # ✅ Enable auto LR scaling
    if args.auto_scale_lr:
        if "auto_scale_lr" in cfg and "enable" in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError("Missing `auto_scale_lr` settings in config.")

    # ✅ Handle resume logic
    if args.resume == "auto":
        cfg.resume = True
        cfg.load_from = None
    elif args.resume:
        cfg.resume = True
        cfg.load_from = args.resume

    # Define the new vis_backends with MLflow
    mlflow_backend = dict(
        type='MLflowVisBackend',
        save_dir='temp_dir',
        exp_name='mvxnet-roi',
        run_name=f'mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class_{datetime.datetime.now()}',
        tags=None,
        params=None,
        tracking_uri=MLFLOW_TRACKING_URI,
        artifact_suffix=['.json', '.log', '.py', 'yaml']
    )

    # Ensure vis_backends exists and append MLflow if not already present
    if not hasattr(cfg, 'vis_backends') or not isinstance(cfg.vis_backends, list):
        cfg.vis_backends = []

    # Append standard and MLflow backends
    cfg.vis_backends = [
        dict(type='LocalVisBackend'),
        mlflow_backend
    ]

    # Update visualizer
    cfg.visualizer = dict(
        type='Det3DLocalVisualizer',
        vis_backends=cfg.vis_backends,
        name='visualizer'
    )

    # ✅ Initialize and start training
    runner = Runner.from_cfg(cfg) if "runner_type" not in cfg else RUNNERS.build(cfg)
    # runner = RUNNERS.build(cfg)

    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_param("work_dir", args.work_dir)
        mlflow.log_param("sync_bn", args.sync_bn)
        mlflow.log_param("amp", args.amp)
        mlflow.log_param("resume", args.resume)
        mlflow.log_param("config_file", args.config_file)

        # Start training
        print("🚀 Starting training...")
        runner.train()

        # Log final model artifact
        mlflow.log_artifact(args.work_dir)  # Logs model checkpoints & logs directory

        print("✅ Training complete. Model artifacts logged to MLflow!")


if __name__ == "__main__":
    main()
