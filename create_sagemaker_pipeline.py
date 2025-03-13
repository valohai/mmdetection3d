import boto3
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.processing import Processor, ProcessingOutput
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.model_step import ModelStep
from sagemaker.model import Model

# ✅ Set up session and role
sagemaker_session = sagemaker.Session()
region = sagemaker_session.boto_region_name
role = "arn:aws:iam::910181886844:role/AmazonSageMaker-ExecutionRole"
pipeline_session = PipelineSession()

# ✅ Define Preprocessing Step
processor = Processor(
    image_uri="910181886844.dkr.ecr.eu-north-1.amazonaws.com/mmdetection3d:with-sagemaker-training-mlflow",
    role=role,
    instance_count=1,
    instance_type="ml.m5.large",
    sagemaker_session=pipeline_session,
)

processing_step = ProcessingStep(
    name="DataPreprocessingStep",
    processor=processor,
    outputs=[
        ProcessingOutput(
            output_name="output-1",  # ✅ Ensure consistency
            source="/opt/ml/processing/output"
        )
    ],
    code="sagemaker_utils/helper_prep_kitti_dataset.py",
)

# ✅ Define Training Step
estimator = Estimator(
    image_uri="910181886844.dkr.ecr.eu-north-1.amazonaws.com/mmdetection3d:with-sagemaker-training-mlflow",
    role=role,
    instance_count=1,
    instance_type="ml.g4dn.xlarge",
    output_path="s3://dd-sample-bucket-north/mmdetection3d/mmdetection3d-train-output/",
    sagemaker_session=pipeline_session,
)

training_step = TrainingStep(
    name="ModelTrainingStep",
    estimator=estimator,
    inputs={
        "kitti": TrainingInput(
            s3_data=processing_step.properties.ProcessingOutputConfig.Outputs["output-1"].S3Output.S3Uri
        )
    },
)

# ✅ Define Model Registration Step
model = Model(
    image_uri="910181886844.dkr.ecr.eu-north-1.amazonaws.com/mmdetection3d:with-sagemaker-training-mlflow",
    role=role,
    sagemaker_session=pipeline_session
)

model_step = ModelStep(
    name="RegisterModelStep",
    step_args=model.create(),
)

# ✅ Create Pipeline (without ConditionStep)
pipeline = Pipeline(
    name="MMD3D-Pipeline",
    steps=[processing_step, training_step, model_step],
    sagemaker_session=sagemaker_session,
)

# ✅ Submit Pipeline
pipeline.upsert(role_arn=role)
print(f"✅ Created pipeline: {pipeline.name}")
