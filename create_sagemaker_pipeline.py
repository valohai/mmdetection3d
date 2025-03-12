import boto3
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.processing import Processor, ProcessingInput, ProcessingOutput
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.steps import ModelStep
from sagemaker.workflow.model_step import ModelStep
from sagemaker.model import Model
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet

# ✅ Set up session and role
sagemaker_session = sagemaker.Session()
region = sagemaker_session.boto_region_name
role = sagemaker.get_execution_role()
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
    inputs=[
        ProcessingInput(source="s3://your-bucket/raw-data/", destination="/opt/ml/processing/input")
    ],
    outputs=[
        ProcessingOutput(output_name="processed_data", source="/opt/ml/processing/output")
    ],
    code="helper_scripts/preprocessing_script.py",
)

# ✅ Define Training Step
estimator = Estimator(
    image_uri="910181886844.dkr.ecr.eu-north-1.amazonaws.com/mmdetection3d:with-sagemaker-training-mlflow",
    role=role,
    instance_count=1,
    instance_type="ml.g4dn.xlarge",
    output_path="s3://your-bucket/training-output/",
    sagemaker_session=pipeline_session,
)

training_step = TrainingStep(
    name="ModelTrainingStep",
    estimator=estimator,
    inputs={
        "kitti": processing_step.properties.ProcessingOutputConfig.Outputs["processed_data"].S3Uri
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

# ✅ Define Conditional Step for Deployment Approval
cond_gte = ConditionGreaterThanOrEqualTo(
    left=JsonGet(
        step_name=training_step.name,
        property_file=PropertyFile(name="TrainingMetrics", output_name="metrics", path="metrics.json"),
        json_path="validation_accuracy",
    ),
    right=0.8,  # Threshold for model approval
)

condition_step = ConditionStep(
    name="CheckModelQuality",
    conditions=[cond_gte],
    if_steps=[model_step],
    else_steps=[],
)

# ✅ Create Pipeline
pipeline = Pipeline(
    name="MMD3D-Pipeline",
    steps=[processing_step, training_step, condition_step],
    sagemaker_session=sagemaker_session,
)

# ✅ Submit Pipeline
pipeline.upsert(role_arn=role)
print(f"✅ Created pipeline: {pipeline.name}")
