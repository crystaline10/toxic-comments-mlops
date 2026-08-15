import os

import joblib

import wandb

MODEL_ARTIFACT = (
    "crystalmcnama/"
    "wandb-registry-toxic-comment-model-registry/"
    "toxic_comment-model:production"
)


def load_production_model():
    run = wandb.init(
        project="toxic-comments-api",
        job_type="model-serving",
    )

    artifact = run.use_artifact(
        MODEL_ARTIFACT,
        type="model",
    )

    artifact_dir = artifact.download()

    model_path = os.path.join(
        artifact_dir,
        "toxic_comment_model.pkl",
    )

    model = joblib.load(model_path)

    run.finish()

    return model