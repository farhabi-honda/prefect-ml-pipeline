"""
Generalized REST trigger for Prefect deployments.

POST /trigger/{model_name}  -> starts a flow run for that model's deployment.

Add a new model by adding one line to MODEL_DEPLOYMENTS below (and, of
course, its flow + prefect.yaml deployment). No new endpoint needed.

Run locally with:
    uvicorn api.trigger:app --host 0.0.0.0 --port 8000 --reload
"""

import os

from fastapi import FastAPI, HTTPException

from pipeline.config import (
    Config,
    PrefectConfig,
    StorageInterface,
    TrainRequest,
    get_config,
)
from pipeline.deployments import run_deployment


class Utils:
    @staticmethod
    def get_model_name(dataset_uri: str) -> str:
        if dataset_uri.startswith("s3://"):
            dataset_uri = dataset_uri[5:]
        model_name = dataset_uri.split("/")[3]
        return model_name

    @staticmethod
    def get_dataset_name(dataset_uri: str) -> str:
        if dataset_uri.startswith("s3://"):
            dataset_uri = dataset_uri[5:]
        dataset_name = dataset_uri.split("/")[4]
        return dataset_name


def get_storage_interface(config: Config) -> StorageInterface:
    """
    Returns an instance of the appropriate StorageInterface based on the storage type specified in the config.
    """
    if config.storage_cfg.type == "s3":
        from pipeline.storage.s3 import S3Storage

        return S3Storage(
            endpoint_url=config.storage_cfg.s3.endpoint_url,
            aws_access_key_id=config.storage_cfg.s3.aws_access_key_id
            or os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=config.storage_cfg.s3.aws_secret_access_key
            or os.environ.get("AWS_SECRET_ACCESS_KEY"),
        )
    elif config.storage_cfg.type == "local":
        from pipeline.storage.local import LocalStorage

        return LocalStorage()
    else:
        raise ValueError(f"Unsupported storage type: {config.storage_cfg.type}")


config = get_config()
app = FastAPI(title="ML Training Trigger API")
storage_iface = get_storage_interface(config)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "registered_models": list(config.registered_model_names),
    }


@app.post("/train")
async def trigger(req: TrainRequest):
    dataset_uri = req.dataset_uri
    dataset_uri = "s3://ml-analytics/datasets/seg_ros/v20260707/dataset.yaml"

    model_name = Utils.get_model_name(dataset_uri)
    dataset_name = Utils.get_dataset_name(dataset_uri)

    if not any(x == model_name for x in config.registered_model_names):
        raise HTTPException(
            status_code=400,
            detail=f"{model_name} is not registered.",
        )

    try:
        train_cfg = next(
            (x for x in config.train_configs if x.model_cfg.name == model_name), None
        )

        if not train_cfg:
            raise ValueError(f"Failed to find train config for {model_name}.")

        flow_cfg = PrefectConfig(
            dataset_uri=dataset_uri,
            model_cfg=train_cfg.model_cfg,
            storage_iface=storage_iface,
            timeout_cfg=config.timeout,
            retry_cfg=config.retry,
        )
        flow_run = await run_deployment(
            name=config.deployment_name,
            parameters=flow_cfg,
            timeout=0,  # don't block waiting for training to finish
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "message": "Training run started.",
        "flow_run_id": str(flow_run.id),
        "model": model_name,
        "dataset": dataset_name,
        "deployment": config.deployment_name,
        "config": str(flow_cfg),  # helpful for debugging exactly this kind of issue
    }
