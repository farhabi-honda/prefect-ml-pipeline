"""
Generalized REST trigger for Prefect deployments.

POST /trigger/{model_name}  -> starts a flow run for that model's deployment.

Add a new model by adding one line to MODEL_DEPLOYMENTS below (and, of
course, its flow + prefect.yaml deployment). No new endpoint needed.

Run locally with:
    uvicorn api.trigger:app --host 0.0.0.0 --port 8000 --reload
"""

from fastapi import FastAPI, HTTPException

from config import PrefectConfig, TrainRequest, get_config
from prefect.deployments import run_deployment


class Utils:
    @staticmethod
    def get_model_name(dataset_uri: str) -> str:
        if dataset_uri.startswith("s3://"):
            dataset_uri = dataset_uri[4:]
        model_name = dataset_uri.split("/")[3]
        return model_name

    @staticmethod
    def get_dataset_name(dataset_uri: str) -> str:
        if dataset_uri.startswith("s3://"):
            dataset_uri = dataset_uri[4:]
        model_name = dataset_uri.split("/")[4]
        return model_name


config = get_config()
app = FastAPI(title="ML Training Trigger API")


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
            model_cfg=train_cfg.model_cfg,
            dataset_uri=dataset_uri,
            backend_type=config.backend_type,
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
