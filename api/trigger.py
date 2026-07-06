"""
Generalized REST trigger for Prefect deployments.

POST /trigger/{model_name}  -> starts a flow run for that model's deployment.

Add a new model by adding one line to MODEL_DEPLOYMENTS below (and, of
course, its flow + prefect.yaml deployment). No new endpoint needed.

Run locally with:
    uvicorn api.trigger:app --host 0.0.0.0 --port 8000 --reload
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prefect.deployments import run_deployment

app = FastAPI(title="ML Training Trigger API")

# model_name (used in the URL) -> Prefect deployment name ("flow-name/deployment-name")
MODEL_DEPLOYMENTS = {
    "segmentation-train": "segmentation-train-flow/segmentation-train-local",
    "detection3d-train": "detection3d-train-flow/detection3d-train-local",
}


class TrainRequest(BaseModel):
    repo_url: str | None = None
    branch: str | None = None
    config_path: str | None = None
    work_dir: str | None = None
    extra_args: list[str] | None = None
    data_s3_uri: str | None = None       # e.g. "s3://my-bucket/datasets/cityscapes/"
    artifacts_s3_uri: str | None = None  # e.g. "s3://my-bucket/runs/segmentation/run-001/"
    dataset_dir: str | None = None


@app.get("/health")
def health():
    return {"status": "ok", "models": list(MODEL_DEPLOYMENTS.keys())}


@app.post("/trigger/{model_name}")
async def trigger_training(model_name: str, req: TrainRequest):
    deployment_name = MODEL_DEPLOYMENTS.get(model_name)
    if not deployment_name:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown model '{model_name}'. Available: {list(MODEL_DEPLOYMENTS.keys())}",
        )

    # Only pass fields the caller actually set — otherwise the flow's own
    # defaults (e.g. repo_url, branch) get overridden with None.
    parameters = {k: v for k, v in req.model_dump().items() if v is not None}

    try:
        flow_run = await run_deployment(
            name=deployment_name,
            parameters=parameters,
            timeout=0,  # don't block waiting for training to finish
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "message": "Training run triggered",
        "flow_run_id": str(flow_run.id),
        "model": model_name,
        "deployment": deployment_name,
    }
