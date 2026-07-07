"""
3D object detection training flow — same baked-env + S3 sync pattern as
segmentation_train.py, using a `detenv` prebuilt into the image.

NOTE: `detenv` isn't created by the Dockerfile in this repo yet — add a
`python3 -m venv /app/detenv` line + a setup_env/install_mmdet3d_cu124.sh
script (mirroring install_mmseg_cu124.sh) before this flow will actually run.
"""

from pathlib import Path

from pipeline import flow, task, get_run_logger

from src.pipeline.flows.common import (
    clone_or_update_repo,
    s3_sync_down,
    s3_sync_up,
    PrebuiltEnv,
    RUNS_ROOT,
    DATASETS_ROOT,
)

ENV_NAME = "detenv"


@task(retries=1, retry_delay_seconds=15)
def clone_model_repo(repo_url: str, branch: str, dir_name: str) -> Path:
    return clone_or_update_repo(repo_url, branch, dir_name=dir_name)


@task
def install_extra_requirements(repo_dir: Path) -> None:
    logger = get_run_logger()
    req_file = repo_dir / "requirements.txt"
    if not req_file.exists():
        logger.info("No extra requirements.txt in model repo, skipping")
        return
    env = PrebuiltEnv(ENV_NAME)
    env.pip_install("-r", str(req_file))


@task(retries=2, retry_delay_seconds=30)
def download_dataset(data_s3_uri: str | None, dataset_dir: str) -> str | None:
    if not data_s3_uri:
        get_run_logger().info("No data_s3_uri provided, skipping dataset download")
        return None
    s3_sync_down(data_s3_uri, dataset_dir)
    return dataset_dir


@task
def run_training(
    repo_dir: Path, config_path: str, work_dir: str, extra_args: list[str] | None = None
) -> dict:
    logger = get_run_logger()
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    env = PrebuiltEnv(ENV_NAME)

    cmd = ["train.py", config_path, "--work-dir", work_dir]
    if extra_args:
        cmd += extra_args

    logger.info(f"[{ENV_NAME}] launching training with config: {config_path}")
    env.run(cmd, cwd=repo_dir)

    return {"status": "completed", "work_dir": work_dir, "config_path": config_path}


@task(retries=2, retry_delay_seconds=30)
def upload_artifacts(work_dir: str, artifacts_s3_uri: str | None) -> str | None:
    if not artifacts_s3_uri:
        get_run_logger().info("No artifacts_s3_uri provided, skipping upload")
        return None
    s3_sync_up(work_dir, artifacts_s3_uri)
    return artifacts_s3_uri


@flow(name="detection3d-train-flow")
def detection3d_train_flow(
    repo_url: str = "https://github.com/YOUR_ORG/detection3d-train.git",
    branch: str = "main",
    config_path: str = "configs/my_detection3d_config.py",
    work_dir: str | None = None,
    extra_args: list[str] | None = None,
    data_s3_uri: str | None = None,
    artifacts_s3_uri: str | None = None,
    dataset_dir: str | None = None,
):
    work_dir = work_dir or str(RUNS_ROOT / "detection3d" / "latest")
    dataset_dir = dataset_dir or str(DATASETS_ROOT / "detection3d")

    repo_dir = clone_model_repo(repo_url, branch, dir_name="detection3d-train")
    install_extra_requirements(repo_dir)
    download_dataset(data_s3_uri, dataset_dir)
    result = run_training(repo_dir, config_path, work_dir, extra_args)
    upload_artifacts(work_dir, artifacts_s3_uri)
    return result


if __name__ == "__main__":
    detection3d_train_flow()
