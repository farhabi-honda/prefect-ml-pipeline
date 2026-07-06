"""
Segmentation training flow.

Three ways this flow can get its mmsegmentation code, controlled by which
parameters you pass (checked in this priority order):

1. local_repo_path given -> use a repo bind-mounted from your HOST machine
   (see docker-compose.yml volume mount below). Use this to test against a
   repo you're actively editing locally, without cloning or rebuilding the
   image.
2. use_baked_repo=True -> use the mmsegmentation baked into the image at
   build time (/app/mmsegmentation, from setup_env/install_mmseg_cu124.sh).
3. otherwise (default) -> clone repo_url fresh, the real production path.
"""

from pathlib import Path

from prefect import flow, task, get_run_logger

from flows.common import (
    clone_or_update_repo,
    s3_sync_down,
    s3_sync_up,
    PrebuiltEnv,
    RUNS_ROOT,
    DATASETS_ROOT,
)

ENV_NAME = "segenv"
BAKED_MMSEG_PATH = Path("/app/mmsegmentation")  # baked in at image build time


@task(retries=1, retry_delay_seconds=15)
def clone_model_repo(repo_url: str, branch: str, dir_name: str) -> Path:
    return clone_or_update_repo(repo_url, branch, dir_name=dir_name)


@task
def install_extra_requirements(repo_dir: Path) -> None:
    """Optional: only if the model repo ships its own requirements.txt with
    a few extra deps on top of segenv. Skips cleanly if there isn't one."""
    logger = get_run_logger()
    req_file = repo_dir / "requirements.txt"
    if not req_file.exists():
        logger.info("No extra requirements.txt in model repo, skipping")
        return
    env = PrebuiltEnv(ENV_NAME)
    env.pip_install("-r", str(req_file))


@task(retries=2, retry_delay_seconds=30)
def download_dataset(data_s3_uri: str | None, dataset_dir: str) -> str | None:
    """Sync training data from S3 to local disk before training. Skipped
    entirely if no data_s3_uri is given."""
    if not data_s3_uri:
        get_run_logger().info("No data_s3_uri provided, skipping dataset download")
        return None
    s3_sync_down(data_s3_uri, dataset_dir)
    return dataset_dir


@task
def run_training(
    repo_dir: Path,
    config_path: str,
    work_dir: str,
    train_script: str = "tools/train.py",
    extra_args: list[str] | None = None,
) -> dict:
    logger = get_run_logger()
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    env = PrebuiltEnv(ENV_NAME)

    cmd = [train_script, config_path, "--work-dir", work_dir]
    if extra_args:
        cmd += extra_args

    logger.info(f"[{ENV_NAME}] launching training: {train_script} {config_path} (cwd={repo_dir})")
    env.run(cmd, cwd=repo_dir)

    return {"status": "completed", "work_dir": work_dir, "config_path": config_path}


@task(retries=2, retry_delay_seconds=30)
def upload_artifacts(work_dir: str, artifacts_s3_uri: str | None) -> str | None:
    """Sync checkpoints/logs up to S3 after training. Skipped if no
    artifacts_s3_uri given."""
    if not artifacts_s3_uri:
        get_run_logger().info("No artifacts_s3_uri provided, skipping upload")
        return None
    s3_sync_up(work_dir, artifacts_s3_uri)
    return artifacts_s3_uri


@flow(name="segmentation-train-flow")
def segmentation_train_flow(
    repo_url: str = "https://github.com/YOUR_ORG/segmentation-train.git",
    branch: str = "main",
    config_path: str = "configs/pspnet/pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py",
    work_dir: str | None = None,
    train_script: str = "tools/train.py",
    extra_args: list[str] | None = None,
    data_s3_uri: str | None = None,
    artifacts_s3_uri: str | None = None,
    dataset_dir: str | None = None,
    use_baked_repo: bool = False,
    local_repo_path: str | None = None,  # e.g. "/workspace/local-repo" — see docker-compose bind mount
):
    logger = get_run_logger()
    work_dir = work_dir or str(RUNS_ROOT / "segmentation" / "latest")
    dataset_dir = dataset_dir or str(DATASETS_ROOT / "segmentation")

    if local_repo_path:
        repo_dir = Path(local_repo_path)
        logger.info(f"LOCAL MODE: using bind-mounted repo at {repo_dir} (your host's live copy)")
        if not repo_dir.exists():
            raise RuntimeError(
                f"local_repo_path {repo_dir} doesn't exist inside the container. "
                f"Check the volume bind mount in docker-compose.yml actually points "
                f"at your local mmsegmentation folder."
            )
    elif use_baked_repo:
        logger.info(f"BAKED MODE: using the mmsegmentation baked into the image at {BAKED_MMSEG_PATH}")
        repo_dir = BAKED_MMSEG_PATH
    else:
        repo_dir = clone_model_repo(repo_url, branch, dir_name="segmentation-train")
        install_extra_requirements(repo_dir)

    download_dataset(data_s3_uri, dataset_dir)
    result = run_training(repo_dir, config_path, work_dir, train_script, extra_args)
    upload_artifacts(work_dir, artifacts_s3_uri)
    return result


if __name__ == "__main__":
    segmentation_train_flow()