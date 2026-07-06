"""
Shared helpers for model-training flows.

Architecture: framework envs (segenv, flowenv, ...) are baked into the
image at BUILD time (see Dockerfile + setup_env/*.sh) — nothing gets
pip/mim-installed at trigger time. At RUN time, a flow only needs to:
  1. clone the lightweight model repo (configs / custom training code)
  2. run it using the pre-built env's python

This keeps triggers fast (no install step) and reproducible (the exact
framework version is fixed in the image, not whatever `pip install` happens
to resolve on a given day).
"""

import os
import subprocess
from pathlib import Path

from prefect import get_run_logger

REPO_ROOT = Path(os.getenv("TRAINING_REPOS_DIR", "/workspace/repos"))
RUNS_ROOT = Path(os.getenv("TRAINING_RUNS_DIR", "/workspace/runs"))
DATASETS_ROOT = Path(os.getenv("TRAINING_DATASETS_DIR", "/workspace/datasets"))
ENVS_ROOT = Path(os.getenv("TRAINING_ENVS_DIR", "/app"))  # segenv/flowenv live at /app/<name>env


def run_cmd(cmd: list[str], cwd: Path | None = None, env: dict | None = None) -> None:
    """Run a command, streaming output into the Prefect run logger."""
    logger = get_run_logger()
    logger.info(f"$ {' '.join(cmd)}  (cwd={cwd})")
    full_env = {**os.environ, **(env or {})}
    process = subprocess.Popen(
        cmd, cwd=cwd, env=full_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    for line in process.stdout:
        logger.info(line.rstrip())
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"Command failed (exit {process.returncode}): {' '.join(cmd)}")


def clone_or_update_repo(repo_url: str, branch: str, dir_name: str) -> Path:
    """Clone (or pull-latest) a lightweight MODEL repo — configs/custom code
    that depends on a framework already baked into the image. This is the
    only thing that happens at trigger time."""
    logger = get_run_logger()
    REPO_ROOT.mkdir(parents=True, exist_ok=True)
    target = REPO_ROOT / dir_name

    if (target / ".git").exists():
        logger.info(f"{dir_name} already cloned at {target}, pulling latest {branch}")
        run_cmd(["git", "fetch", "origin", branch], cwd=target)
        run_cmd(["git", "checkout", branch], cwd=target)
        run_cmd(["git", "pull", "origin", branch], cwd=target)
    else:
        logger.info(f"Cloning {repo_url} ({branch}) into {target}")
        run_cmd(["git", "clone", "--branch", branch, "--depth", "1", repo_url, str(target)])

    return target


def s3_sync_down(s3_uri: str, local_dir: str, extra_args: list[str] | None = None) -> Path:
    """
    Sync training data from S3 to local disk BEFORE training.

    Uses `aws s3 sync` rather than a live mount (s3fs/mountpoint-s3): training
    dataloaders do lots of small random reads, and networked filesystems
    handle that pattern poorly compared to local disk. Sync is idempotent —
    only new/changed objects transfer — so repeat triggers against the same
    S3 prefix are fast once the local cache (a Docker volume) is warm.

    If your dataset is too large to fit on local disk, mountpoint-s3 is the
    right tool instead — but that's a different trade-off (throughput vs
    capacity), not the default here.
    """
    logger = get_run_logger()
    local_path = Path(local_dir)
    local_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Syncing {s3_uri} -> {local_path}")
    cmd = ["aws", "s3", "sync", s3_uri, str(local_path)]
    if extra_args:
        cmd += extra_args
    run_cmd(cmd)
    return local_path


def s3_sync_up(local_path: str, s3_uri: str, extra_args: list[str] | None = None) -> None:
    """Sync training outputs (checkpoints, logs, metrics) from local disk up
    to S3 AFTER training. Same idempotent-sync approach as s3_sync_down."""
    logger = get_run_logger()
    logger.info(f"Syncing {local_path} -> {s3_uri}")
    cmd = ["aws", "s3", "sync", str(local_path), s3_uri]
    if extra_args:
        cmd += extra_args
    run_cmd(cmd)


class PrebuiltEnv:
    """Reference to a venv baked into the image at build time (e.g. segenv).
    Does NOT create or install anything — if it's missing, the image was
    built wrong, so fail loudly rather than silently falling back to a
    runtime install."""

    def __init__(self, env_name: str):
        self.path = ENVS_ROOT / env_name
        self.python = self.path / "bin" / "python"
        self.pip = self.path / "bin" / "pip"
        if not self.python.exists():
            raise RuntimeError(
                f"Expected prebuilt env at {self.path} but it doesn't exist. "
                f"This env must be baked into the Docker image at build time "
                f"(see setup_env/) — it is not created at run time."
            )

    def run(self, args: list[str], cwd: Path | None = None) -> None:
        run_cmd([str(self.python), *args], cwd=cwd)

    def pip_install(self, *args: str) -> None:
        """Only for a model repo's OWN lightweight extra deps (rare — most
        deps should already be in the baked env). Keep this list short;
        anything heavy belongs in setup_env/*.sh instead."""
        run_cmd([str(self.pip), "install", *args])
