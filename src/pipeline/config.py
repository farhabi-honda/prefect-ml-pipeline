from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel

from pipeline.storage.storage import StorageInterface


class S3StorageConfig(BaseModel):
    bucket_name: str
    endpoint_url: str
    aws_access_key_id: str
    aws_secret_access_key: str
    region_name: str | None = None


class LocalStorageConfig(BaseModel):
    base_path: str | None = None


class StorageConfig(BaseModel):
    type: str
    s3: S3StorageConfig | None = None
    local: LocalStorageConfig | None = None


class TimeoutConfig(BaseModel):
    training: int = 3600  # seconds
    evaluation: int = 1800  # seconds


class RetryConfig(BaseModel):
    training: int = 3
    evaluation: int = 2
    delay_seconds: int = 30  # seconds


class ModelConfig(BaseModel):
    name: str
    repo_uri: str
    commit: str
    config_path: str
    train_script_relpath: str
    extra_args: list[str]


class TrainConfig(BaseModel):
    model_cfg: ModelConfig
    flow_name: str


class Config(BaseModel):
    storage_cfg: StorageConfig
    dataset_root_uri: str
    train_configs: list[TrainConfig]
    registered_model_names: list[str]
    deployment_name: str
    timeout: TimeoutConfig
    retry: RetryConfig

    @classmethod
    def from_yaml(cls, filepath: str) -> "Config":
        with open(filepath) as f:
            return cls.model_validate(yaml.safe_load(f))


class TrainRequest(BaseModel):
    dataset_uri: str


class PrefectConfig(BaseModel):
    dataset_uri: str
    storage_iface: StorageInterface
    model_cfg: ModelConfig
    timeout_cfg: TimeoutConfig
    retry_cfg: RetryConfig


@lru_cache
def get_config() -> Config:
    with open("config.yaml") as f:
        return Config.model_validate(yaml.safe_load(f))
