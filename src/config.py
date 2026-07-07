from functools import lru_cache
from pathlib import Path
import yaml
from pydantic import BaseModel


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
    backend_type: str
    dataset_root_uri: str
    train_configs: list[TrainConfig]
    registered_model_names: list[str]
    deployment_name: str

    @classmethod
    def from_yaml(cls, filepath: str) -> "Config":
        with open(filepath) as f:
            return cls.model_validate(yaml.safe_load(f))


class TrainRequest(BaseModel):
    dataset_uri: str


class PrefectConfig(BaseModel):
    model_cfg: ModelConfig
    dataset_uri: str
    backend_type: str


@lru_cache
def get_config() -> Config:
    with open("config.yaml") as f:
        return Config.model_validate(yaml.safe_load(f))
