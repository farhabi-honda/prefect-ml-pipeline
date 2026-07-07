from pathlib import Path
import os
from pipeline.storage.storage import StorageInterface


class LocalStorage(StorageInterface):
    """
    A simple local storage implementation that saves files to a specified directory.
    This can be useful for testing or development purposes, especially when you want to avoid
    using cloud storage services like S3.
    """

    @classmethod
    def upload(cls, file_path: str, destination: str) -> None:
        destination_path = Path(destination)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        os.rename(file_path, destination_path)

    @classmethod
    def download(cls, source: str, destination: str) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        os.rename(source_path, destination_path)

    @classmethod
    def delete(cls, file_path: str) -> None:
        target_path = Path(file_path)
        if target_path.exists():
            os.remove(target_path)
        else:
            raise FileNotFoundError(f"{file_path} not found in local storage.")

    @classmethod
    def list_files(cls, directory: str) -> list:
        target_directory = Path(directory)
        if not target_directory.exists():
            raise FileNotFoundError(f"{directory} not found in local storage.")
        return [f.name for f in target_directory.iterdir() if f.is_file()]

    @classmethod
    def exists(cls, file_path: str) -> bool:
        target_path = Path(file_path)
        return target_path.exists()

    @classmethod
    def read_yaml(cls, file_path: str) -> dict:
        import yaml

        target_path = Path(file_path)
        if not target_path.exists():
            raise FileNotFoundError(f"{file_path} not found in local storage.")
        with open(target_path, "r") as f:
            return yaml.safe_load(f)

    @classmethod
    def read_json(cls, file_path: str) -> dict:
        import json

        target_path = Path(file_path)
        if not target_path.exists():
            raise FileNotFoundError(f"{file_path} not found in local storage.")
        with open(target_path, "r") as f:
            return json.load(f)

    @classmethod
    def read_string(cls, file_path: str) -> str:
        target_path = Path(file_path)
        if not target_path.exists():
            raise FileNotFoundError(f"{file_path} not found in local storage.")
        with open(target_path, "r") as f:
            return f.read()
