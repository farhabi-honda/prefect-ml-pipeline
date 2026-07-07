from abc import ABC, abstractmethod
from pathlib import Path


class Storage(ABC):

    @abstractmethod
    def read_yaml(self, uri: str) -> dict: ...

    @abstractmethod
    def download(self, uri: str, destination: Path) -> Path: ...
