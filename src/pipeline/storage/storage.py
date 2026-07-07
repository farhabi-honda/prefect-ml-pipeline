from abc import ABC, abstractmethod


class StorageInterface(ABC):
    """Abstract base class for storage implementations."""

    @abstractmethod
    def upload(self, src_uri: str, dst_uri: str) -> None:
        """Upload a file to storage."""
        pass

    @abstractmethod
    def download(self, src_uri: str, dst_uri: str) -> None:
        """Download a file from storage."""
        pass

    @abstractmethod
    def delete(self, uri: str) -> None:
        """Delete a file from storage."""
        pass

    @abstractmethod
    def list_files(self, uri: str) -> list:
        """List files in a storage location."""
        pass

    @abstractmethod
    def exists(self, uri: str) -> bool:
        """Check if a file exists in storage."""
        pass

    @abstractmethod
    def read_yaml(self, uri: str) -> dict:
        """Read a YAML file from storage and return its contents as a dictionary."""
        pass

    @abstractmethod
    def read_json(self, uri: str) -> dict:
        """Read a JSON file from storage and return its contents as a dictionary."""
        pass

    @abstractmethod
    def read_string(self, uri: str) -> str:
        """Read a file from storage and return its contents as a string."""
        pass
