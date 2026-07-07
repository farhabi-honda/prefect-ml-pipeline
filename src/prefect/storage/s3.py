"""S3 storage module for interacting with S3-compatible object storage.

This module provides a high-level interface for common S3 operations
including reading YAML files, downloading individual files, and
downloading entire prefixes.
"""

from pathlib import Path
from typing import Any, Dict

import boto3
import yaml


class S3Storage:
    """Client for interacting with S3-compatible object storage.

    This class provides methods to read and download objects from S3-compatible
    storage systems. It supports both individual file downloads and recursive
    prefix downloads.

    Attributes:
        client: The boto3 S3 client instance used for all operations.
    """

    def __init__(
        self,
        endpoint_url: str,
        access_key: str,
        secret_key: str,
    ) -> None:
        """Initialize S3Storage client.

        Args:
            endpoint_url: The S3 endpoint URL (e.g., for MinIO, self-hosted S3).
            access_key: AWS access key ID for authentication.
            secret_key: AWS secret access key for authentication.
        """
        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )

    def read_yaml(self, bucket: str, key: str) -> Any:
        """Read and parse a YAML file from S3.

        Args:
            bucket: The S3 bucket name.
            key: The object key (path) within the bucket.

        Returns:
            The parsed YAML content as a Python object.

        Raises:
            botocore.exceptions.ClientError: If the object cannot be retrieved.
            yaml.YAMLError: If the YAML is malformed.
        """
        obj = self.client.get_object(Bucket=bucket, Key=key)
        return yaml.safe_load(obj["Body"])

    def download(self, bucket: str, key: str, dst: Path) -> None:
        """Download a single file from S3 to the local filesystem.

        Creates parent directories as needed. If the destination file already
        exists, it will be overwritten.

        Args:
            bucket: The S3 bucket name.
            key: The object key (path) within the bucket.
            dst: The local destination path where the file will be saved.

        Raises:
            botocore.exceptions.ClientError: If the download fails.
            OSError: If directory creation fails.
        """
        dst.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(bucket, key, str(dst))

    def download_prefix(self, bucket: str, prefix: str, dst: Path) -> None:
        """Download all objects with a given prefix from S3 recursively.

        Downloads all objects in the bucket that start with the given prefix,
        preserving the relative directory structure. Creates local directories
        as needed.

        Args:
            bucket: The S3 bucket name.
            prefix: The prefix (directory path) to download. Objects are
                downloaded relative to this prefix.
            dst: The local destination directory where objects will be saved,
                preserving the relative structure.

        Raises:
            botocore.exceptions.ClientError: If the listing or download fails.
            OSError: If directory creation fails.
        """
        paginator = self.client.get_paginator("list_objects_v2")

        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]

                # Preserve relative directory structure
                rel = Path(key).relative_to(prefix)
                target = dst / rel

                target.parent.mkdir(parents=True, exist_ok=True)
                self.client.download_file(bucket, key, str(target))
