"""S3 storage implementation for the Prefect ML pipeline."""

import os
from typing import Any, Dict
from pipeline.storage.storage import StorageInterface
import json
import yaml


class S3Storage(StorageInterface):
    """
    S3 storage implementation for the Prefect ML pipeline.
    This class provides methods to interact with an S3-compatible storage service, allowing
    for uploading, downloading, listing, and deleting files in a specified bucket.
    """

    def __init__(
        self, endpoint_url: str, aws_access_key_id: str, aws_secret_access_key: str
    ):
        import boto3

        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )

    def bucket_exists(self, bucket_name: str) -> bool:
        try:
            return self.client.bucket_exists(bucket_name)
        except Exception as e:
            raise Exception(f"Error checking if bucket exists: {e}")

    def object_exists(self, bucket_name: str, object_name: str) -> bool:
        try:
            self.client.head_object(Bucket=bucket_name, Key=object_name)
            return True
        except self.client.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            else:
                raise Exception(f"Error checking if object exists: {e}")

    def create_bucket(self, bucket_name: str) -> None:
        try:
            if not self.bucket_exists(bucket_name):
                self.client.make_bucket(bucket_name)
        except Exception as e:
            raise Exception(f"Error creating bucket: {e}")

    def upload_file(self, bucket_name: str, object_name: str, file_path: str) -> None:
        try:
            self.client.upload_file(file_path, bucket_name, object_name)
        except Exception as e:
            raise Exception(f"Error uploading file: {e}")

    def download_file(self, bucket_name: str, object_name: str, file_path: str) -> None:
        try:
            self.client.download_file(bucket_name, object_name, file_path)
        except Exception as e:
            raise Exception(f"Error downloading file: {e}")

    def list_objects(self, bucket_name: str) -> Dict[str, Any]:
        try:
            response = self.client.list_objects_v2(Bucket=bucket_name)
            return {obj["Key"]: obj for obj in response.get("Contents", [])}
        except Exception as e:
            raise Exception(f"Error listing objects: {e}")

    def delete_object(self, bucket_name: str, object_name: str) -> None:
        try:
            self.client.delete_object(Bucket=bucket_name, Key=object_name)
        except Exception as e:
            raise Exception(f"Error deleting object: {e}")

    def read_yaml(self, uri: str) -> dict:
        bucket_name, object_name = self.parse_uri(uri)
        response = self.client.get_object(
            Bucket=bucket_name,
            Key=object_name,
        )
        return yaml.safe_load(response["Body"].read())

    def read_json(self, uri: str) -> dict:
        bucket_name, object_name = self.parse_uri(uri)
        response = self.client.get_object(
            Bucket=bucket_name,
            Key=object_name,
        )
        return json.loads(response["Body"].read())

    def read_string(self, uri: str) -> str:
        bucket_name, object_name = self.parse_uri(uri)
        response = self.client.get_object(
            Bucket=bucket_name,
            Key=object_name,
        )
        return response["Body"].read().decode()

    def upload_object(self, bucket_name: str, file_path: str, obj_bytes: bytes):
        self.client.put_object(
            Bucket=bucket_name,
            Key=file_path,
            Body=obj_bytes,
        )

    def upload(self, src_uri: str, dst_uri: str) -> None:
        bucket_name, object_name = self.parse_uri(dst_uri)
        self.upload_file(bucket_name, object_name, src_uri)

    def download(self, src_uri: str, dst_uri: str) -> None:
        bucket_name, object_name = self.parse_uri(src_uri)
        self.download_file(bucket_name, object_name, dst_uri)

    def delete(self, uri: str) -> None:
        bucket_name, object_name = self.parse_uri(uri)
        self.delete_object(bucket_name, object_name)

    def list_files(self, uri: str) -> list:
        bucket_name, prefix = self.parse_uri(uri)
        objects = self.list_objects(bucket_name)
        return [key for key in objects.keys() if key.startswith(prefix)]

    def exists(self, uri: str) -> bool:
        bucket_name, object_name = self.parse_uri(uri)
        objects = self.list_objects(bucket_name)
        return object_name in objects.keys()

    def get_uri(self, bucket_name: str, object_name: str) -> str:
        return f"s3://{bucket_name}/{object_name}"

    def parse_uri(self, uri: str) -> tuple[str, str]:
        if not uri.startswith("s3://"):
            raise ValueError(f"Invalid S3 URI: {uri}")
        parts = uri[5:].split("/", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid S3 URI: {uri}")
        return parts[0], parts[1]

    def paginate_list_objects(self, bucket_name: str, prefix: str = "") -> list:
        paginator = self.client.get_paginator("list_objects_v2")
        page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

        all_objects = []
        for page in page_iterator:
            all_objects.extend(page.get("Contents", []))
        return all_objects

    def upload_directory(
        self, bucket_name: str, directory_path: str, prefix: str = ""
    ) -> None:
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                object_name = os.path.join(
                    prefix, os.path.relpath(file_path, directory_path)
                )
                self.upload_file(bucket_name, object_name, file_path)

    def download_directory(
        self, bucket_name: str, prefix: str, local_directory: str
    ) -> None:
        objects = self.paginate_list_objects(bucket_name, prefix)
        for obj in objects:
            object_name = obj["Key"]
            relative_path = os.path.relpath(object_name, prefix)
            local_file_path = os.path.join(local_directory, relative_path)
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            self.download_file(bucket_name, object_name, local_file_path)
