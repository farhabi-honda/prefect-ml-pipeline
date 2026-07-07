from pipeline.storage.minio import Minio
from minio.error import S3Error
from typing import Any, Dict
from pipeline.storage.storage import StorageInterface


class MinioStorage(StorageInterface):
    def __init__(
        self, endpoint: str, access_key: str, secret_key: str, secure: bool = True
    ):
        self.client = Minio(
            endpoint, access_key=access_key, secret_key=secret_key, secure=secure
        )

    def bucket_exists(self, bucket_name: str) -> bool:
        try:
            return self.client.bucket_exists(bucket_name)
        except S3Error as e:
            raise Exception(f"Error checking if bucket exists: {e}")

    def create_bucket(self, bucket_name: str) -> None:
        try:
            if not self.bucket_exists(bucket_name):
                self.client.make_bucket(bucket_name)
        except S3Error as e:
            raise Exception(f"Error creating bucket: {e}")

    def upload_file(self, bucket_name: str, object_name: str, file_path: str) -> None:
        try:
            self.client.fput_object(bucket_name, object_name, file_path)
        except S3Error as e:
            raise Exception(f"Error uploading file: {e}")

    def download_file(self, bucket_name: str, object_name: str, file_path: str) -> None:
        try:
            self.client.fget_object(bucket_name, object_name, file_path)
        except S3Error as e:
            raise Exception(f"Error downloading file: {e}")

    def list_objects(self, bucket_name: str) -> Dict[str, Any]:
        try:
            objects = self.client.list_objects(bucket_name)
            return {obj.object_name: obj for obj in objects}
        except S3Error as e:
            raise Exception(f"Error listing objects: {e}")

    def delete_object(self, bucket_name: str, object_name: str) -> None:
        try:
            self.client.remove_object(bucket_name, object_name)
        except S3Error as e:
            raise Exception(f"Error deleting object: {e}")
