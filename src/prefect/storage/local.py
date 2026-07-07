import yaml
import shutil

from base_storage import Storage


class LocalStorage(Storage):

    def read_yaml(self, uri):
        path = Path(uri.removeprefix("file://"))

        with open(path) as f:
            return yaml.safe_load(f)

    def download(self, uri, destination):
        src = Path(uri.removeprefix("file://"))

        shutil.copytree(src, destination)

        return destination
