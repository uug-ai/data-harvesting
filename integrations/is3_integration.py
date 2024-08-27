from abc import ABC, abstractmethod


class IS3Integration(ABC):

    @abstractmethod
    def upload_file(self, source_path, output_path):
        pass

    @abstractmethod
    def upload_dataset(self, src_project_path):
        pass

    @abstractmethod
    def __connect__(self):
        pass

    @abstractmethod
    def __check_bucket_exists__(self, bucket_name):
        pass
