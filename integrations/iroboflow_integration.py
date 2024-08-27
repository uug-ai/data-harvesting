from abc import ABC, abstractmethod


class IRoboflowIntegration(ABC):

    @abstractmethod
    def upload_dataset(self, src_project_path):
        pass

    @abstractmethod
    def __connect__(self):
        pass
