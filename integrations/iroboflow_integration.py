from abc import ABC, abstractmethod


class IRoboflowIntegration(ABC):
    """
    Interface for Roboflow Integration class.
    """

    @abstractmethod
    def upload_dataset(self, src_project_path):
        """
        Upload dataset to Roboflow platform.

        Args:
            src_project_path: Project save path
        """
        pass
