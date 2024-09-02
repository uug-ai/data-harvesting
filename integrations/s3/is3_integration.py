from abc import ABC, abstractmethod


class IS3Integration(ABC):
    """
    Interface for S3 Integration class.
    """

    @abstractmethod
    def upload_dataset(self, src_project_path):
        """
        Upload dataset to S3 compatible platform.

        Args:
            src_project_path: Project save path
        """
        pass
