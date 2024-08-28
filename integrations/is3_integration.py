from abc import ABC, abstractmethod


class IS3Integration(ABC):
    """
    Interface for S3 Integration class.
    """

    @abstractmethod
    def upload_file(self, source_path, output_path):
        """
        Upload a single file to S3 compatible platform.

        Args:
            source_path: File save path
            output_path: Desired path we want to save in S3
        """
        pass

    @abstractmethod
    def upload_dataset(self, src_project_path):
        """
        Upload dataset to S3 compatible platform.

        Args:
            src_project_path: Projecet save path
        """
        pass

    @abstractmethod
    def __connect__(self):
        """
        Connect to S3 compatible agent.
        You need to provide S3 parameters in .env file.
        """
        pass

    @abstractmethod
    def __check_bucket_exists__(self, bucket_name):
        """
        Check if input bucket exists after connecting to S3 compatible agent.
        You need to provide S3 parameters in .env file.

        Args:
            bucket_name: Bucket name.

        Returns:
            True or False
        """
        pass
