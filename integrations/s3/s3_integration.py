import boto3
import os

from utils.VariableClass import VariableClass


class S3Integration:
    """
    S3 Integration class that implements functions for connecting, uploading single file and dataset
    to S3 compatible platform.
    """

    def __init__(self, name):
        """
        Constructor.
        """
        self.name = name
        self._var = VariableClass()
        self.session, self.agent = self.__connect__()
        self.bucket = self._var.S3_BUCKET
        self.__check_bucket_exists__(self.bucket)

    def upload_dataset(self, src_project_path):
        """
        See is3_integration.py
        """
        # Iterate over all the files in the folder, including sub folders
        for root, dirs, files in os.walk(src_project_path):
            for filename in files:
                # Construct the full file path
                source_path = os.path.join(root, filename)

                # Preserve the folder structure in the S3 path
                # Create the relative path from the source folder to the current file
                relative_path = os.path.relpath(source_path, src_project_path)

                # Construct the output path using DATASET_FORMAT and DATASET_VERSION, including the relative path
                output_path = f"{self._var.DATASET_FORMAT}-v{self._var.DATASET_VERSION}/{relative_path.replace(os.sep, '/')}"

                # Upload the file
                self.__upload_file__(source_path, output_path)
                print(f'Uploaded: {source_path} to s3://{self.bucket}/{output_path}')

    def __connect__(self):
        """
        Connect to S3 compatible agent.
        You need to provide S3 parameters in .env file.

        Returns:
            session: Connected session.
            agent: Connected agent.
        """
        session = boto3.session.Session()
        # Connect to S3 Compatible
        agent = session.client(
            self._var.INTEGRATION_NAME,
            endpoint_url=self._var.S3_ENDPOINT,
            aws_access_key_id=self._var.S3_ACCESS_KEY,
            aws_secret_access_key=self._var.S3_SECRET_KEY,
        )
        print('Connected!')

        return session, agent

    def __upload_file__(self, source_path, output_path):
        """
        Upload a single file to S3 compatible platform.

        Args:
            source_path: File save path
            output_path: Desired path we want to save in S3
        """
        try:
            self.agent.upload_file(source_path, self.bucket, output_path)
            print(f"Successfully uploaded '{source_path}' to 's3://{self.bucket}/{output_path}'")
        except Exception as e:
            print(f"Failed to upload '{source_path}' to 's3://{self.bucket}/{output_path}': {e}")

    def __check_bucket_exists__(self, bucket_name):
        """
        Check if input bucket exists after connecting to S3 compatible agent.
        You need to provide S3 parameters in .env file.

        Args:
            bucket_name: Bucket name.

        Returns:
            True or False
        """
        try:
            self.agent.head_bucket(Bucket=bucket_name)
            print(f"Bucket '{bucket_name}' found.")

        except:
            raise ModuleNotFoundError(f"Bucket '{bucket_name}' does not exist.")
