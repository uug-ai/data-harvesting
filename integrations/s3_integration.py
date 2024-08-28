import boto3
import os

from utils.VariableClass import VariableClass


class S3Integration:
    """
    S3 Integration class that implements functions for connecting, uploading single file and dataset
    to S3 compatible platform.
    """

    def __init__(self):
        """
        Constructor.
        """
        self._var = VariableClass()
        self.session, self.agent = self.__connect__()
        self.bucket = self._var.S3_BUCKET
        self.__check_bucket_exists__(self.bucket)

    def __connect__(self):
        """
        See is3_integration.py

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

    def upload_file(self, source_path, output_path):
        """
        See is3_integration.py
        """
        try:
            self.agent.upload_file(source_path, self.bucket, output_path)
            print(f"Successfully uploaded '{source_path}' to 's3://{self.bucket}/{output_path}'")
        except Exception as e:
            print(f"Failed to upload '{source_path}' to 's3://{self.bucket}/{output_path}': {e}")

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
                self.upload_file(source_path, output_path)
                print(f'Uploaded: {source_path} to s3://{self.bucket}/{output_path}')

    def __check_bucket_exists__(self, bucket_name):
        """
        See is3_integration.py
        """
        try:
            self.agent.head_bucket(Bucket=bucket_name)
            print(f"Bucket '{bucket_name}' found.")

        except:
            raise ModuleNotFoundError(f"Bucket '{bucket_name}' does not exist.")
