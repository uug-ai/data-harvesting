from abc import ABC, abstractmethod


class IBaseProject(ABC):
    """
    Interface for Base Project.
    """

    @abstractmethod
    def condition_func(self, results1, results2, mapping):
        """
        Defines a condition function that operates logic on results of 2 models.
        In Base Class it raises NotImplementedError, every project should override this function with custom logic.

        Args:
            results1: The result of the 1st model.
            results2: The result of the 2nd model.
            mapping: A mapping that defines the relationships between elements
                     in results1 and results2.
        Returns:
            See base_project.py
        """
        pass

    @abstractmethod
    def class_mapping(self, model1, model2):
        """
        Maps classes between two models using a provided mapping.
        As a convention, the 2nd model class would be used as the final result.
        In Base Class it raises NotImplementedError, every project should override this function with custom logic.

        Args:
            model1: The 1st input model.
            model2: The 2nd input model.

        Returns:
            See base_project.py
        """
        pass

    @abstractmethod
    def create_proj_save_dir(self, dir_name):
        """
        Creates a directory to save project-related files.

        Args:
            dir_name: The name of the directory to create.

        Returns:
            See base_project.py
        """
        pass

    @abstractmethod
    def create_result_save_dir(self):
        """
        Creates a directory to save results.

        Args:
            None

        Returns:
            See base_project.py
        """
        pass

    @abstractmethod
    def upload_dataset(self, result_dir_path, yaml_path, model2):
        """
        Uploads a dataset by specifying the paths to the results directory if RBF_UPLOAD=True.
        Create YAML configuration file with DATASET_FORMAT format (default yolov8).

        Args:
            result_dir_path: The path to the directory where results are stored.
            yaml_path: The path to the YAML configuration file.
            model2: The model to which the dataset will be uploaded.

        Returns:
            See base_project.py
        """
        pass