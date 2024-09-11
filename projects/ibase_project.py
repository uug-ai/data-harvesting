from abc import ABC, abstractmethod


class IBaseProject(ABC):
    """
    Interface for Base Project.
    """

    @abstractmethod
    def condition_func(self, total_results):
        """
        Defines a condition function that operates logic on results of 2 models.
        In Base Class it raises NotImplementedError, every project should override this function with custom logic.

        Args:
            total_results: The total results of all models.
        Returns:
            See base_project.py
        """
        pass

    @abstractmethod
    def class_mapping(self, models):
        """
        Maps classes between two models using a provided mapping.
        As a convention, the 2nd model class would be used as the final result.
        In Base Class it raises NotImplementedError, every project should override this function with custom logic.

        Args:
            models: The list of used models.

        Returns:
            See base_project.py
        """
        pass

    @abstractmethod
    def create_proj_save_dir(self):
        """
        Create project save directory after initializing the project.
        """
        pass

    @abstractmethod
    def __read_config__(self, path):
        """
        Read project's configuration file.

        Returns:
            tuple: Configuration file in dictionary format.

        Raises:
            TypeError: If the models cannot be loaded.
        """
        pass

    @abstractmethod
    def __connect_models__(self):
        """
        Initializes the YOLO models and connects them to the appropriate device (CPU or GPU).

        Returns:
            tuple: A tuple containing two YOLO models.

        Raises:
            ModuleNotFoundError: If the models cannot be loaded.
        """
        pass

    @abstractmethod
    def reset_models(self):
        """
        Reset model after processing video to avoid memory allocation error when the upcoming video comes in with
        different resolution.
        """
        pass
