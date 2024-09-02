from abc import ABC, abstractmethod


class IHelmetProject(ABC):
    """
    Interface for Helmet Project.
    """

    @abstractmethod
    def condition_func(self, total_results):
        """
        Defines a condition function that operates logic on results of 2 models.
        Conditions:
            - results1 and results2 not empty
            - Both contains person class
            - results2 contains helmet

        Args:
            total_results: The total results of all models.

        Returns:
            See helmet_project.py
        """
        pass

    @abstractmethod
    def class_mapping(self, models):
        """
        Maps classes between two models using a provided mapping.
        As a convention, the 2nd model class would be used as the final result.

        Args:
            models: List of models

        Returns:
            See helmet_project.py
        """
        pass
