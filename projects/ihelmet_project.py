from abc import ABC, abstractmethod


class IHelmetProject(ABC):
    """
    Interface for Helmet Project.
    """

    @abstractmethod
    def condition_func(self, results1, results2, mapping):
        """
        Defines a condition function that operates logic on results of 2 models.
        Conditions:
            - results1 and results2 not empty
            - Both contains person class
            - results2 contains helmet

        Args:
            results1: The result of the 1st model.
            results2: The result of the 2nd model.
            mapping: A mapping that defines the relationships between elements
                     in results1 and results2.
        Returns:
            See helmet_project.py
        """
        pass

    @abstractmethod
    def class_mapping(self, model1, model2):
        """
        Maps classes between two models using a provided mapping.
        As a convention, the 2nd model class would be used as the final result.

        Args:
            model1: The 1st input model.
            model2: The 2nd input model.

        Returns:
            See helmet_project.py
        """
        pass
