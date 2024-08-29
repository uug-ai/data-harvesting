from ultralytics import YOLO

from projects.base_project import BaseProject
from projects.ihelmet_project import IHelmetProject


class HelmetProject(BaseProject, IHelmetProject):
    """
    Helmet Project that implements functions for helmet-detection project.
    """

    def __init__(self):
        """
        Constructor.
        """
        super().__init__()
        self.mapping = self.class_mapping(self.model, self.model2)
        self.create_proj_save_dir('helmet_detection')

    def condition_func(self, results1, results2, mapping):
        """
        See ihelmet_project.py

        Returns:
            None
        """
        person_model1 = 0
        person_model2 = mapping.get(person_model1)  # Mapping person from model1 to model2

        return (
                len(results1.boxes) > 0
                and len(results2.boxes) > 0
                and any(box.cls == person_model2 for box in results2.boxes)
                and any(box.cls == 1 for box in results2.boxes)
        )

    def class_mapping(self, model1: YOLO, model2: YOLO):
        """
        See ihelmet_project.py

        Returns:
            mapping dictionary.
            e.g: {0:2} where:
            - 0 is the class of model1.
            - 2 is the corresponding class of model2.
        """
        model_1_classes = self._var.MODEL_ALLOWED_CLASSES
        model_2_classes = self._var.MODEL_2_ALLOWED_CLASSES
        mapping = {}

        # Loop through allowed classes in model1 and model2
        for index1 in model_1_classes:
            class_name1 = model1.names.get(index1)  # Get the name for the index from model1

            for index2 in model_2_classes:
                class_name2 = model2.names.get(index2)  # Get the name for the index from model2

                if class_name1 and class_name2 and str.lower(class_name1) == str.lower(class_name2):
                    mapping[index1] = index2

        print('Finish mapping models.')
        return mapping
