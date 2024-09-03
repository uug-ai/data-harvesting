from os.path import (
    join as pjoin,
    dirname as pdirname,
    abspath as pabspath
)

from ultralytics import YOLO

from projects.base_project import BaseProject
from projects.helmet.ihelmet_project import IHelmetProject

import os
import torch

config_path = './projects/helmet/helmet_config.yaml'


class HelmetProject(BaseProject, IHelmetProject):
    """
    Helmet Project that implements functions for helmet-detection project.
    """

    def __init__(self):
        """
        Constructor.
        """
        super().__init__()
        self._config = self.__read_config__(config_path)
        self.temp_path = self._config.get('temp')
        self.min_width = int(self._config.get('min_width')) if self._config.get('min_width') else 0
        self.min_height = int(self._config.get('min_height')) if self._config.get('min_height') else 0
        self.models, self.models_allowed_classes = self.connect_models()
        self.mapping = self.class_mapping(self.models)
        self.create_proj_save_dir()

    def condition_func(self, total_results):
        """
        Apply custom condition for the helmet project.
        For each frame processed by all models, all conditions below have to be satisfied:
        - All models have to return results
        - Model0 has PERSON detection
        - Model1 has PERSON detection
        - Model0 has HELMET detection
        - All models have all PERSON bounding boxes with height greater than minimum_height
        - All models have all PERSON bounding boxes with width greater than minimum_width

        Returns:
            None
        """
        person_model0 = 2
        person_model1 = self.mapping[person_model0][1]  # Mapping person from model1 to model0
        helmet_model0 = 1

        has_person_model0 = any(box.cls == person_model0 for box in total_results[0].boxes)
        has_helmet_model0 = any(box.cls == helmet_model0 for box in total_results[0].boxes)
        has_person_model1 = any(box.cls == person_model1 for box in total_results[1].boxes)
        has_minimum_width_height_model0 = all(box.xywh[0, 2] > self.min_width
                                              and box.xywh[0, 3] > self.min_height for box in total_results[0].boxes
                                              if box.cls == person_model0)
        has_minimum_width_height_model1 = all(box.xywh[0, 2] > self.min_width
                                              and box.xywh[0, 3] > self.min_height for box in total_results[1].boxes
                                              if box.cls == person_model1)
        if has_person_model0 and has_helmet_model0 and has_person_model1 and has_minimum_width_height_model0 and has_minimum_width_height_model1:
            return True
        else:
            return False

    def class_mapping(self, models):
        """
        See ihelmet_project.py

        Returns:
            mapping dictionary.
            e.g: {0:2} where:
            - 0 is the class of model1.
            - 2 is the corresponding class of model2.
        """
        model_classes = self._config.get('allowed_classes')
        model_names = []
        for model in models:
            model_names.append({key: value.lower() for key, value in model.names.items()})

        result = []

        # Iterate through each class index in model_classes[0]
        for i, class_index in enumerate(model_classes[0]):
            class_name = model_names[0][class_index]  # Get the class name from the first model

            # Create a list to store the mapping for this class
            mapping = []

            # Iterate through each model's classes in model_classes
            for j in range(len(model_classes)):
                if class_name in model_names[j].values():
                    # Find the key associated with the class_name in the current model
                    for key, value in model_names[j].items():
                        if value == class_name:
                            mapping.append(key)
                            break
                else:
                    mapping.append(None)

            # Append the mapping to the result list
            result.append(mapping)

        return result

    def map_to_first_model(self, model_idx, class_id):
        # Iterate through the class mappings
        for i, mapping in enumerate(self.mapping):
            if mapping[model_idx] == class_id:
                return mapping[0]  # Get the corresponding class of the first model.
        return None  # Return None if the class_id is not found in the mapping

    def connect_models(self):
        """
        Initializes the YOLO models and connects them to the appropriate device (CPU or GPU).

        Returns:
            tuple: A tuple containing two YOLO models.

        Raises:
            ModuleNotFoundError: If the models cannot be loaded.
        """

        _cur_dir = os.getcwd()
        # initialise the yolo model, additionally use the device parameter to specify the device to run the model on.
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        _cur_dir = pdirname(pabspath(__file__))
        model_dir = pjoin(_cur_dir, f'../../models')
        model_dir = pabspath(model_dir)  # normalise the link

        models = []
        for model_name in self._config.get('models'):
            model = YOLO(pjoin(model_dir, model_name)).to(self.device)
            models.append(model)

        models_allowed_classes = self._config.get('allowed_classes')

        if not models:
            raise ModuleNotFoundError('Model not found!')

        print(f'2. Using device: {self.device}')
        print(f"3. Using {len(models)} models: {[model_name for model_name in self._config.get('models')]}")
        return models, models_allowed_classes
