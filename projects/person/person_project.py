from projects.base_project import BaseProject
from projects.person.iperson_project import IPersonProject

config_path = './projects/person/person_config.yaml'


class PersonProject(BaseProject, IPersonProject):
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
        self.number_of_persons = int(
            self._config.get('number_of_persons', '1'))
        self.models, self.models_allowed_classes = self.connect_models()
        self.mapping = self.class_mapping(self.models)
        self.create_proj_save_dir()

    def condition_func(self, total_results):
        """
        Apply custom condition for the person project.
        For each frame processed by all models, all conditions below have to be satisfied:
        - All models have to return results
        - Model0 has PERSON detection

        Returns:
            None
        """
        person_model0 = 0

        number_of_persons = 0
        for i in range(0, len(total_results[0].boxes)):
            box = total_results[0].boxes[i]
            if box.cls == person_model0:
                number_of_persons += 1

        if number_of_persons == self.number_of_persons:
            return True
        else:
            return False

    def class_mapping(self, models):
        """
        See iperson_project.py

        Returns:
            mapping dictionary.
            e.g: {0:2} where:
            - 0 is the class of model1.
            - 2 is the corresponding class of model2.
        """
        model_classes = self._config.get('allowed_classes')
        model_names = []
        for model in models:
            model_names.append({key: value.lower()
                               for key, value in model.names.items()})

        result = []

        # Iterate through each class index in model_classes[0]
        for class_index in model_classes[0]:
            # Get the class name from the first model
            class_name = model_names[0][class_index]

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
                # Get the corresponding class of the first model.
                return mapping[0]
        return None  # Return None if the class_id is not found in the mapping

    def connect_models(self):
        """
        Initializes the YOLO models and connects them to the appropriate device (CPU or GPU).

        Returns:
            models: A tuple containing two YOLO models.
            models_allowed_classes: List of corresponding allowed classes for each model.

        Raises:
            ModuleNotFoundError: If the models cannot be loaded.
        """

        models = self.__connect_models__()
        models_allowed_classes = self._config.get('allowed_classes')

        if not models:
            raise ModuleNotFoundError('Model not found!')

        print(f'1. Using device: {self.device}')
        print(f"2. Using {len(models)} models: {[model_name for model_name in self._config.get('models')]}")
        return models, models_allowed_classes
