from os.path import (
    join as pjoin,
    dirname as pdirname,
    abspath as pabspath,
    basename as pbasename
)
from projects.ibase_project import IBaseProject
from utils.VariableClass import VariableClass
from ultralytics import YOLO

import yaml
import os
import torch


class BaseProject(IBaseProject):
    """
    Base Project that implements common functions, every project should inherit this.
    """

    def __init__(self):
        """
        Constructor.
        """
        self._var = VariableClass()
        self._config = None
        self.name = self._var.PROJECT_NAME
        self.proj_dir = None
        self.mapping = None
        self.device = None
        self.models = []

    def condition_func(self, total_results):
        """
        See ibase_project.py
        """
        raise NotImplemented('Should override this!!!')

    def class_mapping(self, models):
        """
        See ibase_project.py
        """
        raise NotImplemented('Should override this!!!')

    def create_proj_save_dir(self):
        """
        See ibase_project.py
        """
        _cur_dir = pdirname(pabspath(__file__))
        self.proj_dir = pjoin(_cur_dir, f'../data/{self.name}')
        self.proj_dir = pabspath(self.proj_dir)  # normalise the link
        print(f'3. Created/Found project folder under {self.proj_dir} path')

    def connect_models(self):
        """
        Initializes the YOLO models and connects them to the appropriate device (CPU or GPU).

        Returns:
            tuple: A tuple containing two YOLO models.

        Raises:
            ModuleNotFoundError: If the models cannot be loaded.
        """
        raise NotImplemented('Should override this!!!')

    def __read_config__(self, path):
        """
        See ibase_project.py
        """
        with open(path, 'r') as file:
            config = yaml.safe_load(file)

        print(f'Reading configuration  file {pbasename(path)}...')
        model_names = config.get('models')
        allowed_classes = config.get('allowed_classes')

        if model_names and allowed_classes and len(model_names) == len(allowed_classes):
            print('Configuration file valid!')
            return config

        raise TypeError('Error while reading configuration file, '
                        'make sure models and allowed_classes have the same size')

    def __connect_models__(self):
        """
        See ibase_project.py
        """
        _cur_dir = os.getcwd()
        # initialise the yolo model, additionally use the device parameter to specify the device to run the model on.
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        _cur_dir = pdirname(pabspath(__file__))
        model_dir = pjoin(_cur_dir, f'../models')
        model_dir = pabspath(model_dir)  # normalise the link

        models = []
        for model_name in self._config.get('models'):
            model = YOLO(pjoin(model_dir, model_name)).to(self.device)
            models.append(model)

        return models

    def reset_models(self):
        """
        See ibase_project.py
        """
        self.models = self.__connect_models__()
