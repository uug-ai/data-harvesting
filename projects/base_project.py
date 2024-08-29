from os.path import (
    join as pjoin,
    dirname as pdirname,
    abspath as pabspath
)
from projects.ibase_project import IBaseProject
from utils.VariableClass import VariableClass
from ultralytics import YOLO

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
        self.name = self._var.PROJECT_NAME
        self.proj_dir = None
        self.model, self.model2 = self.__connect_models__()
        self.mapping = None
        self.device = None

    def condition_func(self, results1, results2, mapping):
        """
        See ibase_project.py
        """
        raise NotImplemented('Should override this!!!')

    def class_mapping(self, model1, model2):
        """
        See ibase_project.py
        """
        raise NotImplemented('Should override this!!!')

    def create_proj_save_dir(self, dir_name):
        """
        See ibase_project.py
        """
        _cur_dir = pdirname(pabspath(__file__))
        self.proj_dir = pjoin(_cur_dir, f'../data/{dir_name}')
        self.proj_dir = pabspath(self.proj_dir)  # normalise the link
        print(f'1. Created/Found project folder under {self.proj_dir} path')

    def __connect_models__(self):
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
        model_dir = pjoin(_cur_dir, f'../models')
        model_dir = pabspath(model_dir)  # normalise the link

        if not self._var.MODEL_NAME:
            raise ModuleNotFoundError('Model not found!')

        model = YOLO(pjoin(model_dir, self._var.MODEL_NAME)).to(self.device)
        model2 = None
        if self._var.MODEL_NAME_2:
            model2 = YOLO(pjoin(model_dir, self._var.MODEL_NAME_2)).to(self.device)

        print(f'2. Using device: {self.device}')
        if model and model2:
            print(f'3. Using dual mode for 2 models: {self._var.MODEL_NAME} and  {self._var.MODEL_NAME_2}')
        elif model:
            print(f'3. Using single mode for model: {self._var.MODEL_NAME}')
        else:
            raise ModuleNotFoundError('Something wrong happened!')

        return model, model2
