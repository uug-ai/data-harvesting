from os.path import (
    join as pjoin,
    dirname as pdirname,
    abspath as pabspath
)
import os

from integrations.roboflow_helper import RoboflowHelper
from projects.ibase_project import IBaseProject
from utils.VariableClass import VariableClass
from datetime import datetime


class BaseProject(IBaseProject):
    """
    Base Project that implements common functions, every project should inherit this.
    """

    def __init__(self):
        """
        Constructor.
        """
        self._var = VariableClass()
        self.proj_dir = None

    def condition_func(self, results1, results2, mapping):
        """
        See ibase_project.py
        """
        raise NotImplemented('Should override this!!!')

    def class_mapping(self, results1, results2):
        """
        See ibase_project.py
        """
        raise NotImplemented('Should override this!!!')

    def create_proj_save_dir(self, dir_name):
        """
        See ibase_project.py

        Returns:
            None
        """
        _cur_dir = pdirname(pabspath(__file__))
        self.proj_dir = pjoin(_cur_dir, f'../data/{dir_name}')
        self.proj_dir = pabspath(self.proj_dir)  # normalise the link
        if self._var.DATASET_FORMAT == 'yolov8':
            os.makedirs(self.proj_dir, exist_ok=True)
            print(f'1. Created/Found project folder under {self.proj_dir} path')
        else:
            raise TypeError('Unsupported dataset format!')

    def create_result_save_dir(self):
        """
        See ibase_project.py

        Returns:
            None
        """
        if self._var.DATASET_FORMAT == 'yolov8':
            result_dir_path = pjoin(self.proj_dir, f'{datetime.now().strftime("%d-%m-%Y_%H-%M-%S")}')
            image_dir_path = pjoin(result_dir_path, 'images')
            label_dir_path = pjoin(result_dir_path, 'labels')
            yaml_path = pjoin(result_dir_path, 'data.yaml')
            return result_dir_path, image_dir_path, label_dir_path, yaml_path
        else:
            raise TypeError('Unsupported dataset format!')

    def upload_dataset(self, result_dir_path, yaml_path, model2):
        """
        See ibase_project.py

        Returns:
            None
        """
        if os.path.exists(result_dir_path) and self._var.RBF_UPLOAD:
            label_names = [name for name in list(model2.names.values())]
            self.__create_yaml__(yaml_path, label_names)

            rb = RoboflowHelper()
            if rb:
                rb.upload_dataset(result_dir_path)
        else:
            print(f'RBF_UPLOAD: {self._var.RBF_UPLOAD} or path not found, skipping uploading!')

    def __create_yaml__(self, file_path, label_names):
        """
        Create YAML configuration file with DATASET_FORMAT format (default yolov8).

        Returns:
            None
        """
        if self._var.DATASET_FORMAT == 'yolov8':
            with open(file_path, 'w') as my_file:
                content = 'names:\n'
                for name in label_names:
                    content += f'- {name}\n'  # class mapping for helmet detection project
                content += f'nc: {len(label_names)}'
                my_file.write(content)
        else:
            raise TypeError('Unsupported dataset format!')
