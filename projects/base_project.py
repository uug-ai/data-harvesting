from os.path import (
    join as pjoin,
    dirname as pdirname,
    abspath as pabspath
)
from projects.ibase_project import IBaseProject
from utils.VariableClass import VariableClass


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
        print(f'1. Created/Found project folder under {self.proj_dir} path')

    # def create_result_save_dir(self):
    #     """
    #     See ibase_project.py
    #
    #     Returns:
    #         None
    #     """
    #     if self._var.DATASET_FORMAT == 'yolov8':
    #         result_dir_path = pjoin(self.proj_dir, f'{datetime.now().strftime("%d-%m-%Y_%H-%M-%S")}')
    #         image_dir_path = pjoin(result_dir_path, 'images')
    #         label_dir_path = pjoin(result_dir_path, 'labels')
    #         yaml_path = pjoin(result_dir_path, 'data.yaml')
    #         return result_dir_path, image_dir_path, label_dir_path, yaml_path
    #     else:
    #         raise TypeError('Unsupported dataset format!')
