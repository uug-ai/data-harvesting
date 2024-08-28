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
