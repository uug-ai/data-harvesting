from projects.helmet_project import HelmetProject
from utils.VariableClass import VariableClass


class ProjectFactory:
    """
    Project Factory initializes specific projects.
    """

    def __init__(self):
        self._var = VariableClass()
        self._name = self._var.PROJECT_NAME

    def init(self):
        """
        Initializes specific project with given name.

        Returns:
            Initialized corresponding project object.
        """
        if self._name == 'helmet':
            print('Initializing Helmet Detection Project...')
            return HelmetProject()
        else:
            raise ModuleNotFoundError('Project not found!')
