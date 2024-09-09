from projects.helmet.helmet_project import HelmetProject
from projects.person.person_project import PersonProject
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
        elif self._name == 'person':
            print('Initializing Person Detection Project...')
            return PersonProject()
        else:
            raise ModuleNotFoundError('Project not found!')
