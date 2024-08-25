from projects.helmet_project import HelmetProject


class ProjectFactory:
    """
    Project Factory initializes specific projects.
    """

    def init(self, name):
        """
        Initializes specific projects with given name.

        Args:
            name: name of the project, should be 'helmet'.

        Returns:
            Initialized corresponding project object.
        """
        if name == 'helmet':
            print('Initializing Helmet Detection Project...')
            return HelmetProject()
        else:
            raise ModuleNotFoundError('Project not found!')
