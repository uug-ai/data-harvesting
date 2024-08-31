from integrations.roboflow.roboflow_integration import RoboflowIntegration
from integrations.s3.s3_integration import S3Integration
from utils.VariableClass import VariableClass


class IntegrationFactory:
    """
    Integration Factory initializes specific integration types.
    """
    def __init__(self):
        self._var = VariableClass()
        self.name = self._var.INTEGRATION_NAME

    def init(self):
        """
        Initializes specific integration with given name.

        Returns:
            Initialized corresponding integration object.
        """
        if self.name == 'roboflow':
            print('Initializing Roboflow agent ...')
            return RoboflowIntegration(self.name)
        elif self.name == 's3':
            print('Initializing S3 compatible agent ...')
            return S3Integration(self.name)
        else:
            raise ModuleNotFoundError('Integration type not found!')
