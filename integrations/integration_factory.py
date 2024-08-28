from integrations.roboflow_integration import RoboflowIntegration
from integrations.s3_integration import S3Integration
from utils.VariableClass import VariableClass


class IntegrationFactory:
    """
    Integration Factory initializes specific integration types.
    """
    def __init__(self):
        self._var = VariableClass()
        self.name = self._var.INTEGRATION_NAME

    def init(self):
        if self.name == 'roboflow':
            print('Initializing Roboflow agent ...')
            return RoboflowIntegration()
        elif self.name == 's3':
            print('Initializing S3 compatible agent ...')
            return S3Integration()
        else:
            raise ModuleNotFoundError('Integration type not found!')
