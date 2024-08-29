from exports.base_export import BaseExport
from exports.yolov8_export import Yolov8Export
from utils.VariableClass import VariableClass


class ExportFactory:
    """
    Export Factory initializes specific export types.
    """

    def __init__(self):
        self._var = VariableClass()
        self.name = self._var.DATASET_FORMAT

    def init(self):
        """
        Initializes specific export with given name.

        Returns:
            Initialized corresponding export object.
        """
        if self.name == 'yolov8':
            return Yolov8Export(self.name)
        elif self.name == 'base':
            return BaseExport(self.name)
        else:
            raise ModuleNotFoundError('Export type not found!')

