from exports.flat.flat_export import FlatExport
from exports.yolov8.yolov8_export import Yolov8Export
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
        elif self.name == 'flat':
            return FlatExport(self.name)
        else:
            raise ModuleNotFoundError('Export type not found!')

