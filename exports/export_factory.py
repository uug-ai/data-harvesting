from exports.base_export import BaseExport
from exports.yolov8_export import Yolov8Export
from utils.VariableClass import VariableClass


class ExportFactory:
    """
    Export Factory initializes specific export types.
    """
    def __init__(self):
        self._var = VariableClass()
        self.save_format = self._var.DATASET_FORMAT

    def init(self, proj_name):
        if self.save_format == 'yolov8':
            return Yolov8Export(proj_name)
        elif self.save_format == 'base':
            return BaseExport(proj_name)
        else:
            raise ModuleNotFoundError('Export type not found!')

