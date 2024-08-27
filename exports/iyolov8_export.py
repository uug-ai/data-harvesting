from abc import ABC, abstractmethod


class IYolov8Export(ABC):

    @abstractmethod
    def initialize_save_dir(self):
        pass

    @abstractmethod
    def save_frame(self, frame, predicted_frames, cv2, labels_and_boxes):
        pass

    @abstractmethod
    def create_yaml(self, model2):
        pass
