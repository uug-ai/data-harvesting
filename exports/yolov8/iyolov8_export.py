from abc import ABC, abstractmethod


class IYolov8Export(ABC):
    """
    Interface for Yolov8 Export.
    """
    @abstractmethod
    def initialize_save_dir(self):
        """
        Initializes save directory for Yolov8 export format
        """
        pass

    @abstractmethod
    def save_frame(self, frame, predicted_frames, cv2, labels_and_boxes):
        """
        Saves a single frames as well as it predicted annotation.
        It should save 2 separate files under the same name,
            - 1 .png for the raw frame and is saved in images subdirectory.
            - 1 .txt for the annotations and is saved in labels subdirectory.

        Args:
            frame: The current frame to be saved.
            predicted_frames: Frames with predictions that might need to be saved alongside the original.
            cv2: The OpenCV module used for image processing, passed in to avoid tight coupling.
            labels_and_boxes: A list containing labels and their corresponding bounding boxes for the frame.
        """
        pass

    @abstractmethod
    def create_yaml(self, model2):
        """
        Create .yaml file to map annotation labels with their corresponding names.
        """
        pass
