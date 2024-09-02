from abc import ABC, abstractmethod


class IFlatExport(ABC):
    """
    Interface for Base Export.
    """

    @abstractmethod
    def initialize_save_dir(self):
        """
        Initializes save directory for Base export format
        """
        pass

    @abstractmethod
    def save_frame(self, frame, predicted_frames, cv2, labels_and_boxes):
        """
        Saves a single frames as well as it predicted annotation.
        It should save 2 separate files under the same name, 1 .png for the raw frame and 1 .txt for the annotations.

        Args:
            frame: The current frame to be saved.
            predicted_frames: Frames with predictions that might need to be saved alongside the original.
            cv2: The OpenCV module used for image processing, passed in to avoid tight coupling.
            labels_and_boxes: A list containing labels and their corresponding bounding boxes for the frame.
        """
        pass
