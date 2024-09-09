from exports.yolov8.iyolov8_export import IYolov8Export
from utils.VariableClass import VariableClass
from os.path import (
    join as pjoin,
    dirname as pdirname,
    abspath as pabspath,
)
import os
import time


class Yolov8Export(IYolov8Export):
    """
    Yolov8 Export class that implements functions for
    initializing, saving frame and creating yaml file under specific format.
    """

    def __init__(self, name):
        """
        Constructor.
        """
        self.name = name
        self._var = VariableClass()
        _cur_dir = pdirname(pabspath(__file__))
        self.proj_dir = pjoin(_cur_dir, f'../../data/{name}')
        self.proj_dir = pabspath(self.proj_dir)  # normalise the link
        self.image_dir_path = None
        self.label_dir_path = None
        self.yaml_path = None
        self.result_dir_path = None
        self.result_labeled_dir_path = None

    def initialize_save_dir(self):
        """
        See iyolov8_export.py

        Returns:
            Success true or false.
        """
        self.result_dir_path = pjoin(self.proj_dir, f'{self._var.DATASET_FORMAT}-v{self._var.DATASET_VERSION}')
        os.makedirs(self.result_dir_path, exist_ok=True)

        self.image_dir_path = pjoin(self.result_dir_path, 'images')
        os.makedirs(self.image_dir_path, exist_ok=True)

        self.label_dir_path = pjoin(self.result_dir_path, 'labels')
        os.makedirs(self.label_dir_path, exist_ok=True)

        self.yaml_path = pjoin(self.result_dir_path, 'data.yaml')

        self.result_labeled_dir_path = pjoin(self.proj_dir,
                                             f'{self._var.DATASET_FORMAT}-v{self._var.DATASET_VERSION}-labeled')

        if (os.path.exists(self.result_dir_path)
                and os.path.exists(self.image_dir_path)
                and os.path.exists(self.label_dir_path)):
            print('Successfully initialize save directory!')
            return True
        else:
            print('Something wrong happened!')
            return False

    def save_frame(self, frame, predicted_frames, cv2, labels_and_boxes, labeled_frame=None):
        """
        See iyolov8_export.py

        Returns:
            Predicted frame counter.
        """
        print(f'5.1. Condition met, processing valid frame: {predicted_frames}')
        # Save original frame
        unix_time = int(time.time())
        print("5.2. Saving frame, labels and boxes")
        cv2.imwrite(
            f'{self.image_dir_path}/{unix_time}.png',
            frame)

        if labeled_frame is not None:
            os.makedirs(self.result_labeled_dir_path, exist_ok=True)

            cv2.imwrite(
                f'{self.result_labeled_dir_path}/{unix_time}.png',
                labeled_frame)
        # Save labels and boxes
        with open(f'{self.label_dir_path}/{unix_time}.txt',
                  'w') as my_file:
            my_file.write(labels_and_boxes)

        # Increase the frame_number and predicted_frames by one.
        return predicted_frames + 1

    def create_yaml(self, project):
        """
        Create YAML configuration file with DATASET_FORMAT format.
        As convention, class names of YAML file is configured based on the first model

        Returns:
            None
        """
        model = project.models[0]

        label_names = [name for name in list(model.names.values())]
        with open(self.yaml_path, 'w') as my_file:
            content = 'names:\n'
            for name in label_names:
                content += f'- {name}\n'  # class mapping for helmet detection project
            content += f'path: ./\n'
            content += f'train: ./images\n'
            my_file.write(content)
