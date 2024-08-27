from uugai_python_dynamic_queue.MessageBrokers import RabbitMQ
from uugai_python_kerberos_vault.KerberosVault import KerberosVault

from exports.export_factory import ExportFactory
from services.iharvest_service import IHarvestService
from utils.VariableClass import VariableClass
import time
import requests
from ultralytics import YOLO
import torch
import os
import cv2
from condition import process_frame as con_process_frame


class HarvestService(IHarvestService):
    """
    HarvestService class responsible for handling tasks such as connecting to
    RabbitMQ and Kerberos Vault, receiving and processing messages, downloading
    and opening video files, and processing video frames using YOLO models.
    """

    def __init__(self):
        """
        Constructor.
        """
        self.rabbitmq = None
        self.vault = None
        self.frame_number = 0
        self.predicted_frames = 0
        self.max_frame_number = None
        self.frame_skip_factor = 0
        # Initialize the VariableClass object, which contains all the necessary environment variables.
        self._var = VariableClass()
        self._save_format = ExportFactory().init(proj_name='helmet_detection')

    def connect(self, *agents):
        """
        See iharvest_service.py
        """
        if 'rabbitmq' not in agents and 'kerberos_vault' not in agents:
            raise TypeError('Missing agent!')

        if self._var.LOGGING:
            print('a) Initializing RabbitMQ')

        # Initialize a message broker using the python_queue_reader package
        self.rabbitmq = RabbitMQ(
            queue_name=self._var.QUEUE_NAME,
            target_queue_name=self._var.TARGET_QUEUE_NAME,
            exchange=self._var.QUEUE_EXCHANGE,
            host=self._var.QUEUE_HOST,
            username=self._var.QUEUE_USERNAME,
            password=self._var.QUEUE_PASSWORD)

        if self._var.LOGGING:
            print('b) Initializing Kerberos Vault')

        self.vault = KerberosVault(
            storage_uri=self._var.STORAGE_URI,
            storage_access_key=self._var.STORAGE_ACCESS_KEY,
            storage_secret_key=self._var.STORAGE_SECRET_KEY)

    def receive_message(self):
        """
        See iharvest_service.py

        Returns:
            dict or None: The received message if available, otherwise None.
        """
        # Receive message from the queue,
        # and retrieve the media from the Kerberos Vault utilizing the message information.
        if self._var.LOGGING:
            print('1) Receiving message from RabbitMQ')
        message = self.rabbitmq.receive_message()
        if not message:
            if self._var.LOGGING:
                print('No message received, waiting for 3 seconds')
            time.sleep(3)
            return None
        if self._var.LOGGING:
            print('2) Retrieving media from Kerberos Vault')

        self.vault.retrieve_media(
            message=message,
            media_type='video',
            media_savepath=self._var.MEDIA_SAVEPATH)
        return message

    def delete_media(self, media_key, provider):
        """
        See iharvest_service.py
        """
        if self._var.REMOVE_AFTER_PROCESSED:
            # Delete the recording from Kerberos Vault
            response = requests.delete(
                self._var.STORAGE_URI + '/storage',
                headers={
                    'X-Kerberos-Storage-FileName': media_key,
                    'X-Kerberos-Storage-Provider': provider,
                    'X-Kerberos-Storage-AccessKey': self._var.STORAGE_ACCESS_KEY,
                    'X-Kerberos-Storage-SecretAccessKey': self._var.STORAGE_SECRET_KEY,
                }
            )
            if response.status_code != 200:
                print(
                    "Something went wrong while delete media: " + response.content)
            else:
                print("Delete media from " + self._var.STORAGE_URI)

    def connect_models(self):
        """
        See iharvest_service.py
        """
        _cur_dir = os.getcwd()
        # initialise the yolo model, additionally use the device parameter to specify the device to run the model on.
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = YOLO(self._var.MODEL_NAME).to(device)
        model2 = YOLO(self._var.MODEL_NAME_2).to(device)
        if model and model2:
            print(f'2. Using device: {device}')
            print(f'3. Using models: {self._var.MODEL_NAME} and  {self._var.MODEL_NAME_2}')
            return model, model2
        else:
            raise ModuleNotFoundError('Something wrong happened!')

    def open_video(self, message=''):
        """
        See iharvest_service.py

        Returns:
            cv2.VideoCapture: The video capture object for the opened video.
        """
        if message:
            # Download video from vault if there is a message
            self.__download_video__(message)

        # Open video-capture/recording using the video-path. Throw FileNotFoundError if cap is unable to open.
        if self._var.LOGGING:
            print(f'4. Opening video file: {self._var.MEDIA_SAVEPATH}')
        if not os.path.exists(self._var.MEDIA_SAVEPATH):
            raise FileNotFoundError(f'Cannot find {self._var.MEDIA_SAVEPATH}')
        if not self._var.MEDIA_SAVEPATH.lower().endswith(('.mp4', '.avi', '.mov')):
            raise TypeError('Unsupported file format! Only support videos with .mp4, .avi, .mov extensions')
        cap = cv2.VideoCapture(self._var.MEDIA_SAVEPATH)
        if not cap.isOpened():
            raise FileNotFoundError('Unable to open video file')
        # Initialize the video-writer if the SAVE_VIDEO is set to True.
        # if self._var.SAVE_VIDEO:
        #     fourcc = cv2.VideoWriter.fourcc(*'avc1')
        #     video_out = cv2.VideoWriter(
        #         filename=self._var.OUTPUT_MEDIA_SAVEPATH,
        #         fourcc=fourcc,
        #         fps=self._var.CLASSIFICATION_FPS,
        #         frameSize=(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        #                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        #     )
        self.frame_number = 0
        self.predicted_frames = 0
        self.max_frame_number = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.frame_skip_factor = int(
            cap.get(cv2.CAP_PROP_FPS) / self._var.CLASSIFICATION_FPS)
        return cap

    def process(self, cap, model1, model2, condition_func, mapping):
        if self.max_frame_number > 0:
            skip_frames_counter = 0

            # Create save dir and yaml file
            success = self._save_format.initialize_save_dir()
            if success and self._var.DATASET_FORMAT == 'roboflow':
                self._save_format.create_yaml(model2)

            while (self.predicted_frames < self._var.MAX_NUMBER_OF_PREDICTIONS) and (
                    self.frame_number < self.max_frame_number):
                # Read the frame from the video-capture.
                success, frame, skip_frames_counter = self.get_frame(cap, skip_frames_counter)
                # Increment frame number after processing
                self.frame_number += 1

                if not success:
                    break

                if frame is None:
                    continue

                # Process frame
                skip_frames_counter = self.predict_frame(
                    model1,
                    model2,
                    frame,
                    condition_func,
                    mapping,
                    skip_frames_counter)
        return self._save_format.result_dir_path

    def get_frame(self, cap: cv2.VideoCapture, skip_frames_counter):
        """
        See iharvest_service.py

        Returns:
            tuple: A tuple containing a boolean indicating success, the frame (or None),
                   and the updated skip frames counter.
        """
        # Check if we need to skip the current frame due to the skip_frames_counter.
        if skip_frames_counter > 0:
            return True, None, skip_frames_counter - 1

        success, frame = cap.read()
        if not success:
            return False, None, skip_frames_counter

        return True, frame, skip_frames_counter

    def predict_frame(
            self,
            model1,
            model2,
            frame,
            condition_func,
            mapping,
            skip_frames_counter):
        """
        See iharvest_service.py

        Returns:
            int: The updated skip frames counter.
        """
        if self.frame_number > 0 and self.frame_skip_factor > 0 and self.frame_number % self.frame_skip_factor == 0:
            frame, total_time_class_prediction, condition_met, labels_and_boxes = con_process_frame(
                model1, model2, frame, condition_func, mapping)

            if condition_met:
                self.predicted_frames = self._save_format.save_frame(frame, self.predicted_frames, cv2,
                                                                     labels_and_boxes)
                skip_frames_counter = 50
        print(f'Currently in frame: {self.frame_number}')
        self.frame_number += 1
        return skip_frames_counter

    def __download_video__(self, message):
        """
        Downloads the video from Kerberos Vault using the provided message details.

        Args:
            message: The message containing details required to retrieve the video.
        """
        self.vault.retrieve_media(
            message=message,
            media_type='video',
            media_savepath=self._var.MEDIA_SAVEPATH)
        print(f'Video downloaded under {self._var.MEDIA_SAVEPATH}')
