from uugai_python_dynamic_queue.MessageBrokers import RabbitMQ
from uugai_python_kerberos_vault.KerberosVault import KerberosVault

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
        # Initialize the VariableClass object, which contains all the necessary environment variables.
        self._var = VariableClass()

    def connect(self, *agents):
        """
        Connects to the required agents, specifically RabbitMQ and Kerberos Vault.

        Args:
            agents (tuple): A tuple containing the names of agents to connect to.
                            Must include 'rabbitmq' and/or 'kerberos_vault'.

        Raises:
            TypeError: If neither 'rabbitmq' nor 'kerberos_vault' is included in agents.
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
        Receives a message from RabbitMQ and retrieves the corresponding media
        from Kerberos Vault.

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

    def process_from_vault(self, media_key, provider):
        """
        Deletes the processed recording from Kerberos Vault if
        REMOVE_AFTER_PROCESSED is set to True.

        Args:
            media_key: The key of the media to delete from the vault.
            provider: The provider information for the media in the vault.
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
        Initializes the YOLO models and connects them to the appropriate device (CPU or GPU).

        Returns:
            tuple: A tuple containing two YOLO models.

        Raises:
            ModuleNotFoundError: If the models cannot be loaded.
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
        Opens a video file from the specified path, downloading it from the vault if necessary.

        Args:
            message: The message to use for downloading the video. Defaults to ''.

        Returns:
            cv2.VideoCapture: The video capture object for the opened video.

        Raises:
            FileNotFoundError: If the video file cannot be found or opened.
            TypeError: If the video file format is unsupported.
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
        return cap

    def get_video_frame(self, cap: cv2.VideoCapture, skip_frames_counter):
        """
        Retrieves the next frame from the video capture object, potentially skipping frames.

        Args:
            cap (cv2.VideoCapture): The video capture object.
            skip_frames_counter (int): The number of frames to skip.

        Returns:
            tuple: A tuple containing a boolean indicating success, the frame (or None),
                   and the updated skip frames counter.
        """
        # Check if we need to skip the current frame due to the skip_frames_counter.
        if skip_frames_counter > 0:
            skip_frames_counter -= 1
            self.frame_number += 1
            return True, None, skip_frames_counter
        success, frame = cap.read()
        if not success:
            return False, None, skip_frames_counter
        return True, frame, skip_frames_counter

    def process_frame(self, frame_skip_factor, skip_frames_counter, model1, model2, condition_func, mapping, result_dir_path, image_dir_path, label_dir_path, frame, video_out):
        """
        Processes a single video frame, potentially skipping frames, running YOLO models,
        and saving frames and labels if a condition is met.

        Args:
            frame_skip_factor: The factor determining how often to process frames.
            skip_frames_counter: The counter for how many frames to skip.
            model1 (YOLO): The 1st YOLO model.
            model2 (YOLO): The 2nd YOLO model.
            condition_func (callable): The function that checks the condition for frame processing.
            mapping: The mapping used in the condition function.
            result_dir_path: The directory path for saving results.
            image_dir_path: The directory path for saving images.
            label_dir_path: The directory path for saving labels.
            frame: The current video frame.
            video_out: The video writer object for saving video.

        Returns:
            int: The updated skip frames counter.
        """
        if self.frame_number > 0 and frame_skip_factor > 0 and self.frame_number % frame_skip_factor == 0:
            frame, total_time_class_prediction, condition_met, labels_and_boxes = con_process_frame(
                model1, model2, frame, condition_func, mapping, video_out, result_dir_path)

            # Create new directory to save frames, labels and boxes for when the first frame met the condition
            if self.predicted_frames == 0 and condition_met:
                os.makedirs(f'{image_dir_path}', exist_ok=True)
                os.makedirs(f'{label_dir_path}', exist_ok=True)
            if condition_met:
                print(f'5.1. Processing frame: {self.predicted_frames}')
                # Save original frame
                unix_time = int(time.time())
                cv2.imwrite(
                    f'{image_dir_path}/{unix_time}.png',
                    frame)
                print("5.2. Saving frame, labels and boxes")
                # Save labels and boxes
                with open(f'{label_dir_path}/{unix_time}.txt',
                          'w') as my_file:
                    my_file.write(labels_and_boxes)

                # Set the skip_frames_counter to 50 to skip the next 50 frames.
                skip_frames_counter = 50

                # Increase the frame_number and predicted_frames by one.
                self.predicted_frames += 1
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