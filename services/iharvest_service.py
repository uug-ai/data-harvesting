from abc import ABC, abstractmethod


class IHarvestService(ABC):
    """
    Interface for Harvest Service
    """

    @abstractmethod
    def connect(self, *agents):
        """
        Connects to the required agents, specifically RabbitMQ and Kerberos Vault.

        Args:
            agents (tuple): A tuple containing the names of agents to connect to.
                            Must include 'rabbitmq' and/or 'kerberos_vault'.

        Raises:
            TypeError: If neither 'rabbitmq' nor 'kerberos_vault' is included in agents.
        """
        pass

    @abstractmethod
    def receive_message(self):
        """
        Receives a message from RabbitMQ and retrieves the corresponding media
        from Kerberos Vault.

        """
        pass

    @abstractmethod
    def delete_media(self, media_key, provider):
        """
        Deletes the processed recording from Kerberos Vault if
        REMOVE_AFTER_PROCESSED is set to True.

        Args:
            media_key: The key of the media to delete from the vault.
            provider: The provider information for the media in the vault.
        """
        pass

    @abstractmethod
    def connect_models(self):
        """
        Initializes the YOLO models and connects them to the appropriate device (CPU or GPU).

        Returns:
            tuple: A tuple containing two YOLO models.

        Raises:
            ModuleNotFoundError: If the models cannot be loaded.
        """
        pass

    @abstractmethod
    def open_video(self, message=''):
        """
        Opens a video file from the specified path, downloading it from the vault if necessary.

        Args:
            message: The message to use for downloading the video. Defaults to ''.

        Raises:
            FileNotFoundError: If the video file cannot be found or opened.
            TypeError: If the video file format is unsupported.
        """
        pass

    @abstractmethod
    def get_frame(self, cap, skip_frames_counter):
        """
        Retrieves the next frame from the video capture object, potentially skipping frames.

        Args:
            cap (cv2.VideoCapture): The video capture object.
            skip_frames_counter (int): The number of frames to skip.

        Returns:
            tuple: A tuple containing a boolean indicating success, the frame (or None),
                   and the updated skip frames counter.
        """
        pass
