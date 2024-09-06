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
    def register(self, name, value_obj):
        """
        Registers a module (project, integration, or export format) with the specified name and value object.

        This method dynamically assigns a value object to one of the attributes (`project`, `integration`, or `export`)
        based on the provided `name`.

        Args:
            name: The name of the module to register. Must be one of 'project', 'integration', or 'export'.
            value_obj: value of the module to register.
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
    def evaluate(self, video):
        """
        Process input video, perform model prediction logic for every frame.

        Args
            video: Input video.
        """
        pass

    @abstractmethod
    def __get_frame__(self, cap, skip_frames_counter):
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

    @abstractmethod
    def __predict_frame__(self, frame, skip_frames_counter):
        """
        Predict input frame.

        Args:
            frame: Input frame to be predicted.
            skip_frames_counter: Skipped frame counter (used when condition in 1 frame is met, skip x next frames).
        """
        pass
