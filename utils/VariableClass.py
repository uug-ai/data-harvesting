import os
from dotenv import load_dotenv


class VariableClass:
    """This class is used to store all the environment variables in a single class. This is done to make it easier to access the variables in the code.

    """

    def __init__(self):
        """This function is used to load all the environment variables and store them in the class.

        """
        # Environment variables
        load_dotenv()

        # Model parameters
        self.DATASET_FORMAT = os.getenv("DATASET_FORMAT")
        self.DATASET_VERSION = os.getenv("DATASET_VERSION")
        self.DATASET_UPLOAD = os.getenv("DATASET_UPLOAD") == "True"

        # Queue parameters
        self.QUEUE_NAME = os.getenv("QUEUE_NAME")
        self.TARGET_QUEUE_NAME = os.getenv("TARGET_QUEUE_NAME")
        self.QUEUE_EXCHANGE = os.getenv("QUEUE_EXCHANGE")
        self.QUEUE_HOST = os.getenv("QUEUE_HOST")
        self.QUEUE_USERNAME = os.getenv("QUEUE_USERNAME")
        self.QUEUE_PASSWORD = os.getenv("QUEUE_PASSWORD")

        # Kerberos Vault parameters
        self.STORAGE_URI = os.getenv("STORAGE_URI")
        self.STORAGE_ACCESS_KEY = os.getenv("STORAGE_ACCESS_KEY")
        self.STORAGE_SECRET_KEY = os.getenv("STORAGE_SECRET_KEY")

        # Feature parameters
        self.PROJECT_NAME = os.getenv("PROJECT_NAME")

        # The == "True" is used to convert the string to a boolean.
        self.TIME_VERBOSE = os.getenv("TIME_VERBOSE") == "True"

        self.LOGGING = os.getenv("LOGGING") == "True"

        self.CREATE_BBOX_FRAME = os.getenv("CREATE_BBOX_FRAME") == "True"
        self.SAVE_BBOX_FRAME = os.getenv("SAVE_BBOX_FRAME") == "True"
        self.BBOX_FRAME_SAVEPATH = os.getenv("BBOX_FRAME_SAVEPATH")
        self.REMOVE_AFTER_PROCESSED = os.getenv("REMOVE_AFTER_PROCESSED") == "False"
        if self.SAVE_BBOX_FRAME:
            self.CREATE_BBOX_FRAME = True

        self.CREATE_RETURN_JSON = os.getenv("CREATE_RETURN_JSON") == "True"
        self.SAVE_RETURN_JSON = os.getenv("SAVE_RETURN_JSON") == "True"
        self.RETURN_JSON_SAVEPATH = os.getenv("RETURN_JSON_SAVEPATH")
        if self.SAVE_RETURN_JSON:
            self.CREATE_RETURN_JSON = True

        self.FIND_DOMINANT_COLORS = os.getenv("FIND_DOMINANT_COLORS") == "True"
        if os.getenv("COLOR_PREDICTION_INTERVAL") is not None:
            self.COLOR_PREDICTION_INTERVAL = int(
                os.getenv("COLOR_PREDICTION_INTERVAL"))
        if os.getenv("MIN_CLUSTERS") is not None:
            self.MIN_CLUSTERS = int(os.getenv("MIN_CLUSTERS"))
        if os.getenv("MAX_CLUSTERS") is not None:
            self.MAX_CLUSTERS = int(os.getenv("MAX_CLUSTERS"))

        # Classification parameters
        if os.getenv("CLASSIFICATION_FPS") is not None and os.getenv("CLASSIFICATION_FPS") != "":
            self.CLASSIFICATION_FPS = int(os.getenv("CLASSIFICATION_FPS", "15"))
        if os.getenv("CLASSIFICATION_THRESHOLD") is not None and os.getenv("CLASSIFICATION_THRESHOLD") != "":
            self.CLASSIFICATION_THRESHOLD = float(
                os.getenv("CLASSIFICATION_THRESHOLD"))
        if os.getenv("MAX_NUMBER_OF_PREDICTIONS") is not None and os.getenv("CLASSIFICATION_FPS") != "":
            self.MAX_NUMBER_OF_PREDICTIONS = int(
                os.getenv("MAX_NUMBER_OF_PREDICTIONS", "50"))
        if os.getenv("MIN_DISTANCE") is not None and os.getenv("MIN_DISTANCE") != "":
            self.MIN_DISTANCE = int(os.getenv("MIN_DISTANCE", "500"))
        if os.getenv("MIN_STATIC_DISTANCE") is not None and os.getenv("MIN_STATIC_DISTANCE") != "":
            self.MIN_STATIC_DISTANCE = int(
                os.getenv("MIN_STATIC_DISTANCE", "100"))
        if os.getenv("MIN_DETECTIONS") is not None and os.getenv("MIN_DETECTIONS") != "":
            self.MIN_DETECTIONS = int(os.getenv("MIN_DETECTIONS", "5"))
        self.FRAMES_SKIP_AFTER_DETECT = int(os.getenv("FRAMES_SKIP_AFTER_DETECT", "50"))
        self.IOU = float(os.getenv("IOU", "0.85"))

        # Integration parameters
        self.INTEGRATION_NAME = os.getenv("INTEGRATION_NAME")

        # Roboflow parameters
        self.ROBOFLOW_API_KEY = os.getenv("RBF_API_KEY")
        self.ROBOFLOW_WORKSPACE = os.getenv("RBF_WORKSPACE")
        self.ROBOFLOW_PROJECT = os.getenv("RBF_PROJECT")

        # S3 parameters
        self.S3_ENDPOINT = os.getenv("S3_ENDPOINT")
        self.S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY")
        self.S3_SECRET_KEY = os.getenv("S3_SECRET_KEY")
        self.S3_BUCKET = os.getenv("S3_BUCKET")
