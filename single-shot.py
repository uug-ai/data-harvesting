# This script is used to look for objects under a specific condition (at least 5 persons etc)
# The script reads a video from a message queue, classifies the objects in the video, and does a condition check.
# If condition is met, the video is being forwarded to a remote vault.
from integrations.integration_factory import IntegrationFactory
from projects.project_factory import ProjectFactory
from services.harvest_service import HarvestService
from integrations.roboflow_integration import RoboflowIntegration
# Local imports
from utils.VariableClass import VariableClass

# External imports
import os

import cv2

from utils.time_verbose_object import TimeVerbose

# Initialize the VariableClass object, which contains all the necessary environment variables.
var = VariableClass()


def init():
    harvest_service = HarvestService()
    model1, model2 = harvest_service.connect_models()

    project = ProjectFactory().init('helmet')

    # Mapping classes of 2 models
    mapping = project.class_mapping(model1, model2)
    integration = IntegrationFactory().init()

    # Open video-capture/recording using the video-path. Throw FileNotFoundError if cap is unable to open.
    cap = harvest_service.open_video()
    time_verbose = TimeVerbose()

    if var.LOGGING:
        print(f'5. Classifying frames')
    if var.TIME_VERBOSE:
        time_verbose.add_preprocessing_time()
    save_dir = harvest_service.process(
        cap,
        model1,
        model2,
        project.condition_func,
        mapping)

    if var.DATASET_UPLOAD:
        integration.upload_dataset(save_dir)

    if var.TIME_VERBOSE:
        time_verbose.add_preprocessing_time()

    # Depending on the TIME_VERBOSE parameter, the time it took to classify the objects is printed.
    if var.TIME_VERBOSE:
        time_verbose.show_result()

    if var.LOGGING:
        print('8) Releasing video writer and closing video capture')
        print("\n\n")

    # video_out.release() if var.SAVE_VIDEO else None
    # cap.release()
    # if var.PLOT:
    #     cv2.destroyAllWindows()


# Run the init function.
init()
