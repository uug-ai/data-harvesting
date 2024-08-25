# This script is used to look for objects under a specific condition (at least 5 persons etc)
# The script reads a video from a message queue, classifies the objects in the video, and does a condition check.
# If condition is met, the video is being forwarded to a remote vault.

from integrations.roboflow_helper import RoboflowHelper
# Local imports
from projects.project_factory import ProjectFactory
from services.harvest_service import HarvestService
from utils.VariableClass import VariableClass

# External imports
import os
import cv2

from utils.time_verbose_object import TimeVerbose

# Initialize the VariableClass object, which contains all the necessary environment variables.
var = VariableClass()


def init():
    # Service and Project initializations
    harvest_service = HarvestService()
    harvest_service.connect('rabbitmq', 'kerberos_vault')
    model1, model2 = harvest_service.connect_models()

    project = ProjectFactory().init('helmet')

    while True:
        # Receive message from the queue,
        # and retrieve the media from the Kerberos Vault utilizing the message information.
        message = harvest_service.receive_message()
        if message is None:
            continue  # No message received, continue to the next iteration

        media_key, provider = message['payload']['key'], message['source']

        time_verbose = TimeVerbose()
        cap = harvest_service.open_video(message)

        # frame_number -> The current frame number. Depending on the frame_skip_factor this can make jumps.
        # predicted_frames -> The number of predicted frames that satisfies the condition.
        # frame_skip_factor -> the factor by which the input video frames are skipped.
        frame_number, predicted_frames = 0, 0
        frame_skip_factor = int(
            cap.get(cv2.CAP_PROP_FPS) / var.CLASSIFICATION_FPS)

        # Loop over the video frames, and perform object classification.
        # The classification process is done until the counter reaches the MAX_NUMBER_OF_PREDICTIONS is reached.
        MAX_FRAME_NUMBER = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if var.LOGGING:
            print(f'5. Classifying frames')
        if var.TIME_VERBOSE:
            time_verbose.add_preprocessing_time()

        skip_frames_counter = 0

        result_dir_path, image_dir_path, label_dir_path, yaml_path = project.create_result_save_dir()

        # Mapping classes of 2 models
        mapping = project.class_mapping(model1, model2)

        while (predicted_frames < var.MAX_NUMBER_OF_PREDICTIONS) and (frame_number < MAX_FRAME_NUMBER):
            # Read the frame from the video-capture.
            success, frame, skip_frames_counter = harvest_service.get_video_frame(cap, skip_frames_counter)

            if success and frame is None:
                continue
            if not success:
                break

            # Process frame
            skip_frames_counter = harvest_service.process_frame(
                frame_skip_factor,
                skip_frames_counter,
                model1,
                model2,
                project.condition_func,
                mapping,
                result_dir_path,
                image_dir_path,
                label_dir_path,
                frame,
                None)

        # Upload to roboflow after processing frames if any
        project.upload_dataset(result_dir_path, yaml_path, model2)

        # We might remove the recording from the vault after analyzing it. (default is False)
        # This might be the case if we only need to create a dataset from the recording and do not need to store it.
        # Delete the recording from Kerberos Vault if the REMOVE_AFTER_PROCESSED is set to True.
        harvest_service.process_from_vault(media_key, provider)

        if var.TIME_VERBOSE:
            time_verbose.add_preprocessing_time()

        # Depending on the TIME_VERBOSE parameter, the time it took to classify the objects is printed.
        if var.TIME_VERBOSE:
            time_verbose.show_result()
        # If the videowriter was active, the videowriter is released.
        # Close the video-capture and destroy all windows.
        if var.LOGGING:
            print('8) Releasing video writer and closing video capture')
            print("\n\n")

        # TODO: CARE AB THIS
        # video_out.release() if var.SAVE_VIDEO else None
        # cap.release()
        # if var.PLOT:
        #     cv2.destroyAllWindows()


# Run the init function.
init()
