# This script is used to look for objects under a specific condition (at least 5 persons etc)
# The script reads a video from a message queue, classifies the objects in the video, and does a condition check.
# If condition is met, the video is being forwarded to a remote vault.
import ultralytics.nn.tasks

from roboflow_helper import RoboflowHelper
# Local imports
from utils.VariableClass import VariableClass
from condition import processFrame

# External imports
import os
from os.path import (
    join as pjoin,
    splitext as psplitext,
    basename as pbasename)
from datetime import datetime
import cv2
import time
import torch
from ultralytics import YOLO

# Initialize the VariableClass object, which contains all the necessary environment variables.
var = VariableClass()


def init():

    # Initialize the time variables.
    start_time = time.time()
    total_time_preprocessing = 0
    total_time_class_prediction = 0
    total_time_processing = 0
    total_time_postprocessing = 0
    start_time_preprocessing = time.time()

    # Perform object classification on the media
    # initialise the yolo model, additionally use the device parameter to specify the device to run the model on.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    MODEL = YOLO(var.MODEL_NAME).to(device)
    MODEL2 = None
    if var.MODEL_NAME_2:
        MODEL2 = YOLO(var.MODEL_NAME_2).to(device)
    if var.LOGGING:
        print(f'3) Using device: {device}')

    # Open video-capture/recording using the video-path. Throw FileNotFoundError if cap is unable to open.
    if var.LOGGING:
        print(f'4) Opening video file: {var.MEDIA_SAVEPATH}')
    if not os.path.exists(var.MEDIA_SAVEPATH):
        raise FileNotFoundError(f'Cannot find {var.MEDIA_SAVEPATH}')
    if not var.MEDIA_SAVEPATH.lower().endswith(('.mp4', '.avi', '.mov')):
        raise TypeError('Unsupported file format! Only support videos with .mp4, .avi, .mov extensions')
    cap = cv2.VideoCapture(var.MEDIA_SAVEPATH)
    if not cap.isOpened():
        raise FileNotFoundError('Unable to open video file')
    video_out = None
    # Initialize the video-writer if the SAVE_VIDEO is set to True.
    if var.SAVE_VIDEO:
        fourcc = cv2.VideoWriter.fourcc(*'avc1')
        video_out = cv2.VideoWriter(
            filename=var.OUTPUT_MEDIA_SAVEPATH,
            fourcc=fourcc,
            fps=var.CLASSIFICATION_FPS,
            frameSize=(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                       int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        )

    # Initialize the classification process.
    # 2 lists are initialized:
        # Classification objects
        # Additional list for easy access to the ids.

    # frame_number -> The current frame number. Depending on the frame_skip_factor this can make jumps.
    # predicted_frames -> The number of frames, that were used for the prediction. This goes up by one each prediction iteration.
    # frame_skip_factor is the factor by which the input video frames are skipped.
    frame_number, predicted_frames = 0, 0
    frame_skip_factor = int(cap.get(cv2.CAP_PROP_FPS) / var.CLASSIFICATION_FPS)

    # Loop over the video frames, and perform object classification.
    # The classification process is done until the counter reaches the MAX_NUMBER_OF_PREDICTIONS or the last frame is reached.
    MAX_FRAME_NUMBER = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if var.LOGGING:
        print(f'5) Classifying frames')
    if var.TIME_VERBOSE:
        total_time_preprocessing += time.time() - start_time_preprocessing
        start_time_processing = time.time()

    skip_frames_counter = 0

    result_dir_path = pjoin(pjoin(os.getcwd(), 'out_data'), f'{datetime.now().strftime("%d-%m-%Y_%H-%M-%S")}')
    image_dir_path = pjoin(result_dir_path, 'images')
    label_dir_path = pjoin(result_dir_path, 'labels')
    yaml_path = pjoin(result_dir_path, 'data.yaml')

    while (predicted_frames < var.MAX_NUMBER_OF_PREDICTIONS) and (frame_number < MAX_FRAME_NUMBER):

        # Read the frame from the video-capture.
        success, frame = cap.read()
        if not success:
            break

        # Check if we need to skip the current frame due to the skip_frames_counter.
        if skip_frames_counter > 0:
            skip_frames_counter -= 1
            frame_number += 1
            continue

        # Check if the frame_number corresponds to a frame that should be classified.
        if frame_number > 0 and frame_skip_factor > 0 and frame_number % frame_skip_factor == 0:
            frame, total_time_class_prediction, condition_met, labels_and_boxes = processFrame(
                MODEL, MODEL2, frame, video_out, result_dir_path)

            # Create new directory to save frames, labels and boxes for when the first frame met the condition
            if predicted_frames == 0 and condition_met:
                os.makedirs(f'{image_dir_path}', exist_ok=True)
                os.makedirs(f'{label_dir_path}', exist_ok=True)
            if condition_met:
                print(f'Processing frame: {predicted_frames}')
                # Save original frame
                cv2.imwrite(f'{image_dir_path}/{psplitext(pbasename(var.MEDIA_SAVEPATH))[0]}_f{predicted_frames}.png', frame)
                print("Saving frame, labels and boxes")
                # Save labels and boxes
                with open(f'{label_dir_path}/{psplitext(pbasename(var.MEDIA_SAVEPATH))[0]}_f{predicted_frames}.txt', 'w') as my_file:
                    my_file.write(labels_and_boxes)

                # Set the skip_frames_counter to 50 to skip the next 50 frames.
                skip_frames_counter = 50

                # Increase the frame_number and predicted_frames by one.
                predicted_frames += 1

        frame_number += 1

    # Create yaml file afterward
    label_names = [name for name in list(MODEL.names.values())]
    create_yaml(yaml_path, label_names)

    # Upload to roboflow after processing frames if any
    if os.path.exists(result_dir_path) and var.RBF_UPLOAD:
        rb = RoboflowHelper()
        if rb:
            rb.upload_dataset(result_dir_path)
    else:
        print('Nothing to upload!!')

    if var.TIME_VERBOSE:
        total_time_processing += time.time() - start_time_processing

    # Depending on the TIME_VERBOSE parameter, the time it took to classify the objects is printed.
    if var.TIME_VERBOSE:
        print(
            f'\t - Classification took: {round(time.time() - start_time, 1)} seconds, @ {var.CLASSIFICATION_FPS} fps.')
        print(
            f'\t\t - {round(total_time_preprocessing, 2)}s for preprocessing and initialisation')
        print(
            f'\t\t - {round(total_time_processing, 2)}s for processing of which:')
        print(
            f'\t\t\t - {round(total_time_class_prediction, 2)}s for class prediction')
        print(
            f'\t\t\t - {round(total_time_processing - total_time_class_prediction, 2)}s for other processing')
        print(
            f'\t\t - {round(total_time_postprocessing, 2)}s for postprocessing')
        print(f'\t - Original video: {round(cap.get(cv2.CAP_PROP_FRAME_COUNT)/cap.get(cv2.CAP_PROP_FPS), 1)} seconds, @ {round(cap.get(cv2.CAP_PROP_FPS), 1)} fps @ {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}. File size of {round(os.path.getsize(var.MEDIA_SAVEPATH)/1024**2, 1)} MB')

    # If the videowriter was active, the videowriter is released.
    # Close the video-capture and destroy all windows.
    if var.LOGGING:
        print('8) Releasing video writer and closing video capture')
        print("\n\n")

    video_out.release() if var.SAVE_VIDEO else None
    cap.release()
    if var.PLOT:
        cv2.destroyAllWindows()

def create_yaml(file_path, label_names):
    with open(file_path, 'w') as my_file:
        content ='names:\n'
        for name in label_names:
            content += f'- {name}\n' # class mapping for helmet dection project
        content += f'nc: {len(label_names)}'
        my_file.write(content)

# Run the init function.
init()
