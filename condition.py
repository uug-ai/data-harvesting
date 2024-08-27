from utils.TranslateObject import translate
from utils.VariableClass import VariableClass
import cv2
import time

# Initialize the VariableClass object, which contains all the necessary environment variables.
var = VariableClass()


# Function to process the frame.


def process_frame(MODEL, MODEL2, frame, condition_func, mapping, video_out='', frames_out=''):
    # Perform object classification on the frame.
    # persist=True -> The tracking results are stored in the model.
    # persist should be kept True, as this provides unique IDs for each detection.
    # More information about the tracking results via https://docs.ultralytics.com/reference/engine/results/

    total_time_class_prediction = 0
    if var.TIME_VERBOSE:
        start_time_class_prediction = time.time()

    # Execute the model
    results = MODEL.track(
        source=frame,
        persist=True,
        verbose=False,
        iou=var.IOU,
        conf=var.CLASSIFICATION_THRESHOLD,
        classes=var.MODEL_ALLOWED_CLASSES)
    results2 = None
    if MODEL2:
        results2 = MODEL2.track(
            source=frame,
            persist=True,
            verbose=False,
            iou=var.IOU,
            conf=var.CLASSIFICATION_THRESHOLD,
            classes=var.MODEL_2_ALLOWED_CLASSES)
        results2 = results2[0]

    if var.TIME_VERBOSE:
        total_time_class_prediction += time.time() - start_time_class_prediction

    # ###############################################
    # This is where the custom logic comes into play
    # ###############################################
    # Check if the results are not None,
    #  Otherwise, the postprocessing should not be done.
    # Iterate over the detected objects and their masks.
    results = results[0]  # Pick the first element since it returned a list of Result not the object itself

    annotated_frame = frame.copy()

    # Empty frame containing labels with bounding boxes
    labels_and_boxes = ''

    if results is not None or results2 is not None:
        combined_results = []

        # Check the condition to process frames
        # Since we have over 1k videos per day, the dataset we collect need to be high-quality
        # Valid image need to:
        # + Have at least MIN_DETECTIONS objects detected:
        # + Have to have helmet (since we are lacking of helmet dataset)
        if condition_func(results, results2, mapping):
            # Add labels and boxes of model 1 (add using mapping since we will store the label of model 2)
            combined_results += [(box.xywhn, mapping[int(box.cls)], box.conf) for box in results.boxes]

            # Add labels and boxes of model 2
            combined_results += [(box2.xywhn, box2.cls, box2.conf) for box2 in results2.boxes]

            # sort results based on descending confidences
            sorted_combined_results = sorted(combined_results, key=lambda x: x[2], reverse=True)

            # Remove duplicates (if x and y coordinates of 2 boxes with the same class are < 0.01
            # -> consider as duplication and remove
            combined_results = []
            for element in sorted_combined_results:
                add_flag = True
                for res in combined_results:
                    if res[1] == element[1]:
                        if (abs(res[0][0][0] - element[0][0][0]) < 0.01
                                and (abs(res[0][0][1] - element[0][0][1]) < 0.01)):
                            add_flag = False
                if add_flag:
                    combined_results.append(element)

        if len(combined_results) >= var.MIN_DETECTIONS:  # If the combined result has at least MIN_DETECTIONS boxes found (Could belong to either class)
            print("Condition met, we are gathering the labels and boxes and return results")
            for xywhn, cls, _ in combined_results:
                labels_and_boxes += f'{int(cls)} {xywhn[0, 0].item()} {xywhn[0, 1].item()} {xywhn[0, 2].item()} {xywhn[0, 3].item()}\n'
            return frame, total_time_class_prediction, True, labels_and_boxes

        # Annotate the frame with the classification objects.
        # Draw the class name and the confidence on the frame.
        if var.SAVE_VIDEO or var.PLOT:
            for box, mask in zip(results.boxes, results.masks or [None] * len(results.boxes)):
                # Translate the class name to a human-readable format and display it on the frame.
                object_name = translate(results.names[int(box.cls)])
                cv2.putText(
                    img=annotated_frame,
                    text=object_name,
                    org=(int(box.xyxy.tolist()[0][0]), int(
                        box.xyxy.tolist()[0][1]) - 10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(0, 255, 0),
                    thickness=2)

                # Draw the bounding box on the frame.
                cv2.rectangle(
                    img=annotated_frame,
                    pt1=(int(box.xyxy.tolist()[0][0]), int(
                        box.xyxy.tolist()[0][1])),
                    pt2=(int(box.xyxy.tolist()[0][2]), int(
                        box.xyxy.tolist()[0][3])),
                    color=(0, 255, 0),
                    thickness=2)

    # Depending on the SAVE_VIDEO or PLOT parameter, the frame is annotated.
    # This is done using a custom annotation function.
    # TODO: Fix this later (for some reasons code has error but vid is still saved)
    # if var.SAVE_VIDEO or var.PLOT:
    #
    #     # Show the annotated frame if the PLOT parameter is set to True.
    #     cv2.imshow("YOLOv8 Tracking",
    #                annotated_frame) if var.PLOT else None
    #     cv2.waitKey(1) if var.PLOT else None
    #
    #     # Write the annotated frame to the video-writer if the SAVE_VIDEO parameter is set to True.
    #     video_out.write(
    #         annotated_frame) if var.SAVE_VIDEO else None

    return frame, total_time_class_prediction, False, labels_and_boxes
