
from utils.TranslateObject import translate
from utils.VariableClass import VariableClass
import cv2
import time

# Initialize the VariableClass object, which contains all the necessary environment variables.
var = VariableClass()

# Function to process the frame.


def processFrame(MODEL, MODEL2, frame, video_out='', frames_out=''):
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
        iou=0.85,
        conf=var.CLASSIFICATION_THRESHOLD)
    results2 = None
    if MODEL2:
        results2 = MODEL2.track(
            source=frame,
            persist=True,
            verbose=False,
            iou=0.85,
            conf=var.CLASSIFICATION_THRESHOLD)
        results2 = results2[0]

    if var.TIME_VERBOSE:
        total_time_class_prediction += time.time() - start_time_class_prediction

    # ###############################################
    # This is where the custom logic comes into play
    # ###############################################
    # Check if the results are not None,
    #  Otherwise, the postprocessing should not be done.
    # Iterate over the detected objects and their masks.
    results = results[0] # Pick the first element since it returned a list of Result not the object itself

    annotated_frame = frame.copy()

    # Empty frame containing labels with bounding boxes
    labels_and_boxes = ''

    # if results is not None:
    #     # Using the results of the classification, we can verify if we have a condition met.
    #     # We can look for example for people who are:
    #     # - not wearing a helmet,
    #     # - people with a blue shirt,
    #     # - cars driving in the opposite direction,
    #     # - etc.
    #     # You are in the driving seat so you can write your custom code to detect the condition
    #     # you are looking for.
    #     if len(results.boxes) >= var.MIN_DETECTIONS: # If there are at least 5 boxes found (Could belong to either class)
    #         print("Condition met, we are gathering the labels and boxes and return results")
    #         # Extract label and boxes from result in YOLOv8 format
    #         for cls_item, xywhn_item in zip(results.boxes.cls.tolist(), results.boxes.xywhn):
    #             labels_and_boxes = labels_and_boxes + f'{int(cls_item)} {xywhn_item[0]} {xywhn_item[1]} {xywhn_item[2]} {xywhn_item[3]}\n'
    #
    #         return frame, total_time_class_prediction, True, labels_and_boxes
    #     else:
    #         print("Condition not met, skipping frame")

    if results is not None or results2 is not None:
        combined_results = []

        # Check the condition to process frames
        # Since we have over 1k videos per day, the dataset we collect need to be high-quality
        # Valid image need to:
        # + Have at least MIN_DETECTIONS objects detected:
        # + Have to have helmet (since we are lacking of helmet dataset)
        # + Number of helmet and person detected are equal (make sure every person wearing a helmet is detected)
        if (len(results.boxes) > 0
                and len(results2.boxes) > 0
                and (any(box.cls == 1 for box in results2.boxes)
                     or any(box.cls == 2 for box in results.boxes))
                and sum(box.cls == 1 for box in results.boxes) == sum(box.cls == 2 for box in results.boxes)):
            for box1, box2 in zip(results.boxes, results2.boxes):
                if box1.cls == box2.cls:
                    avg_conf = (box1.conf + box2.conf) / 2
                    if box1.conf >= box2.conf:
                        combined_results.append((box1.xywhn, box1.cls, avg_conf))
                    else:
                        combined_results.append((box2.xywhn, box2.cls, avg_conf))

            # Add any remaining boxes from model 1 or model 2 if their counts are different
            combined_results += [(box.xywhn, box.cls, box.conf) for box in results.boxes[len(combined_results):]]
            combined_results += [(box.xywhn, box.cls, box.conf) for box in results2.boxes[len(combined_results):]]

        if len(combined_results) >= var.MIN_DETECTIONS:  # If the combined result has at least 5 boxes found (Could belong to either class)
            print("Condition met, we are gathering the labels and boxes and return results")
            for xywhn, cls, conf in combined_results:
                labels_and_boxes += f'{int(cls[0])} {xywhn[0, 0].item()} {xywhn[0, 1].item()} {xywhn[0, 2].item()} {xywhn[0, 3].item()}\n'
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
    if var.SAVE_VIDEO or var.PLOT:

        # Show the annotated frame if the PLOT parameter is set to True.
        cv2.imshow("YOLOv8 Tracking",
                   annotated_frame) if var.PLOT else None
        cv2.waitKey(1) if var.PLOT else None

        # Write the annotated frame to the video-writer if the SAVE_VIDEO parameter is set to True.
        video_out.write(
            annotated_frame) if var.SAVE_VIDEO else None

    return frame, total_time_class_prediction, False, labels_and_boxes
