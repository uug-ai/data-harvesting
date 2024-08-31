from utils.VariableClass import VariableClass
import time

# Initialize the VariableClass object, which contains all the necessary environment variables.
var = VariableClass()


def process_frame(frame, project, video_out='', frames_out=''):
    # Perform object classification on the frame.
    # persist=True -> The tracking results are stored in the model.
    # persist should be kept True, as this provides unique IDs for each detection.
    # More information about the tracking results via https://docs.ultralytics.com/reference/engine/results/

    total_time_class_prediction = 0
    labels_and_boxes = ''
    if var.TIME_VERBOSE:
        start_time_class_prediction = time.time()

    total_results = []
    for model, allowed_classes in zip(project.models, project.models_allowed_classes):
        # Execute every model in the list
        cur_results = model.track(
            source=frame,
            persist=True,
            verbose=False,
            iou=var.IOU,
            conf=var.CLASSIFICATION_THRESHOLD,
            classes=allowed_classes,
            device=project.device)

        if var.TIME_VERBOSE:
            total_time_class_prediction += time.time() - start_time_class_prediction

        if len(cur_results[0]) == 0:
            return frame, total_time_class_prediction, False, labels_and_boxes

        total_results.append(cur_results[0])

    # ###############################################
    # This is where the custom logic comes into play
    # ###############################################
    # Check if the results are not None,
    #  Otherwise, the postprocessing should not be done.
    # Iterate over the detected objects and their masks.
    annotated_frame = frame.copy()
    combined_results = []

    # Check the condition to process frames
    # Since we have over 1k videos per day, the dataset we collect need to be high-quality
    # Valid image need to:
    # + Have at least MIN_DETECTIONS objects detected:
    # + Have to have helmet (since we are lacking of helmet dataset)
    if project.condition_func(total_results):
        for index, results in enumerate(total_results):
            # As a convention we will store all result labels under model1's
            # The other models' will be mapped accordingly
            if not combined_results:
                combined_results += [(box.xywhn, box.cls, box.conf) for box in results.boxes]
            else:
                combined_results += [(box.xywhn, project.map_to_first_model(index, box.cls), box.conf) for box in results.boxes]

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

        # If the combined result has at least MIN_DETECTIONS boxes found (Could belong to either class)
        if len(combined_results) >= var.MIN_DETECTIONS:
            print("Condition met, we are gathering the labels and boxes and return results")
            for xywhn, cls, _ in combined_results:
                labels_and_boxes += f'{int(cls)} {xywhn[0, 0].item()} {xywhn[0, 1].item()} {xywhn[0, 2].item()} {xywhn[0, 3].item()}\n'
            return frame, total_time_class_prediction, True, labels_and_boxes

    return frame, total_time_class_prediction, False, labels_and_boxes
