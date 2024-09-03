from utils.VariableClass import VariableClass
import time

# Initialize the VariableClass object, which contains all the necessary environment variables.
var = VariableClass()


def process_frame(frame, project, cv2=None, frames_out=''):
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
            return None, labels_and_boxes, None, total_time_class_prediction, False

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
    # + Have to satisfy the project.condition_func which defines custom condition logics for every specific project.
    if project.condition_func(total_results):
        for index, results in enumerate(total_results):
            # As a convention we will store all result labels under model1's
            # The other models' will be mapped accordingly
            if not combined_results:
                combined_results += [(box.xywhn, box.xyxy, box.cls, box.conf) for box in results.boxes]
            else:
                combined_results += [(box.xywhn, box.xyxy, project.map_to_first_model(index, box.cls), box.conf) for box
                                     in results.boxes]

        # sort results based on descending confidences
        sorted_combined_results = sorted(combined_results, key=lambda x: x[2], reverse=True)

        # Remove duplicates (if x and y coordinates of 2 boxes with the same class are < 0.01
        # -> consider as duplication and remove
        combined_results = []
        for element in sorted_combined_results:
            add_flag = True
            for res in combined_results:
                if res[2] == element[2]: # classes comparison
                    if (abs(res[0][0][0] - element[0][0][0]) < 0.01
                            and (abs(res[0][0][1] - element[0][0][1]) < 0.01)):
                        add_flag = False
            if add_flag:
                combined_results.append(element)

        # If the combined result has at least MIN_DETECTIONS boxes found (Could belong to either class)
        if len(combined_results) >= var.MIN_DETECTIONS:
            print("Condition met, we are gathering the labels and boxes and return results")
            # Crop frane to get only the interested area to reduce storage waste
            cropped_frame, cropped_coordinate = __crop_frame__(frame, combined_results)

            # <For testing> if you want to check if the labels
            # are transformed and applied correctly to the cropped frame -> uncomment the line below
            labeled_frame = None
            # labeled_frame = __get_labeled_frame__(cropped_frame, cropped_coordinate, cv2, combined_results)

            # Transform the labels and boxes accordingly
            labels_and_boxes = __transform_labels__(cropped_frame, cropped_coordinate, combined_results)
            total_time_class_prediction += time.time() - start_time_class_prediction
            return cropped_frame, labels_and_boxes, labeled_frame, total_time_class_prediction, True

    return None, labels_and_boxes, None, total_time_class_prediction, False


def __crop_frame__(frame, combined_results, padding=100):
    """
    Crop frame to get only the interesting area, meanwhile it removes the background that doesn't have any detection.

    Args:
        frame: The original frame to be processed.
        combined_results: List of results detected by models.
        padding: Add some space padding to the cropped frame to avoid object cutoff.
    """
    # If the combined result has at least MIN_DETECTIONS boxes found
    if len(combined_results) >= var.MIN_DETECTIONS:
        # Initialize bounding box limits
        x1_min, y1_min, x2_max, y2_max = float('inf'), float('inf'), float('-inf'), float('-inf')

        for _, xyxy, _, _ in combined_results:
            x1, y1, x2, y2 = xyxy[0]
            x1_min, y1_min = min(x1_min, x1), min(y1_min, y1)
            x2_max, y2_max = max(x2_max, x2), max(y2_max, y2)

        # Apply padding to the bounding box
        orig_height, orig_width = frame.shape[:2]
        x1_min = int(max(0, x1_min - padding))
        y1_min = int(max(0, y1_min - padding))
        x2_max = int(min(orig_width, x2_max + padding))
        y2_max = int(min(orig_height, y2_max + padding))

        # Crop the frame to the union bounding box with padding
        cropped_frame = frame[y1_min:y2_max, x1_min:x2_max]

        return cropped_frame, (x1_min, y1_min, x2_max, y2_max)


def __transform_labels__(cropped_frame, cropped_coordinate, combined_results):
    """
    Transform the labels and boxes coordinates to match with the cropped frame.

    Args:
        cropped_frame: The cropped frame to transform labels.
        cropped_coordinate: Cropped coordinate of the frame (in xyxy format)
        combined_results: List of results detected by models.
    """
    labels_and_boxes = ''
    frame_width, frame_height = cropped_frame.shape[:2]

    for _, xyxy, cls, conf in combined_results:
        x1, y1, x2, y2 = xyxy[0]
        x1, y1, x2, y2 = int(abs(x1 - cropped_coordinate[0])), int(abs(y1 - cropped_coordinate[1])), int(abs(x2 - cropped_coordinate[0])), int(abs(y2 - cropped_coordinate[1]))

        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2

        # Calculate the xywhn values (requirement for ultralytics YOLO models dataset)
        x_center_norm = x_center / frame_width
        y_center_norm = y_center / frame_height
        width_norm = (x2 - x1) / frame_width
        height_norm = (y2 - y1) / frame_height

        labels_and_boxes += f'{int(cls)} {x_center_norm} {y_center_norm} {width_norm} {height_norm}\n'

    return labels_and_boxes


def __get_labeled_frame__(cropped_frame, cropped_coordinate, cv2, combined_results):
    """
    <Used for testing if you want to see the labeled frame>
    Return the cropped frame with transformed labeled applied on the frame.

    Args:
        cropped_frame: The cropped frame to transform labels.
        cropped_coordinate: Cropped coordinate of the frame (in xyxy format)
        cv2: The Capture Video agent,
        combined_results: List of results detected by models.
    """
    labeled_frame = cropped_frame.copy()
    for _, xyxy, cls, _ in combined_results:
        x1, y1, x2, y2 = xyxy[0]
        x1, y1, x2, y2 = int(abs(x1 - cropped_coordinate[0])), int(abs(y1 - cropped_coordinate[1])), int(abs(x2 - cropped_coordinate[0])), int(abs(y2 - cropped_coordinate[1]))
        print(f"Box: {xyxy}, Class: {int(cls)}")
        print(f"Width: {x2 - x1} and height: {y2 - y1}")
        cv2.rectangle(labeled_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(labeled_frame, f'{int(cls)}', (x1 - 10, y1 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

    return labeled_frame
