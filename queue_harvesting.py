# This script is used to look for objects under a specific condition (at least 5 persons etc)
# The script reads a video from a message queue, classifies the objects in the video, and does a condition check.
# If condition is met, the video is being forwarded to a remote vault.
from integrations.integration_factory import IntegrationFactory
from projects.project_factory import ProjectFactory
from services.harvest_service import HarvestService
from utils.VariableClass import VariableClass
from utils.time_verbose_object import TimeVerbose

# Initialize the VariableClass object, which contains all the necessary environment variables.
var = VariableClass()


def init():
    # Service and Project initializations
    harvest_service = HarvestService()
    harvest_service.connect('rabbitmq', 'kerberos_vault')
    model1, model2 = harvest_service.connect_models()

    project = ProjectFactory().init('helmet')
    # Mapping classes of 2 models
    mapping = project.class_mapping(model1, model2)
    integration = IntegrationFactory().init()

    while True:
        # Receive message from the queue,
        # and retrieve the media from the Kerberos Vault utilizing the message information.
        message = harvest_service.receive_message()
        if message is None:
            continue  # No message received, continue to the next iteration

        media_key, provider = message['payload']['key'], message['source']

        time_verbose = TimeVerbose()
        cap = harvest_service.open_video(message)

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

        # We might remove the recording from the vault after analyzing it. (default is False)
        # This might be the case if we only need to create a dataset from the recording and do not need to store it.
        # Delete the recording from Kerberos Vault if the REMOVE_AFTER_PROCESSED is set to True.
        harvest_service.delete_media(media_key, provider)

        if var.TIME_VERBOSE:
            time_verbose.add_preprocessing_time()

        # Depending on the TIME_VERBOSE parameter, the time it took to classify the objects is printed.
        if var.TIME_VERBOSE:
            time_verbose.show_result()

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
