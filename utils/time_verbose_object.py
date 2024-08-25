import time


class TimeVerbose:
    """
    Time Verbose class tracks and reports the time spent in different stages of a process,
    including preprocessing, processing, and postprocessing.
    """

    def __init__(self):
        """
        Constructor.
        """
        self.start_time = time.time()
        self.total_time_preprocessing = 0
        self.total_time_class_prediction = 0
        self.total_time_processing = 0
        self.total_time_postprocessing = 0
        self.start_time_preprocessing = time.time()

    def add_preprocessing_time(self):
        """
        Adds the time spent in preprocessing to the total time and resets the
        start time for the next preprocessing segment.
        """
        self.total_time_preprocessing += time.time() - self.start_time_preprocessing
        self.start_time_preprocessing = time.time()

    def show_result(self):
        """
        Prints a detailed breakdown of the time spent on different stages.
        """
        print(
            f'\t - Classification took: {round(time.time() - self.start_time, 1)} seconds.')
        print(
            f'\t\t - {round(self.total_time_preprocessing, 2)}s for preprocessing and initialization')
        print(
            f'\t\t - {round(self.total_time_processing, 2)}s for processing of which:')
        print(
            f'\t\t\t - {round(self.total_time_class_prediction, 2)}s for class prediction')
        print(
            f'\t\t\t - {round(self.total_time_processing - self.total_time_class_prediction, 2)}s for other processing')
        print(
            f'\t\t - {round(self.total_time_postprocessing, 2)}s for postprocessing')
