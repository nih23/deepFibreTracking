"Exceptions for the tracker submodule"
class StreamlinesAlreadyTrackedError(Exception):
    """Error thrown if streamlines are already tracked."""

    def __init__(self, tracker):
        self.tracker = tracker
        self.data_container = tracker.data_container
        super().__init__(("There are already {sl} streamlines tracked out of dataset {id}. "
                                  "Create a new Tracker object to change parameters.")
                       .format(sl=len(tracker.streamlines), id=self.data_container.id))

class ISMRMStreamlinesNotCorrectError(Exception):
    """Error thrown if streamlines are already tracked."""

    def __init__(self, tracker, path):
        self.tracker = tracker
        self.path = path
        super().__init__(("The streamlines located in {path} do not match the "
                                  "ISMRM 2015 Ground Truth Streamlines.").format(path=path))

class StreamlinesNotTrackedError(Exception):
    """Error thrown if streamlines weren't tracked yet."""

    def __init__(self, tracker):
        self.tracker = tracker
        self.data_container = tracker.data_container
        super().__init__( ("The streamlines weren't tracked yet from Dataset {id}. "
                                  "Call Tracker.track() to track the streamlines.")
                       .format(id=self.data_container.id))
