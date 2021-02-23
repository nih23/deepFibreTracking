# exception classes
class PathAlreadySetError(Exception):
    """
    This Exception is thrown if the Config is already initialized
    and your code tries to set the path.

    Attributes
    ----------
    path : str
        The path your code was trying to set.
    current_path : str
        The actual path the configuration was initialized with.
    """

    def __init__(self, path):
        """
        Parameters
        ----------
        path : str
            The path your code was trying to set.
        """

        self.path = path
        self.current_path = Config.get_config().get_path()
        super().__init__("""Path of config file already set to \"{}\".
                                Setting it to \"{}\" failed.""".format(self.current_path, path))