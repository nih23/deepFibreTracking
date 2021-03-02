"""
The config module covers all functions handling the configuration file.
"""

import configparser
import atexit
from .exceptions import PathAlreadySetError


class Config():
    """
    The Configuration can be used to retrieve configuration parameters or to set them and
    their default values.

    Please do not create a new instance of this class and instead use the Config.get_config()
    function to ensure this is a singleton and therefore prevent file corruption,
    unexpected behaviour etc.

    The typical application of this Config is the following code

    >>> Config.get_config().getboolean("section", "parameter", fallback="no")
    False

    If the value didn't exist yet, it is automatically added correctly.

    The config file is retrieved from a specified location (default: 'config.ini')
    and has the following format:

    [Section]
    key = value

    There are two configuration options that directly influence the Config:
    - immutableConfiguration - decision wether the software is able to alter the config file
    - addDefaults - whether default values should be added to the config.ini, if required


    Attributes
    ----------
    config: ConfigParser
        The real config parser this class is based on.
        It is not advisable to use it directly, use the wrapper functions instead for
        correct behaviour.
    is_immutable: bool
        The bool indicating wether a change of the configuration is possible in software or not.
    path: str
        The current path of the active cache.
    """

    config = None
    _UNSET = configparser._UNSET
    _CONFIGURATION_FILE = "config.ini"
    @classmethod
    def get_config(cls):
        """
        Retrieve the current active and correct instance. If no instance exists, it is created.
        Use this instead of creating a new instance to prevent corrupted config files.

        Returns
        -------
        Config
            The current active config.
        """
        if not cls.config:
            cls.config = Config(path=cls._CONFIGURATION_FILE)
        return cls.config
    @classmethod
    def set_path(cls, path):
        """
        Set the path of config file to use. Only callable ahead of get_config.
        Therefore, call it at the start of your code.

        Parameters
        ----------
        path : str
            The path to set.

        Raises
        ------
        PathAlreadySetError
            Error is thrown if the configuration is already initialized.
        """
        if cls.config:
            raise PathAlreadySetError(path) from None
        cls._CONFIGURATION_FILE = path

    def __init__(self, path):
        """
        Do not call this directly! Instead use Config.get_config()

        Parameters
        ----------
        path : str
            The path the config uses to load.
        """
        self.config = configparser.ConfigParser()
        self.config.optionxform = str
        self.config.read(path)

        if (not self.config.has_section("configuration") or
                not self.config.has_option("configuration", "immutableConfiguration")):
            self.set("configuration", "immutableConfiguration", "no")
        if not self.config.has_option("configuration", "addDefaults"):
            self.set("configuration", "addDefaults", "yes")

        self.is_immutable = self.config.getboolean("configuration", "immutableConfiguration")
        self.path = path
        atexit.register(self.save_configuration)

    def _handle_add_default(self, section, option, fallback):
        """
        Adds fallback values correctly into config, if allowed.

        Parameters
        ----------
        section : str
            The section of this option.
        option : str
            The option name of this option.
        fallback : str
            The fallback value of this option as string, regardless of real type.
        """
        if (fallback is not self._UNSET and (not self.config.has_section(section)
                                             or not self.config.has_option(section, option))
                and self.config.getboolean("configuration", "addDefaults")):
            self.set(section, option, fallback)

    def set(self, section, option, value=None):
        """Sets the configuration option with specified parameters.

        `value` has to be a string.

        Parameters
        ----------
        section : str
            The section of the item.
        option : str
            The name of the option.
        value : str, optional
            The value of the option, by default None.
        """
        if not section in self.config:
            self.config[section] = {}
        self.config.set(section, option, value)

    def get(self, section, option, fallback=_UNSET):
        """Gets the configuration option with specified parameters.

        `fallback` has to be a string.

        Parameters
        ----------
        section : str
            The section of the item.
        option : str
            The name of the option.
        fallback : str, optional
            The fallback value if not set.

        Returns
        -------
        str
            The option to get
        """
        self._handle_add_default(section, option, fallback)
        return self.config.get(section, option, fallback=fallback)
    def getint(self, section, option, fallback=_UNSET):
        """Gets the configuration option with specified parameters as integer.

        `fallback` has to be a string.

        Parameters
        ----------
        section : str
            The section of the item.
        option : str
            The name of the option.
        fallback : str, optional
            The fallback value if not set.

        Returns
        -------
        int
            The option to retrieve as int.
        """
        self._handle_add_default(section, option, fallback)
        return self.config.getint(section, option, fallback=fallback)
    def getfloat(self, section, option, fallback=_UNSET):
        """Gets the configuration option with specified parameters as float.

        `fallback` has to be a string.

        Parameters
        ----------
        section : str
            The section of the item.
        option : str
            The name of the option.
        fallback : str, optional
            The fallback value if not set.

        Returns
        -------
        float
            The option to retrieve as float.
        """
        self._handle_add_default(section, option, fallback)
        return self.config.getfloat(section, option, fallback=fallback)
    def getboolean(self, section, option, fallback=_UNSET):
        """Gets the configuration option with specified parameters as boolean.

        `fallback` has to be a string.

        Parameters
        ----------
        section : str
            The section of the item.
        option : str
            The name of the option.
        fallback : str, optional
            The fallback value if not set.

        Returns
        -------
        bool
            The option to retrieve as boolean.
        """
        self._handle_add_default(section, option, fallback)
        return self.config.getboolean(section, option, fallback=fallback)


    def get_path(self):
        """Returns the path of the configuration file.

        Returns
        -------
        str
            The path.
        """
        return self.path

    def save_configuration(self):
        """
        Saves the current configuration to file, if config file isn't immutable.

        This is automatically called on exit.
        """
        if not self.is_immutable:
            with open(self.path, 'w') as configfile:
                self.config.write(configfile)
