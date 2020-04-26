"""class:

Configuration Handler
    Do not initialize on your own, rather use the Config.get_config() function.
    The documentation is saved into ./config.ini and has the following format

    # Comment
    [Section]
    key = value

    There are two configuration options that directly influence the Config:
    - immutableConfiguration - decision wether the software is able to alter the config file
    - addDefaults - whether default values should be added to the config.ini, if required

    methods:

        Config.get_config()
            returns the currently used instance of Config.

        set(section, option, value=None)
            Sets a value in config, adds the section if not yet defined

        get(section, option, fallback=_UNSET)
            Returns value in config, if not possible returns fallback.
            If no fallback is defined, an error will be thrown.

        getint(section, option, fallback=_UNSET)
            Analogous to get(), however returns int

        getfloat(section, option, fallback=_UNSET)
            Analogous to get(), however returns float

        getboolean(section, option, fallback=_UNSET)
            Analogous to get(), however returns boolean. Also parses yes/no, 1/0 ...

        get_path()
            Returns the save path of the current config file

        save_configuration()
            Saves the loaded state of the configuration to file.
            However, this happens only if immutableConfiguration setting is set to false.
            The function is automatically called on exit.
"""


import configparser
import atexit


class Config():
    """The configuration class."""
    config = False
    _UNSET = configparser._UNSET
    @classmethod
    def get_config(cls):
        """Returns the currently active Config instance.
        Use this instead of creating a new instance to prevent corrupted config files."""
        if not cls.config:
            cls.config = Config()
        return cls.config


    def __init__(self, path="config.ini"):
        """Initalizes the Config file."""
        self.config = configparser.ConfigParser()
        self.config.optionxform = str
        self.config.read(path)
        self.is_immutable = self.config.getboolean("configuration", "immutableConfiguration",
                                                   fallback=False)
        self.path = path
        atexit.register(self.save_configuration)

    def _handle_add_default(self, section, option, fallback):
        """Adds fallback values to config, if option is not defined, if specified in it."""
        if (fallback is not self._UNSET and (not self.config.has_section(section)
                or not self.config.has_option(section, option))
                and self.config.getboolean("configuration", "addDefaults", fallback=True)):
            self.set(section, option, fallback)

    def set(self, section, option, value=None):
        """Set configuration option. Value has to be a string."""
        if not section in self.config:
            self.config[section] = {}
        return self.config.set(section, option, value)

    def get(self, section, option, fallback=_UNSET):
        """Get configuration option. Returns raw string."""
        self._handle_add_default(section, option, fallback)
        return self.config.get(section, option, fallback=fallback)
    def getint(self, section, option, fallback=_UNSET):
        """Get configuration option. Returns integer."""
        self._handle_add_default(section, option, fallback)
        return self.config.getint(section, option, fallback=fallback)
    def getfloat(self, section, option, fallback=_UNSET):
        """Get configuration option. Returns float."""
        self._handle_add_default(section, option, fallback)
        return self.config.getfloat(section, option, fallback=fallback)
    def getboolean(self, section, option, fallback=_UNSET):
        """Get configuration option. Returns boolean."""
        self._handle_add_default(section, option, fallback)
        return self.config.getboolean(section, option, fallback=fallback)


    def get_path(self):
        """Returns save path of config file."""
        return self.path

    def save_configuration(self):
        """Saves the current configuration to file, if config file isn't immutable."""
        if not self.is_immutable:
            with open(self.path, 'w') as configfile:
                self.config.write(configfile)
