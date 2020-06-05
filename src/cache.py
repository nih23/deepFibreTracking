"""
The cache module covers all basic cache function necessary.

This can shorten computation times and therefore reduce stress and
increase the debuggability of the code.

Classes
-------
Error
    The base exception class for cache errors.
KeyNotCachedError
    Error thrown if key isn't cached.
Cache
    The cache class managing it.
"""
import os
import json
import base64
import time
import atexit

import torch
import numpy as np
from dipy.io.streamline import save_vtk_streamlines, load_vtk_streamlines
from src.config import Config

class Error(Exception):
    """
    Base class for Cache exceptions.

    Every Error happening from code of this class will inherit this one.
    The single parameter `msg` represents the error representing message.

    This class can be used to filter the exceptions for data exceptions.

    Attributes
    ----------
    message: str
        The message given is stored here.

    Examples
    --------

    >>> e = Error(msg='Just a sample message')
    >>> raise e from None
    Traceback (most recent call last):
    File "<stdin>", line 1, in <module>
    src.data.Error: Just a sample message
    """

    def __init__(self, msg=''):
        """
        Parameters
        ----------
        msg : str
            The message which accompanying the error, by default ''.
        """
        self.message = msg
        Exception.__init__(self, msg)

    def __repr__(self):
        return self.message

    __str__ = __repr__

class KeyNotCachedError(Error):
    """
    This error is thrown if the given key cannot be mapped to a data record.

    Attributes
    ----------
    message: str
        The error message.
    key: str
        The key the cache was unable to retrieve.
    """

    def __init__(self, key):
        """
        Parameters
        ----------
        key: str
            The key the cache was unable to retrieve.
        """
        self.key = key
        Error.__init__(self, msg="""The key {} isn't cached (anymore).
                                Check if key is cached with in_cache(key).""")

class Cache():
    """
    The cache can be used to cache tensors or streamlines in current implementation.

    Please do not create a new cache instance and instead use the Cache.get_cache()
    function to retrieve the correct instance to prevent corruption of files!

    The data is saved to files in path specified in configuration file and handled
    by the cache class. It is advised to use the methods for saving, retrieving and deleting
    cache information to allow the system to handle cache size and automatic deletion correctly.

    The cache always keeps its level below a maximum cache size specified in config.
    If the maximum size is exceeded, the oldest files will be deleted.
    However, the minimum number of files will be deleted.
    Therefore, for example if the oldest file is really small and wouldn't make a
    difference, it won't be removed.

    The cache will load the files directly on request. There is no online caching!
    Therefore, it is recommendable to keep often used values saved in your code.

    Attributes
    ----------
    path: str
        The path to the folder the cache data is saved in.
    objects: dict
        A dictionary containing the information about the saved data.
        The data itself is only loaded on request.
    current_size: int
        The current real cache size.

    Methods
    -------
    Cache.get_cache()
        Returns the currently used cache instance.
    set(key, tensor)
        Adds the tensor with key to the cache
    get(key)
        Retrieve tensor with key.
    in_cache(key)
        Returns wether key is currently in cache.
    remove(key)
        Removes key from cache.
    clear()
        Empties the cache
    save_configuration()
        Saves the current cache to file. Is automatically called on exit.
    """
    cache = None

    @classmethod
    def get_cache(cls):
        """
        Retrieve the current cache instance. If no instance exists, it is created.

        Returns
        -------
        Cache
            The current active and correct instance.
        """
        if not cls.cache:
            cls.cache = Cache(Config.get_config().get("cache", "cacheFolder", fallback="cache"))
        return cls.cache

    def __init__(self, path):
        """
        It is not recommended to call this function directly!
        Use get_cache() instead.
        Parameters
        ----------
        path : str
            The path to the cache path.
        """
        self.path = path.rstrip(os.path.sep) + os.path.sep
        os.makedirs(path, exist_ok=True)
        config_file = os.path.join(self.path, "config.json")

        if os.path.exists(config_file):
            with open(config_file) as file:
                json_file = json.load(file)
                self.objects = json_file["objects"]
                self.current_size = json_file["current_size"]

        else:
            self.objects = {}
            self.current_size = 0
        atexit.register(self.save_configuration)
        self._clean_cache()

    def set(self, key, tensor):
        """Saves the given tensor into the cache folder.

        Parameters
        ----------
        key : str
            The key to save the object to.
        tensor : torch.Tensor / dipy Streamlines
            The object which should be cached.
        """
        if "custom" in key:
            return
        if self.in_cache(key):
            self.current_size -= self.objects[key]["size"]
        suffix = ".dat"
        is_tensor = isinstance(tensor, torch.Tensor)
        is_streamlines = isinstance(tensor, list) and isinstance(tensor[0], np.ndarray)
        if is_tensor:
            suffix = ".pt"
        elif is_streamlines:
            suffix = ".vtk"
        filename = base64.urlsafe_b64encode(key.encode("UTF-8")).decode("UTF-8") + suffix
        filepath = os.path.join(self.path, filename)
        if is_tensor:
            torch.save(tensor, filepath)
        elif is_streamlines:
            save_vtk_streamlines(tensor, filepath)
        self.objects[key] = {"filename":filename, "size": os.path.getsize(filepath),
                             "last_accessed": int(time.time()*1000.0), "filetype":suffix[1:]}
        self.current_size += self.objects[key]["size"]
        self._clean_cache()
    def get(self, key):
        """Retrieves the object to given key and returns it.

        Parameters
        ----------
        key : str
            The key of the object.

        Returns
        -------
        torch.Tensor / dipy Streamlines
            The requested object.

        Raises
        ------
        KeyNotCachedError
            This exception is thrown if no object is assigned to given key.
        """
        if not self.in_cache(key):
            raise KeyNotCachedError(key)
        self.objects[key]["last_accessed"] = int(time.time()*1000.0)

        filename = self.objects[key]["filename"]
        filepath = os.path.join(self.path, filename)
        if self.objects[key]["filetype"] == "pt":
            tensor = torch.load(filepath)
        elif self.objects[key]["filetype"] == "vtk":
            tensor = load_vtk_streamlines(filepath)
        return tensor
    def in_cache(self, key):
        """Returns if key is currently cached.

        Parameters
        ----------
        key : str
            The key of the object.

        Returns
        -------
        bool
            A boolean indicating wether object is in cache or not.
        """
        return key in self.objects
    def remove(self, key):
        """Remove object with given key from cache.

        Parameters
        ----------
        key : str
            The key to remove.
        """
        self.current_size -= self.objects[key]["size"]
        os.remove(os.path.join(self.path, self.objects[key]["filename"]))
        del self.objects[key]

    def _clean_cache(self):
        """
        Deletes files accordingly if cache size is exceeded.
        This is a LRU Cache, however, the number of elements removed is made minimal.
        """
        cache_size = self.current_size
        max_size = Config.get_config().getint("cache", "maxCacheSize", fallback="10737418240")
        if cache_size <= max_size:
            return
        for key in sorted(self.objects, key=lambda k: self.objects[k]["last_accessed"]):
            #"Remove files" until neccessary size limit is small enough.
            cache_size -= self.objects[key]["size"]
            if cache_size < max_size:
                self.remove(key) # Then, remove this one last file
                break
        self._clean_cache() #Repeat if till size is small enough.

    def clear(self):
        """
        Clears the whole cache.
        """
        keys = [key for key in self.objects]
        for key in keys:
            self.remove(key)

    def save_configuration(self):
        """
        Saves the cache information to file.
        Is automatically called on exit.
        """
        config_file = os.path.join(self.path, "config.json")
        with open(config_file, 'w') as file:
            json.dump({"objects":self.objects, "current_size": self.current_size}, file)
