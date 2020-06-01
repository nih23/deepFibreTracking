"""class:

Cache
    Do not initalize, rather use get_cache() method.
    The cache location is specified in the Configuration

    methods:

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
    """Base class for Cache exceptions."""

    def __init__(self, msg=''):
        self.message = msg
        Exception.__init__(self, msg)

    def __repr__(self):
        return self.message

    __str__ = __repr__

class KeyNotCachedError(Error):
    """Error thrown if key isn't cached anymore."""

    def __init__(self, key):
        self.key = key
        Error.__init__(self, msg="""The key {} isn't cached (anymore).
                                Check if key is cached with in_cache(key).""")

class Cache():
    """The Cache Class to Use for caching torch data"""
    cache = None

    @classmethod
    def get_cache(cls):
        """Returns the currently active Cache instance.
        Use this instead of creating a new instance."""
        if not cls.cache:
            cls.cache = Cache(Config.get_config().get("cache", "cacheFolder", fallback="cache"))
        return cls.cache

    def __init__(self, path):
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
        """Saves the given tensor into the cache folder"""
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
        """Retrieves tensor from file. If keep_available is True, the element will stay in RAM"""
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
        """Returns wether key is currently cached"""
        return key in self.objects
    def remove(self, key):
        """Remove key from cache."""
        self.current_size -= self.objects[key]["size"]
        os.remove(os.path.join(self.path, self.objects[key]["filename"]))
        del self.objects[key]

    def _clean_cache(self):
        """Deletes files accordingly if cache size is exceeded.
           This is a LRU Cache, however, the number of elements removed is made minimal."""
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
        """Clears the whole cache."""
        keys = [key for key in self.objects]
        for key in keys:
            self.remove(key)

    def save_configuration(self):
        """Saves the cache to file. Is automatically called at exit."""
        config_file = os.path.join(self.path, "config.json")
        with open(config_file, 'w') as file:
            json.dump({"objects":self.objects, "current_size": self.current_size}, file)
