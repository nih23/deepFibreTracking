"""class:

Cache
    Do not initalize, rather use get_cache() method.
    The cache location is specified in the Configuration

    methods:

        Cache.get_cache()
            Returns the currently used cache instance.


"""

import os

from config import Config

class Cache():
    """The Cache Class to Use for caching torch data"""
    @classmethod
    def get_cache(cls):
        """Returns the currently active Cache instance.
        Use this instead of creating a new instance."""
        if not cls.cache:
            cls.cache = Cache(Config.get_config().get("cache", "path", fallback="cache/"))
        return cls.cache

    def __init__(self, path):
        self.path = path
        os.makedirs(path, exist_ok=True)
