"Contains all exceptions for the cache submodule"
class KeyNotCachedError(Exception):
    """
    This error is thrown if the given key cannot be mapped to a data record.

    Attributes
    ----------
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
        super().__init__("""The key {} isn't cached (anymore).
                                Check if key is cached with in_cache(key).""")
