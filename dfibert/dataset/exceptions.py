class WrongDatasetTypePassedError(Exception):
    """Error thrown if `ConcatenatedDataset` retrieves wrong datasets.

    This means that the datasets you passed aren't exclusively IterableDatasets

    Attributes
    ----------
    caller: ConcatenatedDataset
        The ConcatenatedDataset raising the error.
    dataset: BaseDataset
        The dataset causing the error.
    """

    def __init__(self, concat, dataset, message):
        """
        Parameters
        ----------
        concat : ConcatenatedDataset
            The dataset raising the error.
        dataset : BaseDataset
            The dataset responsible for the error.
        message : str
            Your specific error message.
        """
        self.caller = concat
        self.dataset = dataset
        super().__init__(message)

class FeatureShapesNotEqualError(Exception):
    """Error thrown if FeatureShapes of `ConcatenatedDataset` are not equal, but requested.

    This error only occurs, if `ConcatenatedDataset().get_feature_shape` is called,
    and the Datasets in the ConcatenatedDataset doesn't have equal feature shapes.

    Attributes
    ----------
    index: int
        The index of the BaseDataset responsible for the error.
    shape1: tuple
        The shape of the reference dataset.
    shape2: tuple
        The shape of the different dataset.
    """
    def __init__(self, index, s1, s2):
        """
        Parameters
        ----------
        index : int
            The index of the BaseDataset responsible for the error.
        s1 : tuple
            The shape of the reference dataset.
        s2 : tuple
            The shape of the dataset causing the error.
        """
        self.shape1 = s1
        self.shape2 = s2
        self.index = index
        super().__init__(("The shape of the dataset {idx} ({s2}) "
                          "is not equal to the base shape of the reference dataset 0 ({s1})"
                          ).format(idx=index, s2=s2, s1=s1))