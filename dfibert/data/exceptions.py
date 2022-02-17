class PointOutsideOfDWIError(LookupError):
    """
    Error thrown if given points are outside of the DWI-Image.
    This can be bypassed by passing `ignore_outside_points = True`
    to the raising function. However, it should be noted that this
    is not recommendable behaviour.
    Attributes
    ----------
    data_container : DataContainer
        The `DataContainer` whose DWI-Image is too small to cover the points.
    points: ndarray
        The point array which is responsible for raising the error.
    affected_points: ndarray
        The affected points beingn outside of the DWI-image.
    """

    def __init__(self, data_container, points, affected_points):
        """
        Parameters
        ----------
        data_container : DataContainer
            The `DataContainer` whose DWI-Image is too small to cover the points.
        points: ndarray
            The point array which is responsible for raising the error.
        affected_points: ndarray
            The affected points beingn outside of the DWI-image.
        """
        self.data_container = data_container
        self.points = points
        self.affected_points = affected_points
        super().__init__(("While parsing {no_points} points for further processing, "
                          "it became apparent that {aff} of the points "
                          "doesn't lay inside of DataContainer 'xyz'.")
                         .format(no_points=len(points), aff=affected_points))
