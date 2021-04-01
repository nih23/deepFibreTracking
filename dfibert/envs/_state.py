"Contains the state for the tractography gym environment"
class TractographyState:
    "The state for the tractography gym environment"
    def __init__(self, coordinate, interpolation_func):
        self.coordinate = coordinate
        self.interpolation_func = interpolation_func
        self.interpolated_dwi = None

    def get_coordinate(self):
        "Returns the coordinate of the state"
        return self.coordinate


    def get_value(self):
        "Returns the state value - the interpolated dwi"
        if self.interpolated_dwi is None:
        # interpolate DWI value at self.coordinate
            self.interpolated_dwi = self.interpolation_func(self.coordinate)
        return self.interpolated_dwi
