import numpy as np

class TractographyState:
    def __init__(self, coordinate, interpolFuncHandle):
        self.coordinate = coordinate
        self.interpolFuncHandle = interpolFuncHandle


    def getCoordinate(self):
        return self.coordinate


    def getValue(self):
        # interpolate DWI value at self.coordinate
        interpolatedDWI = self.interpolFuncHandle(self.coordinate)
        return interpolatedDWI