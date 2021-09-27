import numpy as np

class TractographyState:
    def __init__(self, coordinate, interpolFuncHandle):
        self.coordinate = coordinate
        self.interpolFuncHandle = interpolFuncHandle
        self.interpolatedDWI = None

    def getCoordinate(self):
        return self.coordinate


    def getValue(self):
        if self.interpolatedDWI is None:
        # interpolate DWI value at self.coordinate
            self.interpolatedDWI = self.interpolFuncHandle(self.coordinate)
        return self.interpolatedDWI
    
    def __add__(self, other):
        return self.getCoordinate() + other.getCoordinate()
        
        
    def __sub__(self, other):
        return self.getCoordinate() - other.getCoordinate()