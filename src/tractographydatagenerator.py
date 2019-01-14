import numpy as np
import keras
import src.dwi_tools as dwi_tools

class TractographyDataGenerator(keras.utils.Sequence):

    def __init__(self, dwi, streamlines, affine, list_IDs, batch_size=32, dim=(3,3,3), n_channels=1,rotateTrainingData=1,
                 n_target_coordinates=3, shuffle=True):
        'Initialization'
        self.dwi = dwi
        self.streamlines = streamlines
        self.affine = affine
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_target_coordinates = n_target_coordinates
        self.shuffle = shuffle
        
        self.unitTangent = 0 # dont normalize tangent
        self.step = 1 # mm
        self.coordinateScaling = 1 # scaling factor
        self.rotateTrainingData = rotateTrainingData
        
        self.on_epoch_end()

        
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

            
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        streamlines_batch = [self.streamlines[i] for i in list_IDs_temp]
        interpolatedDWISubvolume, directionToPreviousStreamlinePoint, directionToNextStreamlinePoint, interpolatedDWISubvolumePast = dwi_tools.generateTrainingData(streamlines_batch, self.dwi, unitTension = self.unitTangent, affine=self.affine, noX=self.dim[0],noY=self.dim[1],noZ=self.dim[2],coordinateScaling=self.coordinateScaling,distToNeighbours=1, noCrossings = 1, step = self.step, rotateTrainingData = self.rotateTrainingData)
        
        return interpolatedDWISubvolume, directionToNextStreamlinePoint

    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y