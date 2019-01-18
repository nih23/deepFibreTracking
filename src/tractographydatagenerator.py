import numpy as np
import keras
import src.dwi_tools as dwi_tools
import time
import os
import os.path

class TractographyDataGenerator(keras.utils.Sequence):

    def __init__(self, dwi, streamlines, affine, list_IDs, batch_size=32, dim=(3,3,3), n_channels=1,rotateTrainingData=True,
                 n_target_coordinates=3, shuffle=False, storeTemporaryData = False, endpointPrediction = True):
        'Initialization'
        self.pTempData = '/data/nico/tmp/' + str(time.time()) + '/'
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
        self.storeTemporaryData = storeTemporaryData
        self.keepZeroVectors = False
        self.endpointPredictionData = endpointPrediction
        
        if(self.endpointPredictionData):
            print('keeping zero vectors due to endpoint prediction')
            self.keepZeroVectors = True
        
        os.makedirs(self.pTempData)
        
        self.on_epoch_end()

        
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

            
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        streamlines_batch = [self.streamlines[i] for i in list_IDs_temp]
        interpolatedDWISubvolume, directionToPreviousStreamlinePoint, directionToNextStreamlinePoint = dwi_tools.generateTrainingData(streamlines_batch, self.dwi, unitTension = self.unitTangent, affine=self.affine, noX=self.dim[0],noY=self.dim[1],noZ=self.dim[2],coordinateScaling=self.coordinateScaling,distToNeighbours=1, noCrossings = 1, step = self.step, rotateTrainingData = self.rotateTrainingData)
        
        if(self.endpointPredictionData):
            # predict random data along our streamlines with zero directions
            interpolatedDWISubvolumeEP, _, directionToNextStreamlinePointEP = dwi_tools.generateTrainingData(streamlines_batch, self.dwi, unitTension = self.unitTangent, affine=self.affine, noX=self.dim[0],noY=self.dim[1],noZ=self.dim[2],coordinateScaling=self.coordinateScaling,distToNeighbours=1, noCrossings = 1, step = self.step, rotateTrainingData = self.rotateTrainingData, generateRandomData = True)
            
            interpolatedDWISubvolume = np.concatenate((interpolatedDWISubvolume,interpolatedDWISubvolumeEP))
            directionToNextStreamlinePoint = np.concatenate((directionToNextStreamlinePoint,directionToNextStreamlinePointEP))
            
            noSamples = len(interpolatedDWISubvolume)
            labels = np.zeros((noSamples,1))
            vN = np.sqrt(np.sum(directionToNextStreamlinePoint ** 2 , axis = 1)) 
            idx1 = np.where(vN > 0)[0] # idx of non-zero directions
            labels[idx1] = 1
            
            return interpolatedDWISubvolume, [directionToNextStreamlinePoint, np.squeeze(labels)]
        
        if(self.keepZeroVectors == False):
            vN = np.sqrt(np.sum(directionToNextStreamlinePoint ** 2 , axis = 1))
            idx1 = np.where(vN > 0)[0]
            vN = np.sqrt(np.sum(directionToNextStreamlinePoint ** 2 , axis = 1))
            idx2 = np.where(vN > 0)[0]
            s2 = set(idx2)
            idxNoZeroVectors = [val for val in idx1 if val in s2]
            
            interpolatedDWISubvolume = interpolatedDWISubvolume[idxNoZeroVectors,...]
            directionToNextStreamlinePoint = directionToNextStreamlinePoint[idxNoZeroVectors,]
        
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
        pTmpX = '%sx_%d_%d.npy' % (self.pTempData,list_IDs_temp[0],list_IDs_temp[-1])
        pTmpY = '%sy_%d_%d.npy' % (self.pTempData,list_IDs_temp[0],list_IDs_temp[-1])
        
        if(os.path.isfile(pTmpX)):
            #print('Loading ' + pTmpX)
            X = np.load(pTmpX)
            if(self.endpointPredictionData):
                y1 = np.load(pTmpY+"-0")
                y2 = np.load(pTmpY+"-1")
                y = [y1,y2]
            else:
                y = np.load(pTmpY)
        else:
            # Generate data
            X, y = self.__data_generation(list_IDs_temp)
            if(self.storeTemporaryData):
                np.save(pTmpX,X)
                if(self.endpointPredictionData):
                    np.save(pTmpY+"-0",y[0])
                    np.save(pTmpY+"-1",y[1])
                else:
                    np.save(pTmpY,y)

        return X, y
    
class TwoDimensionalTractographyDataGenerator(keras.utils.Sequence):

    def __init__(self, dwi, streamlines, affine, list_IDs, batch_size=32, dim=(3,3,3), n_channels=1,rotateTrainingData=1,
                 n_target_coordinates=3, shuffle=False, storeTemporaryData = False, keepZeroVectors = False):
        'Initialization'
        self.pTempData = '/data/nico/tmp/' + str(time.time()) + '/'
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
        self.storeTemporaryData = storeTemporaryData
        self.keepZeroVectors = keepZeroVectors
        
        os.makedirs(self.pTempData)
        
        self.on_epoch_end()

        
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

            
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        streamlines_batch = [self.streamlines[i] for i in list_IDs_temp]
        interpolatedDWISubvolume, directionToPreviousStreamlinePoint, directionToNextStreamlinePoint = dwi_tools.generateTrainingData(streamlines_batch, self.dwi, unitTension = self.unitTangent, affine=self.affine, noX=self.dim[0],noY=self.dim[1],noZ=self.dim[2],coordinateScaling=self.coordinateScaling,distToNeighbours=1, noCrossings = 1, step = self.step, rotateTrainingData = self.rotateTrainingData)
        
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
        pTmpX = '%sx_%d_%d.npy' % (self.pTempData,list_IDs_temp[0],list_IDs_temp[-1])
        pTmpY = '%sy_%d_%d.npy' % (self.pTempData,list_IDs_temp[0],list_IDs_temp[-1])
        
        if(os.path.isfile(pTmpX)):
            X = np.load(pTmpX)
            y = np.load(pTmpY)
        else:
            # Generate data
            X, y = self.__data_generation(list_IDs_temp)
            if(self.storeTemporaryData):
                np.save(pTmpX,X)
                np.save(pTmpY,y)

        X = np.reshape(X,[-1,8,8,1]) # we might have problems in case of nx,ny,nz > 1
        return X, y
    
class ThreeDimensionalTractographyDataGenerator(keras.utils.Sequence):

    def __init__(self, dwi, streamlines, affine, list_IDs, batch_size=32, dim=(3,3,3), n_channels=1,rotateTrainingData=1,
                 n_target_coordinates=3, shuffle=False, storeTemporaryData = False, keepZeroVectors = False):
        'Initialization'
        self.pTempData = '/data/nico/tmp/' + str(time.time()) + '/'
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
        self.storeTemporaryData = storeTemporaryData
        self.keepZeroVectors = keepZeroVectors
        
        os.makedirs(self.pTempData)
        
        self.on_epoch_end()

        
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

            
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        streamlines_batch = [self.streamlines[i] for i in list_IDs_temp]
        interpolatedDWISubvolume, directionToPreviousStreamlinePoint, directionToNextStreamlinePoint = dwi_tools.generateTrainingData(streamlines_batch, self.dwi, unitTension = self.unitTangent, affine=self.affine, noX=self.dim[0],noY=self.dim[1],noZ=self.dim[2],coordinateScaling=self.coordinateScaling,distToNeighbours=1, noCrossings = 1, step = self.step, rotateTrainingData = self.rotateTrainingData)
        
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
        pTmpX = '%sx_%d_%d.npy' % (self.pTempData,list_IDs_temp[0],list_IDs_temp[-1])
        pTmpY = '%sy_%d_%d.npy' % (self.pTempData,list_IDs_temp[0],list_IDs_temp[-1])
        
        if(os.path.isfile(pTmpX)):
            X = np.load(pTmpX)
            y = np.load(pTmpY)
        else:
            # Generate data
            X, y = self.__data_generation(list_IDs_temp)
            if(self.storeTemporaryData):
                np.save(pTmpX,X)
                np.save(pTmpY,y)
        X = np.reshape(X,[-1,self.dim[0]*8,self.dim[1]*8,self.dim[2],1]) # we might have problems in case of nx,ny,nz > 1
        return X, y
