from dipy.tracking.streamline import values_from_volume
import dipy.align.vector_fields as vfu
from dipy.core.sphere import Sphere
from dipy.core import subdivide_octahedron 
from scipy.spatial import KDTree
from dipy.core import subdivide_octahedron 

from dipy.tracking.local import LocalTracking
from dipy.viz import window, actor
from dipy.viz.colormap import line_colors
from dipy.tracking.streamline import Streamlines
from dipy.tracking import metrics

import numpy as np
import warnings

import src.nn_helper as nn_helper

import tensorflow as tf
from joblib import Parallel, delayed
import multiprocessing

def simpleDeterministicTracking(seeds, data, model, rec_level_sphere = 3, noX=3, noY=3,noZ=3,dw=288,coordinateSpacing = 0.1, stepWidth = 0.5):
        #with tf.device('/cpu:0'):       
            streamlines = list()
            x_ = coordinateSpacing * np.linspace(-1., 1., noX)
            y_ = coordinateSpacing * np.linspace(-1., 1., noY)
            z_ = coordinateSpacing * np.linspace(-1., 1., noZ)       
            
            noSeeds = 100

            # global tracking loop
            for seedIndex in range(0,noSeeds):
                streamLinePosition = seeds[seedIndex]
                streamLinePosition_old = streamLinePosition
            
                curStreamline = []
                curStreamline.append(streamLinePosition)
                # tracking loop for a single fibre
                for i in range(0,500):
                    # interpolate data given these coordinates for each channel
                    x = np.zeros([noX,noY,noZ,dw])

                    coordVecs = np.vstack(np.meshgrid(x_,y_,z_)).reshape(3,-1).T + streamLinePosition
                    for i in range(0,dw):
                        x[:,:,:,i] = np.reshape(vfu.interpolate_scalar_3d(data[:,:,:,i],coordVecs)[0], [noX,noY,noZ])

                    lastDirection = streamLinePosition - streamLinePosition_old

                    x = nn_helper.normalize(x)
                    x_ext = np.expand_dims(x, axis=0)
                    nextProbableDirections = model.predict(x_ext)

                    # should we terminate?
                    #TODO: implement me

                    # compute next streamline position
                    probs = nextProbableDirections.dot(lastDirection / 255)
                    idx_map = np.argmin(abs(probs))
                    nextDirection = np.squeeze(nextProbableDirections[:,idx_map,:])
                    streamLinePosition_old = streamLinePosition
                    streamLinePosition = streamLinePosition + stepWidth * nextDirection
                    curStreamline.append(streamLinePosition)

                streamlines.append(curStreamline)
            return streamlines
        
        
def simpleParallelDeterministicTracking(seeds, data, model, noX=3, noY=3,noZ=3,dw=288,coordinateSpacing = 0.1, stepWidth = 0.5):
    x_ = coordinateSpacing * np.linspace(-1., 1., noX)
    y_ = coordinateSpacing * np.linspace(-1., 1., noY)
    z_ = coordinateSpacing * np.linspace(-1., 1., noZ)       

    noSeeds = len(seeds)
    noSeeds = 50
    noIterations = 1000

    streamlinePositions = np.zeros([noSeeds,noIterations+1,3])
    streamlinePositions[:,0,] = seeds[0:noSeeds]
    streamlinePositions[:,1,] = seeds[0:noSeeds]


    # interpolate data given these coordinates for each channel
    x = np.zeros([noSeeds,noX,noY,noZ,dw])
    previousDirections = np.zeros([noSeeds,3])

    for iter in range(1,noIterations):
        #print(str(noIterations))
        # interpolate dwi data for each point of our streamline
        for j in range(0,noSeeds):
            coordVecs = np.vstack(np.meshgrid(x_,y_,z_)).reshape(3,-1).T + streamlinePositions[j,iter,]
            for i in range(0,dw):
                x[j,:,:,:,i] = np.reshape(vfu.interpolate_scalar_3d(data[:,:,:,i],coordVecs)[0], [noX,noY,noZ])

        # predict possible directions
        x_ext = nn_helper.normalize(x)
        with tf.device('/cpu:0'):
            nextProbableDirections = model.predict(x_ext, batch_size = 1024)

        # compute next streamline position
        lastDirections = streamlinePositions[:,iter,] - streamlinePositions[:,iter-1,]
        for j in range(0,noSeeds):
            probs = nextProbableDirections[j,].dot(lastDirections[j] / 255)
            idx_map = np.argmin(abs(probs))
            nextDirection = np.squeeze(nextProbableDirections[j,idx_map,:])

            streamlinePositions[j,iter+1,] = streamlinePositions[j,iter,] + stepWidth * nextDirection

    return streamlinePositions

def applyTrackerNetwork(seeds, data, model, noX=3, noY=3,noZ=3,dw=288,coordinateScaling = 0.1, stepWidth = 0.5):
    x_ = coordinateScaling * np.linspace(-1., 1., noX)
    y_ = coordinateScaling * np.linspace(-1., 1., noY)
    z_ = coordinateScaling * np.linspace(-1., 1., noZ)       

    noSeeds = len(seeds)
    noSeeds = 50
    noIterations = 1000

    # initialize streamline positions data
    streamlinePositions = np.zeros([noSeeds,noIterations+1,3])
    streamlinePositions[:,0,] = seeds[0:noSeeds]
    streamlinePositions[:,1,] = seeds[0:noSeeds]

    # interpolate data given these coordinates for each channel
    x = np.zeros([noSeeds,noX,noY,noZ,dw])
    
    for iter in range(1,noIterations):
        # interpolate dwi data for each point of our streamline
        for j in range(0,noSeeds):
            coordVecs = np.vstack(np.meshgrid(x_,y_,z_)).reshape(3,-1).T + streamlinePositions[j,iter,]
            for i in range(0,dw):
                x[j,:,:,:,i] = np.reshape(vfu.interpolate_scalar_3d(data[:,:,:,i],coordVecs)[0], [noX,noY,noZ])

        # predict possible directions
        x_ext = nn_helper.normalize(x)
        lastDirections = streamlinePositions[:,iter,] - streamlinePositions[:,iter-1,]
        lastDirections = np.expand_dims(lastDirections, axis=1)
        with tf.device('/cpu:0'):
            likelyDirections, predictedDirection = model.predict([x_ext,lastDirections])

        # update next streamline position
        for j in range(0,noSeeds):
            streamlinePositions[j,iter+1,] = streamlinePositions[j,iter,] + stepWidth * predictedDirection[j,]

    return streamlinePositions
    

def applySimpleTrackerNetwork(seeds, data, model, noX=3, noY=3,noZ=3,dw=288,coordinateScaling = 0.1, stepWidth = 0.5):
    x_ = coordinateScaling * np.linspace(-1., 1., noX)
    y_ = coordinateScaling * np.linspace(-1., 1., noY)
    z_ = coordinateScaling * np.linspace(-1., 1., noZ)       

    noSeeds = len(seeds)
    #noSeeds = 50
    noIterations = 1000

    # initialize streamline positions data
    streamlinePositions = np.zeros([noSeeds,noIterations+1,3])
    streamlinePositions[:,0,] = seeds[0:noSeeds]
    streamlinePositions[:,1,] = seeds[0:noSeeds]

    # interpolate data given these coordinates for each channel
    x = np.zeros([noSeeds,noX,noY,noZ,dw])
    
    for iter in range(1,noIterations):
        # interpolate dwi data for each point of our streamline
        for j in range(0,noSeeds):
            coordVecs = np.vstack(np.meshgrid(x_,y_,z_)).reshape(3,-1).T + streamlinePositions[j,iter,]
            for i in range(0,dw):
                x[j,:,:,:,i] = np.reshape(vfu.interpolate_scalar_3d(data[:,:,:,i],coordVecs)[0], [noX,noY,noZ])

        # predict possible directions
        x_ext = nn_helper.normalizeDWI(x)
        lastDirections = streamlinePositions[:,iter,] - streamlinePositions[:,iter-1,]
        lastDirections = np.expand_dims(lastDirections, axis=1)
        with tf.device('/cpu:0'):
            predictedDirection = nn_helper.denormalizeStreamlineOrientation(model.predict([x_ext]))

        # normalize prediction
        vecNorms = np.sqrt(np.sum(predictedDirection ** 2 , axis = 1))
        predictedDirection = np.nan_to_num(predictedDirection / vecNorms[:,None])   
            
        # update next streamline position
        for j in range(0,noSeeds):
            streamlinePositions[j,iter+1,] = streamlinePositions[j,iter,] + stepWidth * predictedDirection[j,]
            print(str(predictedDirection[j,]))

    return streamlinePositions