import time
import nrrd
import nibabel as nb
import numpy as np
import matplotlib.pyplot as plt
import h5py

from dipy.tracking.local import LocalTracking, ThresholdTissueClassifier
from dipy.tracking.utils import random_seeds_from_mask
from dipy.reconst.dti import TensorModel, quantize_evecs
from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel,
                                   auto_response)
from dipy.reconst.shm import CsaOdfModel
from dipy.data import default_sphere
from dipy.direction import peaks_from_model
from dipy.data import fetch_stanford_hardi, read_stanford_hardi, get_sphere
from dipy.segment.mask import median_otsu
from dipy.viz import actor, window
from dipy.io.image import save_nifti
from dipy.io import read_bvals_bvecs
from dipy.core import gradients
from dipy.tracking.streamline import Streamlines
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table, gradient_table_from_bvals_bvecs
from dipy.reconst.dti import fractional_anisotropy

from dipy.tracking import utils

import src.dwi_tools as dwi_tools
import src.nn_helper as nn_helper
import src.SelectiveDropout as SelectiveDropout

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

import tensorflow as tf
from joblib import Parallel, delayed
import multiprocessing
from keras.models import load_model


def joinTwoAlignedStreamlineLists(streamlines_left,streamlines_right):
    assert(len(streamlines_left) == len(streamlines_right), "The two lists of streamlines need to have the same number of elements.")
    
    streamlines_joined = []
    
    for i in range(0,len(streamlines_left)):
        sl_l = np.flipud(streamlines_left[i][1:])
        sl_r = streamlines_right[i][1:]
        streamlines_joined.append(np.concatenate([sl_l, sl_r]))
        
    return streamlines_joined


def makeStep(predictedDirection,lastDirections,curStreamlinePos_ras,M,abc,stepWidth,start_time=0,printfProfiling=False):
    ####
    ####
    # check predicted direction and flip it if necessary
    noSeeds = len(predictedDirection)
    
    if(printfProfiling):
        print(" -> 3 " + str(time.time() - start_time) + "s]")
        
    #predictedDirection[np.where(theta < 0),] = -1 * predictedDirection[np.where(theta < 0),] ### probably a little bit faster
    for j in range(0,noSeeds):
        lv1 = predictedDirection[j,]
        pv1 = lastDirections[j,]
        #print(str(lv1) + '--' + str(pv1))
        #theta = np.arcsin(np.dot(pv1,lv1) / (np.linalg.norm(pv1)*np.linalg.norm(lv1)))   

        theta = np.dot(pv1,lv1)

        if(theta < 0): # used to be: and iter > 1 when outside this function
            predictedDirection[j,] = -predictedDirection[j,]

    if(printfProfiling):
        print(" -> 4 " + str(time.time() - start_time) + "s]")

    ####
    ####
    # compute next streamline position and check if this position is valid wrt. our stopping criteria1
    candidatePosition_ras = curStreamlinePos_ras - stepWidth * predictedDirection 
    candidatePosition_ijk = (M.dot(candidatePosition_ras.T) + abc).T   #projectRAStoIJK(candidatePosition,M,abc)
    
    return candidatePosition_ras, candidatePosition_ijk


def getNextDirection(dwi,curPosition_ijk, model, lastDirections = None, x_ = [0], y_ = [0], z_ = [0], noX = 1, noY = 1, noZ = 1, batch_size = 2**10, reshapeForConvNet = False, rotateData = False, usePreviousDirection = False):
    rot = None
    
    if(rotateData):
        print('rotating data')
        start_time = time.time()
        # reference orientation
        vv = dwi_tools.getReferenceOrientation()

        noPositions = len(lastDirections)

        # compute rotation matrices
        rot = np.zeros([noPositions,3,3])

        for k in range(noPositions):
            dwi_tools.R_2vect(rot[k,:,:],vector_orig=lastDirections[k,],vector_fin=vv)
        print(" -> " + str(time.time() - start_time) + "s")

    dwi_at_curPosition = dwi_tools.interpolateDWIVolume(dwi, curPosition_ijk, x_,y_,z_, noX = noX, noY = noY, noZ = noZ, rotations = rot)
    
    if(reshapeForConvNet):
        noSamples,dx,dy,dz,dw = dwi_at_curPosition.shape
        dwi_at_curPosition = np.reshape(dwi_at_curPosition, [noSamples,16,16])
        dwi_at_curPosition = dwi_at_curPosition[..., np.newaxis]
        
    if(usePreviousDirection == False):
        predictedDirection = model.predict([dwi_at_curPosition], batch_size = batch_size)
    else:
        if(reshapeForConvNet):
            # CNN model w/ previous direction is different than the MLP as it doesnt require the last streamline vector but the actual DWI data at the previous streamline position.
            # we hacked that into the last direction :D
            print("ld shape:" + str(lastDirections.shape))
            print("cur shape:" + str(dwi_at_curPosition.shape))
            dwi_at_prev_and_curPosition = np.stack((np.squeeze(lastDirections), np.squeeze(dwi_at_curPosition)))
            dwi_at_prev_and_curPosition = np.moveaxis(dwi_at_prev_and_curPosition,0,-1)
            predictedDirection = model.predict([dwi_at_prev_and_curPosition], batch_size = batch_size)
        else:
            lastDirections = -1 * lastDirections
            predictedDirection = model.predict([dwi_at_curPosition, lastDirections], batch_size = batch_size)
        
    
    #predictedDirection = predictedDirection * 2 - 1
        
        
    if(len(predictedDirection) == 2):
        predictedDirection = predictedDirection[0]
        
    vecNorms = np.sqrt(np.sum(predictedDirection ** 2 , axis = 1))
    predictedDirection = np.nan_to_num(predictedDirection / vecNorms[:,None])
        
    return predictedDirection, vecNorms, dwi_at_curPosition


def getNextDirectionMagicModel(dwi,curPosition_ijk, model, lastDirections = None, x_ = [0], y_ = [0], z_ = [0], noX = 1, noY = 1, noZ = 1, batch_size = 2**10, reshapeForConvNet = False, validIdx = None):
    
    if(validIdx is None):
        validIdx = list(range(len(curPosition_ijk)))
    
    dwi_at_curPosition = dwi_tools.interpolateDWIVolume(dwi, curPosition_ijk[validIdx,], x_,y_,z_, noX = noX, noY = noY, noZ = noZ)
    
    if(reshapeForConvNet):
        noSamples,dx,dy,dz,dw = dwi_at_curPosition.shape
        dwi_at_curPosition = np.reshape(dwi_at_curPosition, [noSamples,16,16])
        dwi_at_curPosition = dwi_at_curPosition[..., np.newaxis]
    
    predictedDirection = np.zeros([len(curPosition_ijk),3])
    stopTrackingProbability = np.zeros([len(curPosition_ijk),1])
    
    
    
    if(lastDirections is None):
        predictedDirectionAtIdx, stopTrackingProbabilityAtIdx = model.predict([dwi_at_curPosition], batch_size = batch_size)
    else:
        lastDirections = lastDirections[validIdx,]
        if(reshapeForConvNet):
            # CNN model w/ previous direction is different than the MLP as it doesnt require the last streamline vector but the actual DWI data at the previous streamline position.
            # we hacked that into the last direction :D
            print("ld shape:" + str(lastDirections.shape))
            print("cur shape:" + str(dwi_at_curPosition.shape))
            dwi_at_prev_and_curPosition = np.stack((np.squeeze(lastDirections), np.squeeze(dwi_at_curPosition)))
            dwi_at_prev_and_curPosition = np.moveaxis(dwi_at_prev_and_curPosition,0,-1)
            predictedDirectionAtIdx, stopTrackingProbabilityAtIdx = model.predict([dwi_at_prev_and_curPosition], batch_size = batch_size)
        else:
            lastDirections = -1 * lastDirections
            predictedDirectionAtIdx, stopTrackingProbabilityAtIdx = model.predict([dwi_at_curPosition, lastDirections], batch_size = batch_size)

    if(len(predictedDirection) == 2):
        predictedDirection = predictedDirection[0]
            
    predictedDirection[validIdx,] = predictedDirectionAtIdx
    stopTrackingProbability[validIdx,] = stopTrackingProbabilityAtIdx
    #predictedDirection = predictedDirection * 2 - 1
        
    vecNorms = np.sqrt(np.sum(predictedDirection ** 2 , axis = 1))
    predictedDirection = np.nan_to_num(predictedDirection / vecNorms[:,None])
    
    stopTrackingProbability = np.squeeze(stopTrackingProbability)
        
    return predictedDirection, vecNorms, dwi_at_curPosition, stopTrackingProbability


def start(seeds, data, model, affine, mask, fa, printProgress = False, fa_threshold = 0.2, noX=3, noY=3,noZ=3,dw=288,coordinateScaling = 0.1, stepWidth = 0.1, useSph = False,inverseDirection = False, printfProfiling = False, noIterations = 200, batch_size = 2**12, usePreviousDirection = True, reshapeForConvNet = False, rotateData = False):   
    '''
    fibre tracking using neural networks
    '''    
    mask = mask.astype(np.float)
    
    noSeeds = len(seeds)

    # initialize streamline positions data
    vNorms = np.zeros([2*noSeeds,noIterations+1])
    streamlinePositions = np.zeros([2*noSeeds,noIterations+1,3])
    
    streamlinePositions[0:noSeeds,0,] = seeds[0:noSeeds] ## FORWARD
    streamlinePositions[noSeeds:,1,] = seeds[0:noSeeds] ## BACKWARD
    
    indexLastStreamlinePosition = noIterations * np.ones([2*noSeeds], dtype=np.intp)

    del seeds # its not supposed to use the seeds anymore
    
    # interpolate data given these coordinates for each channel
    #x = np.zeros([2*noSeeds,noX,noY,noZ,dw])
    x_,y_,z_ = dwi_tools._getCoordinateGrid(noX,noY,noZ,coordinateScaling)

    # prepare transformations to project a streamline point from IJK into RAS coordinate system to interpolate DWI data
    aff_ras_ijk = np.linalg.inv(affine) # aff: IJK -> RAS
    M = aff_ras_ijk[:3, :3]
    abc = aff_ras_ijk[:3, 3]
    abc = abc[:,None]
    
    
    ### FIRST STEP PREDICTION ###
    # just predict the forward direction (the first half of streamlinePositions)
    curStreamlinePos_ras = streamlinePositions[0:noSeeds,0,]
    curStreamlinePos_ijk = (M.dot(curStreamlinePos_ras.T) + abc).T
    lastDirections = np.zeros([noSeeds,3])
    
    ld_input = lastDirections
    if(reshapeForConvNet):
        dwi_at_lastPosition = dwi_tools.interpolateDWIVolume(data, curStreamlinePos_ijk, x_,y_,z_, noX = noX, noY = noY, noZ = noZ) ## otherwise try zero.. depending on model
        noSamples,dx,dy,dz,dw = dwi_at_lastPosition.shape
        dwi_at_lastPosition = np.reshape(dwi_at_lastPosition, [noSamples,16,16])
        dwi_at_lastPosition = np.zeros((noSamples,16,16))
        dwi_at_lastPosition = dwi_at_lastPosition[..., np.newaxis]
        ld_input = dwi_at_lastPosition
    
    # predict direction but in never rotate data
    predictedDirection, vecNorms, dwi_at_curPosition = getNextDirection(data, curPosition_ijk = curStreamlinePos_ijk, model = model, lastDirections = ld_input, reshapeForConvNet = reshapeForConvNet, rotateData = False, usePreviousDirection = usePreviousDirection, noX = noX, noY = noY, noZ = noZ, x_ = x_, y_ = y_, z_ = z_)
        
    # compute next streamline position
    candidatePosition_ras, candidatePosition_ijk = makeStep(stepWidth=stepWidth,printfProfiling=printfProfiling,predictedDirection = predictedDirection, lastDirections = lastDirections, curStreamlinePos_ras = curStreamlinePos_ras, M = M, abc = abc);
    
    # update positions
    streamlinePositions[0:noSeeds,1,] = candidatePosition_ras
    streamlinePositions[noSeeds:,0,] = candidatePosition_ras
    candidatePosition_ras = streamlinePositions[:,1,]
    candidatePosition_ijk = (M.dot(candidatePosition_ras.T) + abc).T
    
    if(reshapeForConvNet and usePreviousDirection):
        cp = streamlinePositions[:,0,] 
        cp_ijk = (M.dot(cp.T) + abc).T
        dwi_at_curPosition = dwi_tools.interpolateDWIVolume(data, cp_ijk, x_,y_,z_, noX = noX, noY = noY, noZ = noZ) ## otherwise try zero.. depending on model
        noSamples,dx,dy,dz,dw = dwi_at_curPosition.shape
        dwi_at_curPosition = np.reshape(dwi_at_curPosition, [noSamples,16,16])
        dwi_at_curPosition = dwi_at_curPosition[..., np.newaxis]
    
    ### START ITERATION UNTIL NOITERATIONS REACHED ###
    start_time = time.time()    
    for iter in range(1,noIterations):
        ####
        ####
        # compute current position and last direction
        if(printProgress):
            print(str(iter-1) + "/" + str(noIterations) + " [" + str(time.time() - start_time) + "s]")
        curStreamlinePos_ras = candidatePosition_ras
        curStreamlinePos_ijk = candidatePosition_ijk
        lastDirections = (streamlinePositions[:,iter-1,] - streamlinePositions[:,iter,]) # previousPosition - currentPosition
        
        ####
        ####
        # compute direction
        ld_input = lastDirections
        if(reshapeForConvNet):
            ld_input = dwi_at_curPosition
            
        predictedDirection, vecNorms, dwi_at_curPosition = getNextDirection(data, curPosition_ijk = curStreamlinePos_ijk, model = model, lastDirections = ld_input, reshapeForConvNet = reshapeForConvNet, rotateData = rotateData, usePreviousDirection = usePreviousDirection, noX = noX, noY = noY, noZ = noZ, x_ = x_, y_ = y_, z_ = z_)
        vNorms[:,iter,] = vecNorms
        
        ####
        ####
        # compute next streamline position and check if this position is valid wrt. our stopping criteria1
        candidatePosition_ras, candidatePosition_ijk = makeStep(stepWidth=stepWidth,predictedDirection = predictedDirection, lastDirections = lastDirections, curStreamlinePos_ras = curStreamlinePos_ras, M = M, abc = abc, start_time = start_time, printfProfiling=printfProfiling)
        
        validPoints = areVoxelsValidStreamlinePoints(candidatePosition_ijk, mask, fa, fa_threshold)
        
        if(printfProfiling):
            print(" -> 5 " + str(time.time() - start_time) + "s]")
        
        if(printProgress):
            print("valid ratio %d / %d (%.2f)" % (sum(validPoints), 2*noSeeds, float(sum(validPoints)) / float(2*noSeeds)))
            
        for j in range(0,2*noSeeds):
            if(validPoints[j]):
                streamlinePositions[j,iter+1,] = candidatePosition_ras[j,]
            else:
                #streamlinePositions[j,iter+1,] = candidatePosition_ras[j,]
                indexLastStreamlinePosition[j] = np.min((indexLastStreamlinePosition[j],iter))
        
        
    streamlinePositions = streamlinePositions.tolist()
    
    ####
    ####
    # 
    # crop streamlines to length specified by stopping criteria
    vNorms = vNorms.tolist()
    
    for seedIdx in range(0,2*noSeeds):
        currentStreamline = np.array(streamlinePositions[seedIdx])
        currentStreamline = currentStreamline[0:indexLastStreamlinePosition[seedIdx],]
        
        currentNorm = np.array(vNorms[seedIdx])
        currentNorm = currentNorm[0:indexLastStreamlinePosition[seedIdx],]
        
        streamlinePositions[seedIdx] = currentStreamline
        vNorms[seedIdx] = currentNorm

    # extract both directions
    sl_forward = streamlinePositions[0:noSeeds]
    sl_backward = streamlinePositions[noSeeds:]

    # join directions
    streamlines = Streamlines(joinTwoAlignedStreamlineLists(sl_backward, sl_forward))
    
    return streamlines, vNorms



def startMagicModel(seeds, data, model, affine, mask, printProgress = False, noX=3, noY=3,noZ=3,dw=288,coordinateScaling = 0.1, stepWidth = 0.1, useSph = False,inverseDirection = False, printfProfiling = False, noIterations = 200, batch_size = 2**12, usePreviousDirection = True, reshapeForConvNet = False):   
    '''
    fibre tracking using neural networks
    '''    
    mask = mask.astype(np.float)
    
    noSeeds = len(seeds)

    # initialize streamline positions data
    vNorms = np.zeros([2*noSeeds,noIterations+1])
    stopProbabilities = np.zeros([2*noSeeds,noIterations+1])
    streamlinePositions = np.zeros([2*noSeeds,noIterations+1,3])
    
    streamlinePositions[0:noSeeds,0,] = seeds[0:noSeeds] ## FORWARD
    streamlinePositions[noSeeds:,1,] = seeds[0:noSeeds] ## BACKWARD
    
    indexLastStreamlinePosition = noIterations * np.ones([2*noSeeds], dtype=np.intp)

    del seeds # its not supposed to use the seeds anymore
    
    # interpolate data given these coordinates for each channel
    #x = np.zeros([2*noSeeds,noX,noY,noZ,dw])
    x_,y_,z_ = dwi_tools._getCoordinateGrid(noX,noY,noZ,coordinateScaling)

    # prepare transformations to project a streamline point from IJK into RAS coordinate system to interpolate DWI data
    aff_ras_ijk = np.linalg.inv(affine) # aff: IJK -> RAS
    M = aff_ras_ijk[:3, :3]
    abc = aff_ras_ijk[:3, 3]
    abc = abc[:,None]
    
    
    ### FIRST STEP PREDICTION ###
    # just predict the forward direction (the first half of streamlinePositions)
    curStreamlinePos_ras = streamlinePositions[0:noSeeds,0,]
    curStreamlinePos_ijk = (M.dot(curStreamlinePos_ras.T) + abc).T
    lastDirections = np.zeros([noSeeds,3])
    ld_i = lastDirections
    
#   if(usePreviousDirection == False):
#       predictedDirection, vecNorms, dwi_at_curPosition, stopTrackingProbability = getNextDirectionMagicModel(data, curPosition_ijk = curStreamlinePos_ijk, model = model, lastDirections = None, reshapeForConvNet = reshapeForConvNet)
#   else:
    if(reshapeForConvNet):
        dwi_at_lastPosition = dwi_tools.interpolateDWIVolume(data, curStreamlinePos_ijk, x_,y_,z_, noX = noX, noY = noY, noZ = noZ) ## otherwise try zero.. depending on model
        noSamples,dx,dy,dz,dw = dwi_at_lastPosition.shape
        dwi_at_lastPosition = np.reshape(dwi_at_lastPosition, [noSamples,16,16])
        dwi_at_lastPosition = np.zeros((noSamples,16,16))
        dwi_at_lastPosition = dwi_at_lastPosition[..., np.newaxis]
        ld_i = dwi_at_lastPosition
#           predictedDirection, vecNorms, dwi_at_curPosition, stopTrackingProbability = getNextDirectionMagicModel(data, curPosition_ijk = curStreamlinePos_ijk, model = model, lastDirections = dwi_at_lastPosition, reshapeForConvNet = reshapeForConvNet)
#       else:
    
    predictedDirection, vecNorms, dwi_at_curPosition, stopTrackingProbability = getNextDirectionMagicModel(data, curPosition_ijk = curStreamlinePos_ijk, model = model, lastDirections = ld_i, reshapeForConvNet = reshapeForConvNet)
        
    candidatePosition_ras, candidatePosition_ijk = makeStep(stepWidth=stepWidth,printfProfiling=printfProfiling,predictedDirection = predictedDirection, lastDirections = lastDirections, curStreamlinePos_ras = curStreamlinePos_ras, M = M, abc = abc);
            
    streamlinePositions[0:noSeeds,1,] = candidatePosition_ras
    streamlinePositions[noSeeds:,0,] = candidatePosition_ras
    
    stopProbabilities[0:noSeeds,1] = stopTrackingProbability
    stopProbabilities[noSeeds:,0] = stopTrackingProbability
  
    candidatePosition_ras = streamlinePositions[:,1,]
    candidatePosition_ijk = (M.dot(candidatePosition_ras.T) + abc).T
    
    if(reshapeForConvNet and usePreviousDirection):
        cp = streamlinePositions[:,0,] 
        cp_ijk = (M.dot(cp.T) + abc).T
        dwi_at_curPosition = dwi_tools.interpolateDWIVolume(data, cp_ijk, x_,y_,z_, noX = noX, noY = noY, noZ = noZ) ## otherwise try zero.. depending on model
        noSamples,dx,dy,dz,dw = dwi_at_curPosition.shape
        dwi_at_curPosition = np.reshape(dwi_at_curPosition, [noSamples,16,16])
        dwi_at_curPosition = dwi_at_curPosition[..., np.newaxis]
    
    validSls = list(range(0,2*noSeeds))
    
    ### START ITERATION UNTIL NOITERATIONS REACHED ###
    start_time = time.time()    
    for iter in range(1,noIterations):
        ####
        ####
        # compute current position and last direction
        if(printProgress):
            print(str(iter-1) + "/" + str(noIterations) + " [" + str(time.time() - start_time) + "s]")
        curStreamlinePos_ras = candidatePosition_ras
        curStreamlinePos_ijk = candidatePosition_ijk
        lastDirections = (streamlinePositions[:,iter-1,] - streamlinePositions[:,iter,]) # previousPosition - currentPosition
        
        ####
        ####
        # compute direction
        if(usePreviousDirection == False):
            predictedDirection, vecNorms, dwi_at_curPosition, stopTrackingProbability = getNextDirectionMagicModel(data, curPosition_ijk = curStreamlinePos_ijk, model = model, lastDirections = None, reshapeForConvNet = reshapeForConvNet, validIdx = validSls)
        else:
            if(reshapeForConvNet):
                predictedDirection, vecNorms, dwi_at_curPosition, stopTrackingProbability = getNextDirectionMagicModel(data, curPosition_ijk = curStreamlinePos_ijk, model = model, lastDirections = dwi_at_curPosition, reshapeForConvNet = reshapeForConvNet, validIdx = validSls)
            else:
                predictedDirection, vecNorms, dwi_at_curPosition, stopTrackingProbability = getNextDirectionMagicModel(data, curPosition_ijk = curStreamlinePos_ijk, model = model, lastDirections = lastDirections, reshapeForConvNet = reshapeForConvNe, validIdx = validSls)
        vNorms[:,iter,] = vecNorms
        stopProbabilities[:,iter] = stopTrackingProbability
        ####
        ####
        # compute next streamline position and check if this position is valid wrt. our stopping criteria1
        candidatePosition_ras, candidatePosition_ijk = makeStep(stepWidth=stepWidth,predictedDirection = predictedDirection, lastDirections = lastDirections, curStreamlinePos_ras = curStreamlinePos_ras, M = M, abc = abc, start_time = start_time, printfProfiling=printfProfiling)
        
        #streamlinePositions[:,iter+1,] = candidatePosition_ras[:,]
        print('Average valid prob %.2f (valid ratio %.2f)' % (np.mean(stopTrackingProbability), len(np.where(stopTrackingProbability > 0.5)[0]) / (2*noSeeds) ))
        
        validSls = np.where(stopTrackingProbability > 0.5)[0]
        
        if(len(validSls) == 0):
            break
        
        for j in range(0,2*noSeeds):
            if(stopTrackingProbability[j] > 0.5):
                streamlinePositions[j,iter+1,] = candidatePosition_ras[j,]
            else:
                    #streamlinePositions[j,iter+1,] = candidatePosition_ras[j,]
                indexLastStreamlinePosition[j] = np.min((indexLastStreamlinePosition[j],iter))
        
    streamlinePositions = streamlinePositions.tolist()
    
    ####
    ####
    # 
    # crop streamlines to length specified by stopping criteria
    vNorms = vNorms.tolist()
    
    for seedIdx in range(0,2*noSeeds):
        currentStreamline = np.array(streamlinePositions[seedIdx])
        currentStreamline = currentStreamline[0:indexLastStreamlinePosition[seedIdx],]
        
        currentNorm = np.array(vNorms[seedIdx])
        currentNorm = currentNorm[0:indexLastStreamlinePosition[seedIdx],]
        
        streamlinePositions[seedIdx] = currentStreamline
        vNorms[seedIdx] = currentNorm

    # extract both directions
    sl_forward = streamlinePositions[0:noSeeds]
    sl_backward = streamlinePositions[noSeeds:]

    # join directions
    streamlines = Streamlines(joinTwoAlignedStreamlineLists(sl_backward, sl_forward))
    
    return streamlines, vNorms, stopProbabilities


def startAggregatedMagicModel(seeds, data, model, affine, mask, printProgress = False, noX=3, noY=3,noZ=3,dw=288,coordinateScaling = 0.1, stepWidth = 0.1, useSph = False,inverseDirection = False, printfProfiling = False, noIterations = 200, batch_size = 2**12, usePreviousDirection = True, reshapeForConvNet = False):   
    '''
    fibre tracking using neural networks
    '''    
    mask = mask.astype(np.float)
    
    noSeeds = len(seeds)

    # initialize streamline positions data
    vNorms = np.zeros([2*noSeeds,noIterations+1])
    stopProbabilities = np.zeros([2*noSeeds,noIterations+1])
    streamlinePositions = np.zeros([2*noSeeds,noIterations+1,3])
    
    streamlinePositions[0:noSeeds,0,] = seeds[0:noSeeds] ## FORWARD
    streamlinePositions[noSeeds:,1,] = seeds[0:noSeeds] ## BACKWARD
    
    indexLastStreamlinePosition = noIterations * np.ones([2*noSeeds], dtype=np.intp)

    del seeds # its not supposed to use the seeds anymore
    
    # interpolate data given these coordinates for each channel
    #x = np.zeros([2*noSeeds,noX,noY,noZ,dw])
    x_,y_,z_ = dwi_tools._getCoordinateGrid(noX,noY,noZ,coordinateScaling)

    # prepare transformations to project a streamline point from IJK into RAS coordinate system to interpolate DWI data
    aff_ras_ijk = np.linalg.inv(affine) # aff: IJK -> RAS
    M = aff_ras_ijk[:3, :3]
    abc = aff_ras_ijk[:3, 3]
    abc = abc[:,None]
    
    
    ### FIRST STEP PREDICTION ###
    # just predict the forward direction (the first half of streamlinePositions)
    curStreamlinePos_ras = streamlinePositions[0:noSeeds,0,]
    curStreamlinePos_ijk = (M.dot(curStreamlinePos_ras.T) + abc).T
    lastDirections = np.zeros([noSeeds,3])
    
    dwi_at_lastPosition = dwi_tools.interpolateDWIVolume(data, curStreamlinePos_ijk, x_,y_,z_, noX = noX, noY = noY, noZ = noZ) ## otherwise try zero.. depending on model
    predictedDirection, vecNorms, dwi_at_curPosition, stopTrackingProbability = getNextDirectionMagicModel(data, curPosition_ijk = curStreamlinePos_ijk, model = model, lastDirections = dwi_at_lastPosition, reshapeForConvNet = reshapeForConvNet)
 
    candidatePosition_ras, candidatePosition_ijk = makeStep(stepWidth=stepWidth,printfProfiling=printfProfiling,predictedDirection = predictedDirection, lastDirections = lastDirections, curStreamlinePos_ras = curStreamlinePos_ras, M = M, abc = abc);
            
    streamlinePositions[0:noSeeds,1,] = candidatePosition_ras
    streamlinePositions[noSeeds:,0,] = candidatePosition_ras
    
    stopProbabilities[0:noSeeds,1] = stopTrackingProbability
    stopProbabilities[noSeeds:,0] = stopTrackingProbability
  
    candidatePosition_ras = streamlinePositions[:,1,]
    candidatePosition_ijk = (M.dot(candidatePosition_ras.T) + abc).T
    
    cp = streamlinePositions[:,0,] 
    cp_ijk = (M.dot(cp.T) + abc).T
    lastPos = dwi_tools.interpolateDWIVolume(data, cp_ijk, x_,y_,z_, noX = noX, noY = noY, noZ = noZ) ## otherwise try zero.. depending on model
    lastPosDiv = 1
    
    ### START ITERATION UNTIL NOITERATIONS REACHED ###
    start_time = time.time()    
    for iter in range(1,noIterations):
        ####
        ####
        # compute current position and last direction
        if(printProgress):
            print(str(iter-1) + "/" + str(noIterations) + " [" + str(time.time() - start_time) + "s]")
        curStreamlinePos_ras = candidatePosition_ras
        curStreamlinePos_ijk = candidatePosition_ijk
        lastDirections = (streamlinePositions[:,iter-1,] - streamlinePositions[:,iter,]) # previousPosition - currentPosition
        
        ####
        ####
        # compute direction
        dwi_in = dwi_at_curPosition
        if(iter>1):
            lastPos = lastPos + dwi_at_curPosition 
            
        predictedDirection, vecNorms, dwi_at_curPosition, stopTrackingProbability = getNextDirectionMagicModel(data, curPosition_ijk = curStreamlinePos_ijk, model = model, lastDirections = lastPos / iter, reshapeForConvNet = reshapeForConvNet)

        vNorms[:,iter,] = vecNorms
        stopProbabilities[:,iter] = stopTrackingProbability

        ####
        ####
        # compute next streamline position and check if this position is valid wrt. our stopping criteria1
        candidatePosition_ras, candidatePosition_ijk = makeStep(stepWidth=stepWidth,predictedDirection = predictedDirection, lastDirections = lastDirections, curStreamlinePos_ras = curStreamlinePos_ras, M = M, abc = abc, start_time = start_time, printfProfiling=printfProfiling)
        
        #streamlinePositions[:,iter+1,] = candidatePosition_ras[:,]
        print('Average valid prob %.2f (valid ratio %.2f)' % (np.mean(stopTrackingProbability), np.sum(np.where(stopTrackingProbability > 0.5)[0]) / (2*noSeeds) ))

        for j in range(0,2*noSeeds):
            if(stopTrackingProbability[j] > 0.5):
                streamlinePositions[j,iter+1,] = candidatePosition_ras[j,]
            else:
                    #streamlinePositions[j,iter+1,] = candidatePosition_ras[j,]
                indexLastStreamlinePosition[j] = np.min((indexLastStreamlinePosition[j],iter))
        
    streamlinePositions = streamlinePositions.tolist()
    
    ####
    ####
    # 
    # crop streamlines to length specified by stopping criteria
    vNorms = vNorms.tolist()
    
    for seedIdx in range(0,2*noSeeds):
        currentStreamline = np.array(streamlinePositions[seedIdx])
        currentStreamline = currentStreamline[0:indexLastStreamlinePosition[seedIdx],]
        
        currentNorm = np.array(vNorms[seedIdx])
        currentNorm = currentNorm[0:indexLastStreamlinePosition[seedIdx],]
        
        streamlinePositions[seedIdx] = currentStreamline
        vNorms[seedIdx] = currentNorm

    # extract both directions
    sl_forward = streamlinePositions[0:noSeeds]
    sl_backward = streamlinePositions[noSeeds:]

    # join directions
    streamlines = Streamlines(joinTwoAlignedStreamlineLists(sl_backward, sl_forward))
    
    return streamlines, vNorms, stopProbabilities



def areVoxelsValidStreamlinePoints(nextCandidatePositions_ijk,mask,fa,fa_threshold):
    return np.logical_and( np.greater(vfu.interpolate_scalar_3d(mask,nextCandidatePositions_ijk)[0], 0), np.greater(vfu.interpolate_scalar_3d(fa,nextCandidatePositions_ijk)[0], fa_threshold)  )
    

def isVoxelValidStreamlinePoint(nextCandidatePosition_ijk,mask,fa,fa_threshold):  
    if(vfu.interpolate_scalar_3d(mask,nextCandidatePosition_ijk)[0] == 0):
        return False

    if(vfu.interpolate_scalar_3d(fa,nextCandidatePosition_ijk)[0] < fa_threshold):
        return False
    
    return True


def projectRAStoIJK(coordinateRAS, M, abc):
    coordinateRAS = coordinateRAS[:,None]
    return (M.dot(coordinateRAS) + abc).T
