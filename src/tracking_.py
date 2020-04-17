import time
import nrrd
import nibabel as nb
import numpy as np
import matplotlib.pyplot as plt
import h5py
import torch

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

from src.util import progress
from dipy.tracking import utils

import src.dwi_tools as dwi_tools

from dipy.tracking.streamline import values_from_volume
import dipy.align.vector_fields as vfu
from dipy.core.sphere import Sphere
from dipy.core import subdivide_octahedron
from scipy.spatial import KDTree
from dipy.core import subdivide_octahedron

from dipy.tracking.local import LocalTracking
from dipy.tracking.streamline import Streamlines

from dipy.tracking import metrics

from src.model import ModelLSTM, ModelMLP

import numpy as np
import warnings

from joblib import Parallel, delayed
import multiprocessing


def getNextDirection(myState, dwi, curPosition_ijk, model, rot_ijk_val, x_ = [0], y_ = [0], z_ = [0]):

    ## 
    dwi_at_curPosition = dwi_tools.interpolateDWIVolume(myState, dwi, curPosition_ijk, x_,y_,z_, rotations_ijk = rot_ijk_val)
    
    
    dwi_at_curPosition = dwi_tools.projectIntoAppropriateSpace(myState, dwi_at_curPosition)
    dwi_at_curPosition = torch.from_numpy(dwi_at_curPosition).float() #.cuda()
    
    batchSize = dwi_at_curPosition.shape[0]
    featureSize = dwi_at_curPosition.shape[-1] * dwi_at_curPosition.shape[-2] * dwi_at_curPosition.shape[-3] * dwi_at_curPosition.shape[-4]
    dwi_at_curPosition = dwi_at_curPosition.view(1, batchSize, featureSize)
    predictedDirection, pContinueTracking = model(dwi_at_curPosition)

    return predictedDirection[0].numpy(), dwi_at_curPosition, pContinueTracking[0].numpy()
    #return predictedDirection[0].cpu().numpy(), dwi_at_curPosition, pContinueTracking[0].cpu().numpy()
    
def joinTwoAlignedStreamlineLists(streamlines_left,streamlines_right):
    assert(len(streamlines_left) == len(streamlines_right), "The two lists of streamlines need to have the same number of elements.")

    streamlines_joined = []

    for i in range(0,len(streamlines_left)):
        sl_l = np.flipud(streamlines_left[i][1:])
        sl_r = streamlines_right[i][1:]
        streamlines_joined.append(np.concatenate([sl_l, sl_r]))

    return streamlines_joined


def makeStep(myState, predictedDirection,lastDirections,curStreamlinePos_ijk,curStreamlinePos_ras,M,abc,M2,abc2,start_time=0,printfProfiling=False):
    #M2, abc2: IJK -> ras
    ####
    ####
    # check predicted direction and flip it if necessary
    noSeeds = len(predictedDirection)
    if(printfProfiling):
        print(" -> 3 " + str(time.time() - start_time) + "s]")

    for j in range(0,noSeeds):
        lv1 = predictedDirection[j,]
        pv1 = lastDirections[j,]
        if(printfProfiling):
            print(str(pv1) + ":" + str(lv1))

        theta = np.dot(pv1,lv1)
        # flip predictedDirection if the vectors point in opposite directions
        if(theta < 0):
            predictedDirection[j,] = -predictedDirection[j,]

    if(printfProfiling):
        print(" -> 4 " + str(time.time() - start_time) + "s]")

    ####
    ####
    # compute next streamline position and check if this position is valid wrt. our stopping criteria1
    if(myState.predictionInIJK):
        # predicted direction in IJK
        candidatePosition_ijk = curStreamlinePos_ijk - myState.stepWidth * predictedDirection
        candidatePosition_ras = (M2.dot(candidatePosition_ijk.T) + abc2).T
    else:
        # predicted direction in RAS
        candidatePosition_ras = curStreamlinePos_ras - myState.stepWidth * predictedDirection
        candidatePosition_ijk = (M.dot(candidatePosition_ras.T) + abc).T   #projectRAStoIJK(candidatePosition,M,abc)

    return candidatePosition_ras, candidatePosition_ijk


def projectRAStoIJK(coordinateRAS, M, abc):
    coordinateRAS = coordinateRAS[:,None]
    return (M.dot(coordinateRAS) + abc).T


# WORKING CODE
# DO NOT CHANGE, JUST COPY AND CHANGE THEN
def startWithRNNAndMLPCombination(myState, seeds, data, model,mlp_model, affine, mask, printProgress = False, inverseDirection = False, noIterations = 200, rot_ijk_val = None, batchSize=1024):
    ''' #
    fibre tracking using neural networks
    '''
    printfProfiling = False
    mask = mask.astype(np.float)

    noSeeds = len(seeds)
    print("Initalize Streamline positions data")
    # initialize streamline positions data
    vNorms = np.zeros([2*noSeeds,noIterations+1])
    streamlinePositions = np.zeros([2*noSeeds,noIterations+1,3])
    streamlinePositions_ijk = np.zeros([2*noSeeds,noIterations+1,3])

    streamlinePositions[0:noSeeds,0,] = seeds[0:noSeeds] ## FORWARD
    streamlinePositions[noSeeds:,1,] = seeds[0:noSeeds] ## BACKWARD

    indexLastStreamlinePosition = noIterations * np.ones([2*noSeeds], dtype=np.intp)

    #del seeds # its not supposed to use the seeds anymore

    # interpolate data given these coordinates for each channel
    x_,y_,z_ = dwi_tools._getCoordinateGrid(myState)
    
    # prepare transformations to project a streamline point from IJK into RAS coordinate system to interpolate DWI data
    aff_ras_ijk = np.linalg.inv(affine) # aff: IJK -> RAS
    M = aff_ras_ijk[:3, :3]
    abc = aff_ras_ijk[:3, 3]
    abc = abc[:,None]

    aff_ijk_ras = affine # affine: IJK -> RAS
    M2 = aff_ijk_ras[:3, :3]
    abc2 = aff_ijk_ras[:3, 3]
    abc2 = abc2[:,None]
    print("First step prediction")
    ### FIRST STEP PREDICTION ###
    # just predict the forward direction (the first half of streamlinePositions)
    curStreamlinePos_ras = streamlinePositions[0:noSeeds,0,]
    curStreamlinePos_ijk = (M.dot(curStreamlinePos_ras.T) + abc).T
    lastDirections = np.zeros([noSeeds,3])

    streamlinePositions_ijk[0:noSeeds,0,] = curStreamlinePos_ijk ## FORWARD
    streamlinePositions_ijk[noSeeds:,1,] = curStreamlinePos_ijk ## BACKWARD

    ld_input = lastDirections
    start_time = time.time()
    # never rotate the data of the first step
    oldRotationState = myState.rotateData
    myState.rotateData = False
    # predict direction but in never rotate data
    predictedDirection, dwi_at_curPosition, pContinueTracking = getNextDirection(myState, data, curPosition_ijk = curStreamlinePos_ijk, model = model, x_ = x_, y_ = y_, z_ = z_, rot_ijk_val = rot_ijk_val)
    myState.rotateData = oldRotationState
    # compute next streamline position
    candidatePosition_ras, candidatePosition_ijk = makeStep(myState, predictedDirection = predictedDirection,  curStreamlinePos_ras = curStreamlinePos_ras, M = M, abc = abc, start_time = start_time, printfProfiling=printfProfiling, M2 = M2, abc2 = abc2, curStreamlinePos_ijk = curStreamlinePos_ijk, lastDirections = lastDirections_ijk)

    # update positions
    streamlinePositions[0:noSeeds,1,] = candidatePosition_ras
    streamlinePositions[noSeeds:,0,] = candidatePosition_ras
    streamlinePositions_ijk[0:noSeeds,1,] = candidatePosition_ijk
    streamlinePositions_ijk[noSeeds:,0,] = candidatePosition_ijk
    candidatePosition_ras = streamlinePositions[:,1,]
    candidatePosition_ijk = (M.dot(candidatePosition_ras.T) + abc).T

    ### START ITERATION UNTIL NOITERATIONS REACHED ###
    # tracking steps are done in RAS
    validSls = None
   

    for seedIdx in range(0,2*noSeeds,batchSize):
        print('Streamline %d / %d ' % (seedIdx, 2*noSeeds))
        if isinstance(model, ModelLSTM):
            model.reset()
        seedIdxRange = range(seedIdx, min(seedIdx+batchSize, 2*noSeeds))
        candidatePosition_ras = streamlinePositions[seedIdxRange,1,]
        #candidatePosition_ras = candidatePosition_ras[None, ...]
        candidatePosition_ijk = (M.dot(candidatePosition_ras.T) + abc).T
        for iter in range(1,noIterations):
            progress(iter * 100 / noIterations, text="{}/{} steps calculated... ".format(iter, noIterations))
            ####
            ####
            # estimate tangent at current position
            curStreamlinePos_ras = candidatePosition_ras
            curStreamlinePos_ijk = candidatePosition_ijk
            lastDirections =  (streamlinePositions[seedIdxRange,iter-1,] - streamlinePositions[seedIdxRange,iter,]) # previousPosition - currentPosition in RAS
            #lastDirections = lastDirections[None, ...]
            lastDirections_ijk = (streamlinePositions_ijk[seedIdxRange,iter-1,] - streamlinePositions_ijk[seedIdxRange,iter,])
            #lastDirections_ijk = lastDirections_ijk[None, ...]
            ####
            ####
            # compute direction
            ld_input = (lastDirections, lastDirections_ijk)
            
            predictedDirection, dwi_at_curPosition, _ = getNextDirection(myState, data, curPosition_ijk = curStreamlinePos_ijk, model = model, x_ = x_, y_ = y_, z_ = z_, rot_ijk_val = rot_ijk_val)
            _, _, pContinueTracking = getNextDirection(myState, data, curPosition_ijk = curStreamlinePos_ijk, model = mlp_model, x_ = x_, y_ = y_, z_ = z_, rot_ijk_val = rot_ijk_val)
            
            validPoints = np.round(pContinueTracking > myState.pStopTracking)
             #vNorms[:,iter,] = vecNorms
            
            ####
            ####
            # compute next streamline position and check if this position is valid wrt. our stopping criteria1
            candidatePosition_ras, candidatePosition_ijk = makeStep(myState, predictedDirection = predictedDirection,  curStreamlinePos_ras = curStreamlinePos_ras, M = M, abc = abc, start_time = start_time, printfProfiling=printfProfiling, M2 = M2, abc2 = abc2, curStreamlinePos_ijk = curStreamlinePos_ijk, lastDirections = lastDirections_ijk)
            
            for i in range(len(validPoints)):
                if(validPoints[i] == 0.):
                    indexLastStreamlinePosition[i] = np.min((indexLastStreamlinePosition[i], iter))
            if np.sum(validPoints) == 0:
                break

            streamlinePositions[seedIdxRange, iter + 1,] = candidatePosition_ras
            streamlinePositions_ijk[seedIdxRange, iter + 1,] = candidatePosition_ijk

    #CHANGED
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

    prob_forward = streamlinePositions[0:noSeeds]
    prob_backward = streamlinePositions[noSeeds:]

    # join directions
    streamlines = Streamlines(joinTwoAlignedStreamlineLists(sl_backward, sl_forward))

    return streamlines, vNorms
# WORKING CODE
# DO NOT CHANGE, JUST COPY AND CHANGE THEN
def startWithRNNBatches(seeds, myState, data, model, affine, mask, printProgress = False, inverseDirection = False, noIterations = 200, rot_ijk_val = None, batchSize=2**16):
    ''' #
    fibre tracking using neural networks
    '''
    printfProfiling = False
    mask = mask.astype(np.float)

    noSeeds = len(seeds)
    print("Initalize Streamline positions data")
    # initialize streamline positions data
    vNorms = np.zeros([2*noSeeds,noIterations+1])
    streamlinePositions = np.zeros([2*noSeeds,noIterations+1,3])
    streamlinePositions_ijk = np.zeros([2*noSeeds,noIterations+1,3])

    streamlinePositions[0:noSeeds,0,] = seeds[0:noSeeds] ## FORWARD
    streamlinePositions[noSeeds:,1,] = seeds[0:noSeeds] ## BACKWARD

    indexLastStreamlinePosition = noIterations * np.ones([2*noSeeds], dtype=np.intp)

    #del seeds # its not supposed to use the seeds anymore

    # interpolate data given these coordinates for each channel
    x_,y_,z_ = dwi_tools._getCoordinateGrid(myState)
    
    # prepare transformations to project a streamline point from IJK into RAS coordinate system to interpolate DWI data
    aff_ras_ijk = np.linalg.inv(affine) # aff: IJK -> RAS
    M = aff_ras_ijk[:3, :3]
    abc = aff_ras_ijk[:3, 3]
    abc = abc[:,None]

    aff_ijk_ras = affine # affine: IJK -> RAS
    M2 = aff_ijk_ras[:3, :3]
    abc2 = aff_ijk_ras[:3, 3]
    abc2 = abc2[:,None]
    lastDirections = np.zeros([noSeeds,3])
    print("First step prediction")
    for seedIdx in range(0,noSeeds,batchSize):
        print('first step Streamline %d / %d ' % (seedIdx, noSeeds))
        if isinstance(model, ModelLSTM):
            model.reset()
        seedIdxRange = range(seedIdx, min(seedIdx+batchSize, noSeeds))    
        seedIdxRange2 = range(seedIdx + noSeeds, min(seedIdx+batchSize, noSeeds) + noSeeds)    
        ### FIRST STEP PREDICTION ###
        # just predict the forward direction (the first half of streamlinePositions)
        curStreamlinePos_ras = streamlinePositions[seedIdxRange,0,]
        curStreamlinePos_ijk = (M.dot(curStreamlinePos_ras.T) + abc).T
        

        streamlinePositions_ijk[seedIdxRange,0,] = curStreamlinePos_ijk ## FORWARD
        streamlinePositions_ijk[(seedIdxRange2),1,] = curStreamlinePos_ijk ## BACKWARD

        ld_input = lastDirections
        start_time = time.time()
        # never rotate the data of the first step
        oldRotationState = myState.rotateData
        myState.rotateData = False
        # predict direction but in never rotate data
        predictedDirection, dwi_at_curPosition, pContinueTracking = getNextDirection(myState, data, curPosition_ijk = curStreamlinePos_ijk, model = model, x_ = x_, y_ = y_, z_ = z_, rot_ijk_val = rot_ijk_val)
        myState.rotateData = oldRotationState
        # compute next streamline position
        candidatePosition_ras, candidatePosition_ijk = makeStep(myState, predictedDirection = predictedDirection,  curStreamlinePos_ras = curStreamlinePos_ras, M = M, abc = abc, start_time = start_time, printfProfiling=printfProfiling, M2 = M2, abc2 = abc2, curStreamlinePos_ijk = curStreamlinePos_ijk, lastDirections = lastDirections)

        # update positions
        streamlinePositions[seedIdxRange,1,] = candidatePosition_ras
        streamlinePositions[(seedIdxRange2),0,] = candidatePosition_ras
        streamlinePositions_ijk[seedIdxRange,1,] = candidatePosition_ijk
        streamlinePositions_ijk[seedIdxRange2,0,] = candidatePosition_ijk
    candidatePosition_ras = streamlinePositions[:,1,]
    candidatePosition_ijk = (M.dot(candidatePosition_ras.T) + abc).T

    ### START ITERATION UNTIL NOITERATIONS REACHED ###
    # tracking steps are done in RAS
    validSls = None
   

    for seedIdx in range(0,2*noSeeds,batchSize):
        print('Streamline %d / %d ' % (seedIdx, 2*noSeeds))
        if isinstance(model, ModelLSTM):
            model.reset()
        seedIdxRange = range(seedIdx, min(seedIdx+batchSize, 2*noSeeds))
        candidatePosition_ras = streamlinePositions[seedIdxRange,1,]
        #candidatePosition_ras = candidatePosition_ras[None, ...]
        candidatePosition_ijk = (M.dot(candidatePosition_ras.T) + abc).T
        for iter in range(1,noIterations):
            progress(iter * 100 / noIterations, text="{}/{} steps calculated... ".format(iter, noIterations))
            ####
            ####
            # estimate tangent at current position
            curStreamlinePos_ras = candidatePosition_ras
            curStreamlinePos_ijk = candidatePosition_ijk
            lastDirections =  (streamlinePositions[seedIdxRange,iter-1,] - streamlinePositions[seedIdxRange,iter,]) # previousPosition - currentPosition in RAS
            #lastDirections = lastDirections[None, ...]
            lastDirections_ijk = (streamlinePositions_ijk[seedIdxRange,iter-1,] - streamlinePositions_ijk[seedIdxRange,iter,])
            #lastDirections_ijk = lastDirections_ijk[None, ...]
            ####
            ####
            # compute direction
            ld_input = (lastDirections, lastDirections_ijk)
            
            predictedDirection, dwi_at_curPosition, pContinueTracking = getNextDirection(myState, data, curPosition_ijk = curStreamlinePos_ijk, model = model, x_ = x_, y_ = y_, z_ = z_, rot_ijk_val = rot_ijk_val)
            
            validPoints = np.round(pContinueTracking > myState.pStopTracking)
             #vNorms[:,iter,] = vecNorms
            
            ####
            ####
            # compute next streamline position and check if this position is valid wrt. our stopping criteria1
            candidatePosition_ras, candidatePosition_ijk = makeStep(myState, predictedDirection = predictedDirection,  curStreamlinePos_ras = curStreamlinePos_ras, M = M, abc = abc, start_time = start_time, printfProfiling=printfProfiling, M2 = M2, abc2 = abc2, curStreamlinePos_ijk = curStreamlinePos_ijk, lastDirections = lastDirections_ijk)
            
            for i in range(len(validPoints)):
                if(validPoints[i] == 0.):
                    indexLastStreamlinePosition[i] = np.min((indexLastStreamlinePosition[i], iter))
            if np.sum(validPoints) == 0:
                break

            streamlinePositions[seedIdxRange, iter + 1,] = candidatePosition_ras
            streamlinePositions_ijk[seedIdxRange, iter + 1,] = candidatePosition_ijk
    

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

    prob_forward = streamlinePositions[0:noSeeds]
    prob_backward = streamlinePositions[noSeeds:]

    # join directions
    streamlines = Streamlines(joinTwoAlignedStreamlineLists(sl_backward, sl_forward))

    return streamlines, vNorms

# DOESN'T WORK
# SEE METHOD ABOVE FOR WORKING, EQUAL CODE
# HOWEVER, IF YOU FIND THE CAUSE FOR THE ERROR HERE, THAT WOULD BE GREAT :)
def start(myState, seeds, data, model, affine, mask, batch_size = 128, printProgress = False, inverseDirection = False, noIterations = 115, rot_ijk_val = None):
    '''
    fibre tracking using neural networks
    '''
    #mask = mask.astype(np.float)
    noSeeds = len(seeds)
    printfProfiling = False
    # initialize streamline positions data
    vNorms = np.zeros([2*noSeeds,noIterations+1])
    streamlinePositions = np.zeros([2*noSeeds,noIterations+1,3])
    streamlinePositions_ijk = np.zeros([2*noSeeds,noIterations+1,3])
    lastDirections = np.zeros([noSeeds,3])
    
    streamlinePositions[0:noSeeds,0,] = seeds[0:noSeeds] ## FORWARD
    streamlinePositions[noSeeds:,1,] = seeds[0:noSeeds] ## BACKWARD

    indexLastStreamlinePosition = noIterations * np.ones([2*noSeeds],dtype=np.int64)

    # interpolate data given these coordinates for each channel
    x_,y_,z_ = dwi_tools._getCoordinateGrid(myState)

    # prepare transformations to project a streamline point from IJK into RAS coordinate system to interpolate DWI data
    aff_ras_ijk = np.linalg.inv(affine) # aff: IJK -> RAS
    M = aff_ras_ijk[:3, :3]
    abc = aff_ras_ijk[:3, 3]
    abc = abc[:,None]

    aff_ijk_ras = affine # affine: IJK -> RAS
    M2 = aff_ijk_ras[:3, :3]
    abc2 = aff_ijk_ras[:3, 3]
    abc2 = abc2[:,None]

    ### START ITERATION UNTIL NOITERATIONS REACHED ###
    # tracking is done in RAS
    validSls = None
    start_time = time.time()

    for seedIdx in range(0, noSeeds, batch_size):
        print('Streamlines %d-%d / %d ' % (seedIdx, min(seedIdx+batch_size, noSeeds), noSeeds))
        
        seedIdxRange = range(seedIdx,min(seedIdx+batch_size, noSeeds))

        candidatePosition_ras = seeds[seedIdxRange]
        candidatePosition_ijk = (M @ candidatePosition_ras.T + abc).T
        
        # reset state of our recurrent network
        #model.reset_states()
        if isinstance(model, ModelLSTM):
            model.reset()
        ##############
        ##############
        # forward pass
        ##############
        ##############
        
        for iter in range(noIterations):
            ####
            ####
            # estimate tangent at current position
            curStreamlinePos_ras = candidatePosition_ras
            curStreamlinePos_ijk = candidatePosition_ijk
            ####
            ####
            # prediction direction
            predictedDirection, dwi_at_curPosition, stopTrackingProbability = getNextDirection(myState, data, curPosition_ijk = curStreamlinePos_ijk, model = model, x_ = x_, y_ = y_, z_ = z_, rot_ijk_val = rot_ijk_val)

            ####
            ####
            # compute next streamline position and check if this position is valid wrt. our stopping criteria
            candidatePosition_ras, candidatePosition_ijk = makeStep(myState, predictedDirection = predictedDirection,  curStreamlinePos_ras = curStreamlinePos_ras, M = M, abc = abc, start_time = start_time, printfProfiling=printfProfiling, M2 = M2, abc2 = abc2, curStreamlinePos_ijk = curStreamlinePos_ijk, lastDirections = lastDirections)
            
            validPoints = np.round(stopTrackingProbability > myState.pStopTracking)
            for k in range(len(stopTrackingProbability)):
                if(validPoints[k] == 0.):
                    indexLastStreamlinePosition[seedIdx+k] = np.min((indexLastStreamlinePosition[seedIdx+k], iter))
            
            if(sum(validPoints) == 0):
                break
            
            streamlinePositions[seedIdxRange, iter + 1,] = candidatePosition_ras
            streamlinePositions_ijk[seedIdxRange, iter + 1,] = candidatePosition_ijk


        ##############
        ##############
        ## backward pass
        ##############
        ##############
        if isinstance(model, ModelLSTM):
            model.reset()
            
        streamlinePositions[seedIdx+noSeeds, 0,] = streamlinePositions[seedIdx, 1,]
        streamlinePositions[seedIdx+noSeeds, 1,] = streamlinePositions[seedIdx, 0,]
        candidatePosition_ras = streamlinePositions[seedIdx+noSeeds, 1,]
        candidatePosition_ras = candidatePosition_ras[np.newaxis, ...]
        candidatePosition_ijk = (M.dot(candidatePosition_ras.T) + abc).T

        seedIdx = seedIdx + noSeeds
        seedIdxRange = range(seedIdx, min(seedIdx + batch_size, 2*noSeeds))

        for iter in range(noIterations):
            ####
            ####
            # estimate tangent at current position
            curStreamlinePos_ras = candidatePosition_ras
            curStreamlinePos_ijk = candidatePosition_ijk
            #print("Length cur_pos: {}".format(len(curStreamlinePos_ijk)))
        
            ####
            ####
            # prediction direction
            predictedDirection, dwi_at_curPosition, stopTrackingProbability = getNextDirection(myState, data, curPosition_ijk = curStreamlinePos_ijk, model = model, x_ = x_, y_ = y_, z_ = z_, rot_ijk_val = rot_ijk_val)

            ####
            ####
            # compute next streamline position and check if this position is valid wrt. our stopping criteria
            candidatePosition_ras, candidatePosition_ijk = makeStep(myState, predictedDirection = predictedDirection,  curStreamlinePos_ras = curStreamlinePos_ras, M = M, abc = abc, start_time = start_time, printfProfiling=printfProfiling, M2 = M2, abc2 = abc2, curStreamlinePos_ijk = curStreamlinePos_ijk, lastDirections = lastDirections)
            
            validPoints = np.round(stopTrackingProbability > myState.pStopTracking)
            for k in range(len(stopTrackingProbability)):
                if(validPoints[k] == 0.):
                    indexLastStreamlinePosition[seedIdx+k] = np.min((indexLastStreamlinePosition[seedIdx], iter))
            
            if(sum(validPoints) == 0):
                break
            
            streamlinePositions[seedIdxRange, iter + 1,] = candidatePosition_ras
            streamlinePositions_ijk[seedIdxRange, iter + 1,] = candidatePosition_ijk
        
            
        ###
        seedIdx = seedIdx - noSeeds
        ###
        # continue with next streamline

    streamlinePositions = streamlinePositions.tolist()

    ####
    ####
    #
    # crop streamlines to length specified by stopping criteria

    for seedIdx in range(0,2*noSeeds):
        currentStreamline = np.array(streamlinePositions[seedIdx])
        currentStreamline = currentStreamline[0:indexLastStreamlinePosition[seedIdx],]

        streamlinePositions[seedIdx] = currentStreamline

    # extract both directions
    sl_forward = streamlinePositions[0:noSeeds]
    sl_backward = streamlinePositions[noSeeds:]

    prob_forward = streamlinePositions[0:noSeeds]
    prob_backward = streamlinePositions[noSeeds:]

    # join directions
    streamlines = Streamlines(joinTwoAlignedStreamlineLists(sl_backward, sl_forward))
    #streamlines = Streamlines(sl_forward)
    return streamlines, vNorms
