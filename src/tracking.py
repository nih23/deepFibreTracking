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


def makeStep(myState, predictedDirection,lastDirections,curStreamlinePos_ras,M,abc,start_time=0,printfProfiling=False):
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
    candidatePosition_ras = curStreamlinePos_ras - myState.stepWidth * predictedDirection
    candidatePosition_ijk = (M.dot(candidatePosition_ras.T) + abc).T   #projectRAStoIJK(candidatePosition,M,abc)
    
    return candidatePosition_ras, candidatePosition_ijk


def getNextDirection(myState, dwi,curPosition_ijk, model, lastDirections = None, batch_size = 2**12, x_ = [0], y_ = [0], z_ = [0], validIdx = None):
    rot = None

    if(validIdx is None):
        validIdx = list(range(len(curPosition_ijk)))
       
    if(myState.rotateData):
        #print('rotating data')
        start_time = time.time()
        # reference orientation
        vv = dwi_tools.getReferenceOrientation()

        noPositions = len(lastDirections)

        # compute rotation matrices
        rot = np.zeros([noPositions,3,3])

        for k in range(noPositions):
            dwi_tools.R_2vect(rot[k,:,:],vector_orig=lastDirections[k,],vector_fin=vv)
        print(" -> " + str(time.time() - start_time) + "s")

    dwi_at_curPosition = dwi_tools.interpolateDWIVolume(myState, dwi, curPosition_ijk[validIdx,], x_,y_,z_, rotations = rot)

    if(myState.use2DProjection):
        noSamples,dx,dy,dz,dw = dwi_at_curPosition.shape
        dim = np.concatenate([[-1,], model.get_input_shape_at(0)[1:]])
        dwi_at_curPosition = np.reshape(dwi_at_curPosition, dim)

    predictedDirection = np.zeros([len(curPosition_ijk),3])
    vecNorms = np.zeros([len(curPosition_ijk)])

    if(myState.usePreviousDirection == False):
        networkInput = dwi_at_curPosition
    else:
        if(myState.use2DProjection):
            # CNN model w/ previous direction is different than the MLP as it doesnt require the last streamline vector but the actual DWI data at the previous streamline position.
            # we hacked that into the last direction :D
            dwi_at_prev_and_curPosition = np.stack((np.squeeze(lastDirections), np.squeeze(dwi_at_curPosition)))
            dwi_at_prev_and_curPosition = np.moveaxis(dwi_at_prev_and_curPosition,0,-1)
            networkInput = dwi_at_prev_and_curPosition
        else:
            lastDirections = -1 * lastDirections
            networkInput = [dwi_at_curPosition, lastDirections]

    predictedDirectionAtIdx = model.predict(networkInput, batch_size=batch_size)
        
    if(len(predictedDirectionAtIdx) == 2):
        #multi output models
        predictedDirectionAtIdx = predictedDirectionAtIdx[0]

    vecNorms[validIdx] = np.sqrt(np.sum(predictedDirectionAtIdx ** 2 , axis = 1))
    predictedDirection[validIdx,] = np.nan_to_num(predictedDirectionAtIdx / vecNorms[validIdx,None])
        
    return predictedDirection, vecNorms, dwi_at_curPosition


def getNextDirectionMagicModel(myState, dwi,curPosition_ijk, model, lastDirections = None, x_ = [0], y_ = [0], z_ = [0], batch_size = 2**10, validIdx = None):
    
    if(validIdx is None):
        validIdx = list(range(len(curPosition_ijk)))

    rot = None
       
    if(myState.rotateData):
        #print('rotating data')
        start_time = time.time()
        # reference orientation
        vv = myState.getReferenceOrientation()

        noPositions = len(lastDirections)

        # compute rotation matrices
        rot = np.zeros([noPositions,3,3])

        for k in range(noPositions):
            dwi_tools.R_2vect(rot[k,:,:],vector_orig=lastDirections[k,],vector_fin=vv)
        print(" -> " + str(time.time() - start_time) + "s")

    if(not rot == None):
        rot = rot[validIdx,]

    dwi_at_curPosition = dwi_tools.interpolateDWIVolume(myState, dwi, curPosition_ijk[validIdx,], x_,y_,z_, rotations = rot)
       
    if(myState.use2DProjection):
        noSamples,dx,dy,dz,dw = dwi_at_curPosition.shape
        dim = np.concatenate([[-1,], model.get_input_shape_at(0)[0][1:]])
        dwi_at_curPosition = np.reshape(dwi_at_curPosition, dim)
    
    predictedDirection = np.zeros([len(curPosition_ijk),3])
    stopTrackingProbability = np.zeros([len(curPosition_ijk),1])
    
    
    if(not myState.usePreviousDirection):
        predictedDirectionAtIdx, stopTrackingProbabilityAtIdx = model.predict([dwi_at_curPosition], batch_size = batch_size)
    else:
        lastDirections = lastDirections[validIdx,]
        if(myState.use2DProjection):
            # CNN model w/ previous direction is different than the MLP as it doesnt require the last streamline vector but the actual DWI data at the previous streamline position.
            # we hacked that into the last direction :D
            dwi_at_prev_and_curPosition = np.stack((np.squeeze(lastDirections), np.squeeze(dwi_at_curPosition)))
            dwi_at_prev_and_curPosition = np.moveaxis(dwi_at_prev_and_curPosition,0,-1)
            predictedDirectionAtIdx, stopTrackingProbabilityAtIdx = model.predict([dwi_at_prev_and_curPosition], batch_size = batch_size)
        else:
            lastDirections = -1 * lastDirections
            predictedDirectionAtIdx, stopTrackingProbabilityAtIdx = model.predict([dwi_at_curPosition, lastDirections], batch_size = batch_size)

    if(len(predictedDirectionAtIdx) == 2):
        predictedDirectionAtIdx = predictedDirectionAtIdx[0]
            
    predictedDirection[validIdx,] = predictedDirectionAtIdx
    stopTrackingProbability[validIdx,] = stopTrackingProbabilityAtIdx
    #predictedDirection = predictedDirection * 2 - 1
        
    vecNorms = np.sqrt(np.sum(predictedDirection ** 2 , axis = 1))
    predictedDirection = np.nan_to_num(predictedDirection / vecNorms[:,None])
    
    stopTrackingProbability = np.squeeze(stopTrackingProbability)
        
    return predictedDirection, vecNorms, dwi_at_curPosition, stopTrackingProbability


def start(myState, seeds, data, model, affine, mask, fa, printProgress = False, nverseDirection = False, printfProfiling = False, noIterations = 200):
    '''
    fibre tracking using neural networks
    '''    
    
    print('Reshape: ' + str(myState.use2DProjection))
    
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
    x_,y_,z_ = dwi_tools._getCoordinateGrid(myState)

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

    # predict direction but in never rotate data
    if(myState.magicModel):
        predictedDirection, vecNorms, dwi_at_curPosition, stopTrackingProbability = getNextDirectionMagicModel(myState,data,curPosition_ijk=curStreamlinePos_ijk,model=model,lastDirections=ld_i,x_=x_,y_=y_,z_=z_)
    else:
        predictedDirection, vecNorms, dwi_at_curPosition = getNextDirection(myState, data, curPosition_ijk = curStreamlinePos_ijk, model = model, lastDirections = ld_input, x_ = x_, y_ = y_, z_ = z_)
        
    # compute next streamline position
    candidatePosition_ras, candidatePosition_ijk = makeStep(myState, printfProfiling=printfProfiling, predictedDirection = predictedDirection, lastDirections = lastDirections, curStreamlinePos_ras = curStreamlinePos_ras, M = M, abc = abc)
    
    # update positions
    streamlinePositions[0:noSeeds,1,] = candidatePosition_ras
    streamlinePositions[noSeeds:,0,] = candidatePosition_ras
    candidatePosition_ras = streamlinePositions[:,1,]
    candidatePosition_ijk = (M.dot(candidatePosition_ras.T) + abc).T
    
    if(myState.use2DProjection and myState.usePreviousDirection):
        cp = streamlinePositions[:,0,] 
        cp_ijk = (M.dot(cp.T) + abc).T
        dwi_at_curPosition = dwi_tools.interpolateDWIVolume(myState, data, cp_ijk, x_,y_,z_)
        noSamples,dx,dy,dz,dw = dwi_at_curPosition.shape
        dwi_at_curPosition = np.reshape(dwi_at_curPosition, [noSamples,16,16])
        dwi_at_curPosition = dwi_at_curPosition[..., np.newaxis]
    
    ### START ITERATION UNTIL NOITERATIONS REACHED ###
    validSls = None
    start_time = time.time()
    for iter in range(1,noIterations):
        if(printProgress and iter > 1):
            print(str(iter-1) + "/" + str(noIterations) + " [" + str(time.time() - start_time) + "s]")
        ####
        ####
        # compute current position and last direction
        curStreamlinePos_ras = candidatePosition_ras
        curStreamlinePos_ijk = candidatePosition_ijk
        lastDirections = (streamlinePositions[:,iter-1,] - streamlinePositions[:,iter,]) # previousPosition - currentPosition
        
        ####
        ####
        # compute direction
        ld_input = lastDirections
#        if(reshapeForConvNet):
#            ld_input = dwi_at_curPosition
        if(myState.magicModel):
            predictedDirection, vecNorms, dwi_at_curPosition, stopTrackingProbability = getNextDirectionMagicModel(
                myState, data, curPosition_ijk=curStreamlinePos_ijk, model=model, lastDirections=ld_input,
                validIdx=validSls, x_=x_, y_=y_, z_=z_)
            validPoints = np.greater(stopTrackingProbability > myState.pStopTracking)
            print('Average valid prob %.2f (valid ratio %.2f)' % (np.mean(stopTrackingProbability), len(
                np.where(stopTrackingProbability > myState.pStopTracking)[0]) / (2 * noSeeds)))
        else:
            predictedDirection, vecNorms, dwi_at_curPosition = getNextDirection(myState, data, curPosition_ijk = curStreamlinePos_ijk, model = model, lastDirections = ld_input, x_ = x_, y_ = y_, z_ = z_, validIdx = validSls)

        vNorms[:,iter,] = vecNorms
        
        ####
        ####
        # compute next streamline position and check if this position is valid wrt. our stopping criteria1
        candidatePosition_ras, candidatePosition_ijk = makeStep(myState, predictedDirection = predictedDirection, lastDirections = lastDirections, curStreamlinePos_ras = curStreamlinePos_ras, M = M, abc = abc, start_time = start_time, printfProfiling=printfProfiling)

        if(not myState.magicModel):
            validPoints = areVoxelsValidStreamlinePoints(candidatePosition_ijk, mask, fa, myState.faThreshold)

        validSls = np.where(validPoints == 1)[0]

        if (len(validSls) == 0):
            break
        
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
