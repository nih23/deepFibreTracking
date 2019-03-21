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


def makeStep(myState, predictedDirection,lastDirections,curStreamlinePos_ijk,curStreamlinePos_ras,M,abc,M2,abc2,start_time=0,printfProfiling=False):
    #M2, abc2: IJK -> ras
    ####
    ####
    # check predicted direction and flip it if necessary
    noSeeds = len(predictedDirection)

    if(printfProfiling):
        print(" -> 3 " + str(time.time() - start_time) + "s]")

    if(myState.predictionInIJK):
        lastDirections = lastDirections[1]
    else:
        lastDirections = lastDirections[0]

    #predictedDirection[np.where(theta < 0),] = -1 * predictedDirection[np.where(theta < 0),] ### probably a little bit faster
    for j in range(0,noSeeds):
        lv1 = predictedDirection[j,]
        pv1 = lastDirections[j,]
        #print(str(lv1) + '--' + str(pv1))
        #theta = np.arcsin(np.dot(pv1,lv1) / (np.linalg.norm(pv1)*np.linalg.norm(lv1)))

        theta = np.dot(pv1,lv1)

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


def getNextDirection(myState, dwi, curPosition_ijk, model, lastDirections_ras = None, batch_size =2 ** 10, x_ = [0], y_ = [0], z_ = [0], validIdx = None, rnnModel = None):
    # TODO: changed rotation such that the input data is rotated wrt to the inverse rotation matrix and the predicted direction is rotated back with the actual rotation matrix
    rot_ijk_val = None
    bv_ijk_val = None
    if(len(lastDirections_ras) == 2):
       lastDirections_ijk = lastDirections_ras[1]
       lastDirections_ras = lastDirections_ras[0]

    if(validIdx is None):
        validIdx = list(range(len(curPosition_ijk)))


    ###################
    ### compute rotation matrices
    ###################
    if(myState.rotateData):
        print('   rotating data')
        start_time = time.time()
        # reference orientation
        vv = myState.getReferenceOrientation()

        noPositions = len(lastDirections_ras)
        noValPositions = len(validIdx)
        # compute rotation matrices
        rot_ijk_val = np.zeros([noValPositions,3,3])
        bv_ijk_val = np.zeros([noValPositions,len(myState.bvecs),3])
        for k in range(noValPositions):
            valIdx = validIdx[k]
            #print(str(lastDirections_ijk[valIdx,].shape))
            dwi_tools.R_2vect(rot_ijk_val[k, :, :], vector_orig=vv, vector_fin=lastDirections_ijk[valIdx,]) # fixed 01/24/19. see @dwi_tools
            rot_ijk_val[k, :, :] = rot_ijk_val[k, :, :].T #TODO 02/20/19: bug fix by using the inverse?
            bv_ijk_val[k,] = np.dot(rot_ijk_val[k,], myState.bvecs.T).T

    if(rot_ijk_val is not None):
        i = np.min([100,len(validIdx)-1])
        valIdx = validIdx[i]
        ldv = lastDirections_ijk[valIdx,]
        print(str(ldv))
        print(str(rot_ijk_val[i, :]))
        print(str(np.dot(rot_ijk_val[i, :], ldv.T).T))

    ###################
    ### interpolate, rotate
    ### and project data
    ###################
    dwi_at_curPosition = dwi_tools.interpolateDWIVolume(myState, dwi, curPosition_ijk[validIdx,], x_,y_,z_, rotations_ijk = rot_ijk_val)
    dwi_at_curPosition = dwi_tools.projectIntoAppropriateSpace(myState, dwi_at_curPosition)

    if(myState.use2DProjection):
        noSamples,dx,dy,dz,dw = dwi_at_curPosition.shape
        dim = np.concatenate([[-1,], model.get_input_shape_at(0)[1:]])
        dwi_at_curPosition = np.reshape(dwi_at_curPosition, dim)

    predictedDirection = np.zeros([len(curPosition_ijk),3])
    stopTrackingProbability = np.zeros([len(curPosition_ijk), 1])

    vecNorms = np.zeros([len(curPosition_ijk)])

    if(myState.usePreviousDirection == False):
        i1 = myState.bvecs
        i1 = np.concatenate([i1, [[0., 0., 0.]]])
        if(not bv_ijk_val is None):
            networkInput = [dwi_at_curPosition, bv_ijk_val]
        else:
            networkInput = [dwi_at_curPosition, np.repeat(i1[None, :], len(validIdx), axis=0)]
        #networkInput = [dwi_at_curPosition, np.repeat(i1[None, :], len(validIdx), axis=0)]

        networkInput = dwi_at_curPosition
    else:
        if(myState.use2DProjection):
            # CNN model w/ previous direction is different than the MLP as it doesnt require the last streamline vector but the actual DWI data at the previous streamline position.
            # we hacked that into the last direction :D
            if(myState.predictionInIJK):
                dwi_at_prev_and_curPosition = np.stack((np.squeeze(lastDirections_ijk), np.squeeze(dwi_at_curPosition)))
            else:
                dwi_at_prev_and_curPosition = np.stack((np.squeeze(lastDirections_ras), np.squeeze(dwi_at_curPosition)))
            dwi_at_prev_and_curPosition = np.moveaxis(dwi_at_prev_and_curPosition,0,-1)
            networkInput = dwi_at_prev_and_curPosition
        else:
            if (myState.predictionInIJK):
                lastDirections_ijk = -1 * lastDirections_ijk
                networkInput = [dwi_at_curPosition, lastDirections_ijk]
            else:
                lastDirections_ras = -1 * lastDirections_ras
                networkInput = [dwi_at_curPosition, lastDirections_ras]

    ###################
    ### make prediction
    ###################
    predictedDirectionAtIdx = model.predict(networkInput, batch_size=batch_size)

    if(myState.magicModel):
        stopTrackingProbability[validIdx,] = predictedDirectionAtIdx[1]
        predictedDirectionAtIdx = predictedDirectionAtIdx[0]

    if(model.layers[-1].output_shape[1] > 3):
        # discretization-based model
        predictedDirectionAtIdx = np.argmax(predictedDirectionAtIdx,axis=1)
        sphere = get_sphere('repulsion100')
        predictedDirectionAtIdx = dwi_tools.projectDiscretizedTangentsBack(bvecs=sphere.vertices, label=predictedDirectionAtIdx)
        predictedDirectionAtIdx = np.array(predictedDirectionAtIdx)

    if(len(predictedDirectionAtIdx.shape) == 1):
        predictedDirectionAtIdx = predictedDirectionAtIdx[None, :]

    mupred = np.mean(predictedDirectionAtIdx, axis=0)
    stdpred = np.std(predictedDirectionAtIdx, axis=0)
    print(" prediction stats %.3f+-%.3f %.3f+-%.3f %.3f+-%.3f" % (mupred[0], stdpred[0], mupred[1], stdpred[1], mupred[2], stdpred[2]))
    ###################
    ### postprocess prediction
    ###################

    vNorms = np.sqrt(np.sum(predictedDirectionAtIdx ** 2 , axis = 1))
    predictedDirectionAtIdx = np.nan_to_num(predictedDirectionAtIdx / vNorms[:,None])

    mupred = np.mean(predictedDirectionAtIdx, axis=0)
    stdpred = np.std(predictedDirectionAtIdx, axis=0)
    print(" prediction stats 0 %.3f+-%.3f %.3f+-%.3f %.3f+-%.3f" % (mupred[0], stdpred[0], mupred[1], stdpred[1], mupred[2], stdpred[2]))

    if(not rnnModel is None):
        predictedDirectionAtIdx = predictedDirectionAtIdx[None, ...]
        predictedDirectionAtIdx = rnnModel.predict(predictedDirectionAtIdx, verbose=0)

#        mupred2 = np.mean(predictedDirectionAtIdx, axis=0)
#        stdpred2 = np.std(predictedDirectionAtIdx, axis=0)

#        print(" prediction stats rnn %.3f+-%.3f %.3f+-%.3f %.3f+-%.3f" % (
#            mupred2[0], stdpred2[0], mupred2[1], stdpred2[1], mupred2[2], stdpred2[2]))

    # the output of the neural network is rotated just as the input meaning that we need to rotate it back to get the
    # direction in standard IJK coordinate system
    if(myState.rotateData and myState.resampleDWIAfterRotation):
#        print('   rotating prediction of the neural network back :D')
#        start_time = time.time()

        for i in range(noValPositions):
            predictedDirectionAtIdx[i,] = np.dot(rot_ijk_val[i, :].T, predictedDirectionAtIdx[i,].T).T # transpose is inverse in case of rotation matrices

#        print("     -> " + str(time.time() - start_time) + "s")
#        mupred2 = np.mean(predictedDirectionAtIdx, axis=0)
#        stdpred2 = np.std(predictedDirectionAtIdx, axis=0)

#        print(" prediction stats  %.3f+-%.3f %.3f+-%.3f %.3f+-%.3f" % (mupred2[0], stdpred2[0], mupred2[1], stdpred2[1], mupred2[2], stdpred2[2]))
    vecNorms[validIdx] = vNorms
    predictedDirection[validIdx,] = predictedDirectionAtIdx

#    if (myState.rotateData):
#        print("     -> " + str(time.time() - start_time) + "s")
#        mupred = np.mean(predictedDirection[validIdx,], axis=0)
#        stdpred = np.std(predictedDirection[validIdx,], axis=0)
#        print(" prediction stats2 %.3f+-%.3f %.3f+-%.3f %.3f+-%.3f" % (
#        mupred[0], stdpred[0], mupred[1], stdpred[1], mupred[2], stdpred[2]))

    return predictedDirection, vecNorms, dwi_at_curPosition, stopTrackingProbability


def start(myState, seeds, data, model, affine, mask, fa, printProgress = False, printfProfiling = False, noIterations = 200):
    '''
    fibre tracking using neural networks
    '''

    mask = mask.astype(np.float)

    noSeeds = len(seeds)

    # initialize streamline positions data
    vNorms = np.zeros([2*noSeeds,noIterations+1])
    vProbs = np.zeros([2 * noSeeds, noIterations + 1])
    streamlinePositions = np.zeros([2*noSeeds,noIterations+1,3])
    streamlinePositions_ijk = np.zeros([2*noSeeds,noIterations+1,3])

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

    aff_ijk_ras = affine # aff: IJK -> RAS
    M2 = aff_ijk_ras[:3, :3]
    abc2 = aff_ijk_ras[:3, 3]
    abc2 = abc2[:,None]

    ### FIRST STEP PREDICTION ###
    # just predict the forward direction (the first half of streamlinePositions)
    curStreamlinePos_ras = streamlinePositions[0:noSeeds,0,]
    curStreamlinePos_ijk = (M.dot(curStreamlinePos_ras.T) + abc).T
    lastDirections = np.zeros([noSeeds,3])

    streamlinePositions_ijk[0:noSeeds,0,] = curStreamlinePos_ijk ## FORWARD
    streamlinePositions_ijk[noSeeds:,1,] = curStreamlinePos_ijk ## BACKWARD

    ld_input = [lastDirections, lastDirections]

    # never rotate the data of the first step
    oldRotationState = myState.rotateData
    myState.rotateData = False
    # predict direction but never rotate data
    #rotate wrt. to major direction?

    predictedDirection, vecNorms, dwi_at_curPosition, stopTrackingProbability = getNextDirection(myState, data, curPosition_ijk = curStreamlinePos_ijk, model = model, lastDirections_ras= ld_input, x_ = x_, y_ = y_, z_ = z_)

    myState.rotateData = oldRotationState
    # compute next streamline position
    candidatePosition_ras, candidatePosition_ijk = makeStep(myState, printfProfiling=printfProfiling, predictedDirection = predictedDirection, lastDirections = ld_input, curStreamlinePos_ras = curStreamlinePos_ras, curStreamlinePos_ijk = curStreamlinePos_ijk, M = M, abc = abc, M2=M2, abc2=abc2)

    # update positions
    streamlinePositions[0:noSeeds,1,] = candidatePosition_ras
    streamlinePositions[noSeeds:,0,] = candidatePosition_ras
    streamlinePositions_ijk[0:noSeeds,1,] = candidatePosition_ijk
    streamlinePositions_ijk[noSeeds:,0,] = candidatePosition_ijk
    candidatePosition_ras = streamlinePositions[:,1,]
    candidatePosition_ijk = (M.dot(candidatePosition_ras.T) + abc).T

    if(myState.use2DProjection and myState.usePreviousDirection):
        print('2D projection')
        cp = streamlinePositions[:,0,]
        cp_ijk = (M.dot(cp.T) + abc).T
        dwi_at_curPosition = dwi_tools.interpolateDWIVolume(myState, data, cp_ijk, x_,y_,z_)
        dwi_at_curPosition = dwi_tools.projectIntoAppropriateSpace(myState, dwi_at_curPosition)
        noSamples,dx,dy,dz,dw = dwi_at_curPosition.shape
        dwi_at_curPosition = np.reshape(dwi_at_curPosition, [noSamples,16,16])
        dwi_at_curPosition = dwi_at_curPosition[..., np.newaxis]


    ### START ITERATION UNTIL NOITERATIONS REACHED ###
    # tracking steps are done in RAS
    validSls = list(range(len(candidatePosition_ijk))) 
    start_time = time.time()
    for iter in range(1,noIterations):
        ####
        ####
        # estimate tangent at current position
        print('\nIteration %d' % iter)
        curStreamlinePos_ras = candidatePosition_ras
        curStreamlinePos_ijk = candidatePosition_ijk
        lastDirections =  (streamlinePositions[:,iter-1,] - streamlinePositions[:,iter,]) # previousPosition - currentPosition in RAS
        lastDirections_ijk = (streamlinePositions_ijk[:,iter-1,] - streamlinePositions_ijk[:,iter,])
        ####
        ####
        # estimate tangent
        ld_input = (lastDirections, lastDirections_ijk)
        predictedDirection, vecNorms, dwi_at_curPosition, stopTrackingProbability = getNextDirection(myState, data, curPosition_ijk = curStreamlinePos_ijk, model = model, lastDirections_ras= ld_input, x_ = x_, y_ = y_, z_ = z_, validIdx = validSls)
        vNorms[:,iter,] = vecNorms
        vProbs[:,iter,] = np.squeeze(stopTrackingProbability)
        print("PD" + str(predictedDirection[validSls[0],])) 
 
        ####
        ####
        # compute next streamline position and check if this position is valid wrt. our stopping criteria1
        candidatePosition_ras, candidatePosition_ijk = makeStep(myState, predictedDirection = predictedDirection, lastDirections = ld_input, curStreamlinePos_ras = curStreamlinePos_ras, M = M, abc = abc, M2=M2, abc2=abc2, start_time = start_time, printfProfiling=printfProfiling, curStreamlinePos_ijk = curStreamlinePos_ijk)
        print("CP" + str(curStreamlinePos_ijk[validSls[0],])) 
        print("CP" + str(curStreamlinePos_ras[validSls[0],]))
        if(myState.magicModel):
            validPoints = np.greater(stopTrackingProbability, myState.pStopTracking)
            print('Average valid prob %.2f (valid ratio %.2f)' % (np.mean(stopTrackingProbability), len(np.where(stopTrackingProbability > myState.pStopTracking)[0]) / (2 * noSeeds)))
        else:
            validPoints = areVoxelsValidStreamlinePoints(candidatePosition_ijk, mask, fa, myState.faThreshold)

        validSls = np.where(validPoints == 1)[0]

        if (len(validSls) == 0):
            break

        if(printfProfiling):
            print(" -> 5 " + str(time.time() - start_time) + "s]")

        if(printProgress):
            print("   valid ratio %d of %d (%.2f)" % (sum(validPoints), 2*noSeeds, float(sum(validPoints)) / float(2*noSeeds)))
            print("   runtime: " +  str(time.time() - start_time) + "s")


        for j in range(0,2*noSeeds):
            if(validPoints[j]):
                streamlinePositions[j,iter+1,] = candidatePosition_ras[j,]
                streamlinePositions_ijk[j,iter+1,] = candidatePosition_ijk[j,]
            else:
                #streamlinePositions[j,iter+1,] = candidatePosition_ras[j,]
                indexLastStreamlinePosition[j] = np.min((indexLastStreamlinePosition[j],iter))


    streamlinePositions = streamlinePositions.tolist()

    ####
    ####
    #
    # crop streamlines to length specified by stopping criteria
    vNorms = vNorms.tolist()
    vProbs = vProbs.tolist()

    for seedIdx in range(0,2*noSeeds):
        currentStreamline = np.array(streamlinePositions[seedIdx])
        currentStreamline = currentStreamline[0:indexLastStreamlinePosition[seedIdx],]

        currentNorm = np.array(vNorms[seedIdx])
        currentNorm = currentNorm[0:indexLastStreamlinePosition[seedIdx],]

        currentProb = np.array(vProbs[seedIdx])
        currentProb = currentProb[0:indexLastStreamlinePosition[seedIdx], ]

        streamlinePositions[seedIdx] = currentStreamline
        vNorms[seedIdx] = currentNorm
        vProbs[seedIdx] = currentProb

    # extract both directions
    sl_forward = streamlinePositions[0:noSeeds]
    sl_backward = streamlinePositions[noSeeds:]

    # extract both directions
    prob_forward = vProbs[0:noSeeds]
    prob_backward = vProbs[noSeeds:]

    # join directions
    streamlines = Streamlines(joinTwoAlignedStreamlineLists(sl_backward, sl_forward))
    probs = Streamlines(joinTwoAlignedStreamlineLists(prob_backward, prob_forward))

    return streamlines, vNorms, probs


def startWithRNN(myState, seeds, data, model, rnn_model, affine, mask, fa, printProgress = False, nverseDirection = False, printfProfiling = False, noIterations = 115):
    '''
    fibre tracking using neural networks
    '''

    mask = mask.astype(np.float)

    noSeeds = len(seeds)

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


    ### FIRST STEP PREDICTION ###
    # just predict the forward direction (the first half of streamlinePositions)
    curStreamlinePos_ras = streamlinePositions[0:noSeeds,0,]
    curStreamlinePos_ijk = (M.dot(curStreamlinePos_ras.T) + abc).T
    lastDirections = np.zeros([noSeeds,3])

    streamlinePositions_ijk[0:noSeeds,0,] = curStreamlinePos_ijk ## FORWARD
    streamlinePositions_ijk[noSeeds:,1,] = curStreamlinePos_ijk ## BACKWARD

    ld_input = lastDirections

    # never rotate the data of the first step
    oldRotationState = myState.rotateData
    myState.rotateData = False
    # predict direction but in never rotate data
    if(myState.magicModel):
        predictedDirection, vecNorms, dwi_at_curPosition, stopTrackingProbability = getNextDirectionMagicModel(myState,data,curPosition_ijk=curStreamlinePos_ijk,model=model,lastDirections=ld_i,x_=x_,y_=y_,z_=z_)
    else:
        predictedDirection, vecNorms, dwi_at_curPosition = getNextDirection(myState, data, curPosition_ijk = curStreamlinePos_ijk, model = model, lastDirections_ras= ld_input, x_ = x_, y_ = y_, z_ = z_)
    myState.rotateData = oldRotationState
    # compute next streamline position
    candidatePosition_ras, candidatePosition_ijk = makeStep(myState, printfProfiling=printfProfiling, predictedDirection = predictedDirection, lastDirections = lastDirections, curStreamlinePos_ras = curStreamlinePos_ras, M = M, abc = abc)

    # update positions
    streamlinePositions[0:noSeeds,1,] = candidatePosition_ras
    streamlinePositions[noSeeds:,0,] = candidatePosition_ras
    streamlinePositions_ijk[0:noSeeds,1,] = candidatePosition_ijk
    streamlinePositions_ijk[noSeeds:,0,] = candidatePosition_ijk
    candidatePosition_ras = streamlinePositions[:,1,]
    candidatePosition_ijk = (M.dot(candidatePosition_ras.T) + abc).T

    if(myState.use2DProjection and myState.usePreviousDirection):
        print('2D projection')
        cp = streamlinePositions[:,0,]
        cp_ijk = (M.dot(cp.T) + abc).T
        dwi_at_curPosition = dwi_tools.interpolateDWIVolume(myState, data, cp_ijk, x_,y_,z_)
        dwi_at_curPosition = dwi_tools.projectIntoAppropriateSpace(myState, dwi_at_curPosition)
        noSamples,dx,dy,dz,dw = dwi_at_curPosition.shape
        dwi_at_curPosition = np.reshape(dwi_at_curPosition, [noSamples,16,16])
        dwi_at_curPosition = dwi_at_curPosition[..., np.newaxis]


    ### START ITERATION UNTIL NOITERATIONS REACHED ###
    # tracking steps are done in RAS
    validSls = None
    start_time = time.time()

    for seedIdx in range(2*noSeeds):
        print('\nStreamline %d / %d ' % (seedIdx, 2*noSeeds))
        rnn_model.reset_states()
        candidatePosition_ras = streamlinePositions[seedIdx,1,]
        candidatePosition_ras = candidatePosition_ras[None, ...]
        candidatePosition_ijk = (M.dot(candidatePosition_ras.T) + abc).T

        for iter in range(1,noIterations):
            ####
            ####
            # estimate tangent at current position
            curStreamlinePos_ras = candidatePosition_ras
            curStreamlinePos_ijk = candidatePosition_ijk
            lastDirections =  (streamlinePositions[seedIdx,iter-1,] - streamlinePositions[seedIdx,iter,]) # previousPosition - currentPosition in RAS
            lastDirections = lastDirections[None, ...]
            lastDirections_ijk = (streamlinePositions_ijk[seedIdx,iter-1,] - streamlinePositions_ijk[seedIdx,iter,])
            lastDirections_ijk = lastDirections_ijk[None, ...]
            ####
            ####
            # compute direction
            ld_input = (lastDirections, lastDirections_ijk)
            if(myState.magicModel):
                predictedDirection, vecNorms, dwi_at_curPosition, stopTrackingProbability = getNextDirectionMagicModel(
                    myState, data, curPosition_ijk=curStreamlinePos_ijk, model=model, lastDirections=ld_input,
                    validIdx=validSls, x_=x_, y_=y_, z_=z_)
                validPoints = np.greater(stopTrackingProbability > myState.pStopTracking)
            else:
                predictedDirection, vecNorms, dwi_at_curPosition = getNextDirection(myState, data, curPosition_ijk = curStreamlinePos_ijk, model = model, lastDirections_ras= ld_input, x_ = x_, y_ = y_, z_ = z_, validIdx = validSls, rnnModel = rnn_model)

            #vNorms[:,iter,] = vecNorms

            ####
            ####
            # compute next streamline position and check if this position is valid wrt. our stopping criteria1
            candidatePosition_ras, candidatePosition_ijk = makeStep(myState, predictedDirection = predictedDirection, lastDirections = lastDirections, curStreamlinePos_ras = curStreamlinePos_ras, M = M, abc = abc, start_time = start_time, printfProfiling=printfProfiling)

            if(not myState.magicModel):
                validPoints = areVoxelsValidStreamlinePoints(candidatePosition_ijk, mask, fa, myState.faThreshold)

            if(validPoints == 0):
                indexLastStreamlinePosition[seedIdx] = np.min((indexLastStreamlinePosition[seedIdx], iter))
                break

            streamlinePositions[seedIdx, iter + 1,] = candidatePosition_ras
            streamlinePositions_ijk[seedIdx, iter + 1,] = candidatePosition_ijk


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


def recurrentStart(myState, seeds, data, rnn_model, affine, mask, fa, printProgress = False, nverseDirection = False, printfProfiling = False, noIterations = 115):
    '''
    fibre tracking using neural networks
    '''

    mask = mask.astype(np.float)

    noSeeds = len(seeds)

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

    ### START ITERATION UNTIL NOITERATIONS REACHED ###
    # tracking steps are done in RAS
    validSls = None
    start_time = time.time()

    for seedIdx in range(noSeeds):
        print('\nStreamline %d / %d ' % (seedIdx, 2*noSeeds))
        rnn_model.reset_states()
        
		candidatePosition_ras = seeds[seedIdx]
		candidatePosition_ijk = (M.dot(candidatePosition_ras.T) + abc).T
		
		# forward pass
        for iter in range(noIterations):
            ####
            ####
            # estimate tangent at current position
            curStreamlinePos_ras = candidatePosition_ras
            curStreamlinePos_ijk = candidatePosition_ijk
            lastDirections = np.zeros([1,3])
            lastDirections_ijk = np.zeros([1,3])
            if(iter > 0):
                lastDirections =  (streamlinePositions[seedIdx,iter-1,] - streamlinePositions[seedIdx,iter,]) # previousPosition - currentPosition in RAS
                lastDirections = lastDirections[None, ...]
                lastDirections_ijk = (streamlinePositions_ijk[seedIdx,iter-1,] - streamlinePositions_ijk[seedIdx,iter,])
                lastDirections_ijk = lastDirections_ijk[None, ...]
            else:
                oldRotationState = myState.rotateData
                myState.rotateData = False
            
            ####
            ####
            # compute direction
            ld_input = (lastDirections, lastDirections_ijk)
             predictedDirection, vecNorms, dwi_at_curPosition, stopTrackingProbability = getNextDirection(myState, data, curPosition_ijk = curStreamlinePos_ijk, model = model, lastDirections_ras= ld_input, x_ = x_, y_ = y_, z_ = z_, validIdx = validSls, rnnModel = rnn_model)

            if(iter == 0):
                myState.rotateData = oldRotationState

            ####
            ####
            # compute next streamline position and check if this position is valid wrt. our stopping criteria1
            candidatePosition_ras, candidatePosition_ijk = makeStep(myState, predictedDirection = predictedDirection, lastDirections = lastDirections, curStreamlinePos_ras = curStreamlinePos_ras, M = M, abc = abc, start_time = start_time, printfProfiling=printfProfiling)

            if(myState.magicModel):
                validPoints = np.greater(stopTrackingProbability > myState.pStopTracking)
            else:
                validPoints = areVoxelsValidStreamlinePoints(candidatePosition_ijk, mask, fa, myState.faThreshold)

            if(validPoints == 0):
                indexLastStreamlinePosition[seedIdx] = np.min((indexLastStreamlinePosition[seedIdx], iter))
                break

            streamlinePositions[seedIdx, iter + 1,] = candidatePosition_ras
            streamlinePositions_ijk[seedIdx, iter + 1,] = candidatePosition_ijk


		## backward pass
		streamlinePositions[seedIdx+noSeeds, 0,] = streamlinePositions[seedIdx, 1,]
		streamlinePositions[seedIdx+noSeeds, 1,] = streamlinePositions[seedIdx, 0,]
		
		candidatePosition_ras = streamlinePositions[seedIdx+noSeeds, 1,]
		candidatePosition_ijk = (M.dot(candidatePosition_ras.T) + abc).T
		
		seedIdx = seedIdx + noSeeds
		
        for iter in range(1, noIterations):
            ####
            ####
            # estimate tangent at current position
            curStreamlinePos_ras = candidatePosition_ras
            curStreamlinePos_ijk = candidatePosition_ijk
            if(iter > 0):
                lastDirections =  (streamlinePositions[seedIdx,iter-1,] - streamlinePositions[seedIdx,iter,]) # previousPosition - currentPosition in RAS
                lastDirections = lastDirections[None, ...]
                lastDirections_ijk = (streamlinePositions_ijk[seedIdx,iter-1,] - streamlinePositions_ijk[seedIdx,iter,])
                lastDirections_ijk = lastDirections_ijk[None, ...]
            else:
                oldRotationState = myState.rotateData
                myState.rotateData = False
            ####
            ####
            # compute direction
            ld_input = (lastDirections, lastDirections_ijk)
            predictedDirection, vecNorms, dwi_at_curPosition, stopTrackingProbability = getNextDirection(myState, data, curPosition_ijk = curStreamlinePos_ijk, model = model, lastDirections_ras= ld_input, x_ = x_, y_ = y_, z_ = z_, validIdx = validSls, rnnModel = rnn_model)

            if(iter == 0):
                myState.rotateData = oldRotationState

            ####
            ####
            # compute next streamline position and check if this position is valid wrt. our stopping criteria1
            candidatePosition_ras, candidatePosition_ijk = makeStep(myState, predictedDirection = predictedDirection, lastDirections = lastDirections, curStreamlinePos_ras = curStreamlinePos_ras, M = M, abc = abc, start_time = start_time, printfProfiling=printfProfiling)

            if(myState.magicModel):
                validPoints = np.greater(stopTrackingProbability > myState.pStopTracking)
            else:
                validPoints = areVoxelsValidStreamlinePoints(candidatePosition_ijk, mask, fa, myState.faThreshold)

            if(validPoints == 0):
                indexLastStreamlinePosition[seedIdx] = np.min((indexLastStreamlinePosition[seedIdx], iter))
                break

            streamlinePositions[seedIdx, iter + 1,] = candidatePosition_ras
            streamlinePositions_ijk[seedIdx, iter + 1,] = candidatePosition_ijk
            
        ###
		seedIdx = seedIdx - noSeeds
		###
		# continue with next streamline

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
