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

def start(seeds, data, model, affine, noX=3, noY=3,noZ=3,dw=288,coordinateScaling = 0.1, stepWidth = 0.1, useSph = False,inverseDirection = False, bitracker = False):   
    '''
    fibre tracking using neural networks
    '''    
    # the stepDirection is currently employed to track fibre's into the predicted as well as its opposite direction
    stepDirection = 1
    if(inverseDirection):
        stepDirection = -1
    
    noSeeds = len(seeds)
    noIterations = 200 #TODO: DONT HARDCODE THE NO OF ITERATIONS, IMPLEMENT STOPPING CRITERIA

    # initialize streamline positions data
    vNorms = np.zeros([noSeeds,noIterations+1])
    streamlinePositions = np.zeros([noSeeds,noIterations+1,3])
    streamlinePositions[:,0,] = seeds[0:noSeeds]
    streamlinePositions[:,1,] = seeds[0:noSeeds]

    # interpolate data given these coordinates for each channel
    x = np.zeros([noSeeds,noX,noY,noZ,dw])
    x_,y_,z_ = dwi_tools._getCoordinateGrid(noX,noY,noZ,coordinateScaling)

    # prepare transformations to project a streamline point from IJK into RAS coordinate system to interpolate DWI data
    aff_ras_ijk = np.linalg.inv(affine) # aff: IJK -> RAS
    M = aff_ras_ijk[:3, :3]
    abc = aff_ras_ijk[:3, 3]
    abc = abc[:,None]
    
    for iter in range(1,noIterations):
        # interpolate dwi data at each point of our streamline
        for j in range(0,noSeeds):
            # project from RAS to image coordinate system
            curStreamlinePos_ras = streamlinePositions[j,iter,]
            curStreamlinePos_ras = curStreamlinePos_ras[:,None]
            curStreamlinePos_ijk = (M.dot(curStreamlinePos_ras) + abc).T
            
            coordVecs = np.vstack(np.meshgrid(x_,y_,z_)).reshape(3,-1).T + curStreamlinePos_ijk
            x[j,] = dwi_tools.interpolatePartialDWIVolume(data,coordVecs, noX = noX, noY = noY, noZ = noZ, coordinateScaling = coordinateScaling,x_ = x_,y_ = y_,z_ = z_)

        lastDirections = (streamlinePositions[:,iter-1,] - streamlinePositions[:,iter,]) # previousPosition - currentPosition
        vecNorms = np.sqrt(np.sum(lastDirections ** 2 , axis = 1)) # make unit vector
        lastDirections = -1 * np.nan_to_num(lastDirections / vecNorms[:,None])
            
        if(bitracker):
            predictedDirection = model.predict([x, lastDirections], batch_size = 2**12)
        else:
            predictedDirection = model.predict([x],  batch_size = 2**12)
        
        # depending on the coordinates change different de-normalization approach
        if(useSph == True):
            #predictedDirection = dwi_tools.convAllFromSphToEuclCoords((2*np.pi)*predictedDirection + np.pi)
            vecNorms = np.sqrt(np.sum(predictedDirection ** 2 , axis = 1)) # should be unit in this case.. 
        else:
            # squash output to unit length
            vecNorms = np.sqrt(np.sum(predictedDirection ** 2 , axis = 1))
            #predictedDirection = np.nan_to_num(predictedDirection / vecNorms[:,None])   
        vNorms[:,iter,] = vecNorms
        
        # update next streamline position
        for j in range(0,noSeeds):
            lv1 = predictedDirection[j,]
            pv1 = lastDirections[j,]
            theta = np.arcsin(np.dot(pv1,lv1) / (np.linalg.norm(pv1)*np.linalg.norm(lv1)))            
            if(theta < 0 and iter>1):
                predictedDirection[j,] = -predictedDirection[j,]
            
            streamlinePositions[j,iter+1,] = streamlinePositions[j,iter,] - stepDirection * stepWidth * predictedDirection[j,]
        
        stepDirection = 1

    return streamlinePositions, vNorms


def joinTwoAlignedStreamlineLists(streamlines_left,streamlines_right):
    assert(len(streamlines_left) == len(streamlines_right), "The two lists of streamlines need to have the same number of elements.")
    
    streamlines_joined = []
    
    for i in range(0,len(streamlines_left)):
        sl_l = np.flipud(streamlines_left[i][2:])
        sl_r = streamlines_right[i][1:]
        streamlines_joined.append(np.concatenate([sl_l, sl_r]))
        
    return streamlines_joined


def startWithStopping(seeds, data, model, affine, mask, fa, printProgress = False, fa_threshold = 0.2, noX=3, noY=3,noZ=3,dw=288,coordinateScaling = 0.1, stepWidth = 0.1, useSph = False,inverseDirection = False, bitracker = False, bayesianModel = False, printfProfiling = False, noIterations = 200):   
    '''
    fibre tracking using neural networks
    '''    
    # the stepDirection is currently employed to track fibre's into the predicted as well as its opposite direction
    mask = mask.astype(np.float)
    stepDirection = 1
    if(inverseDirection):
        stepDirection = -1
    
    noSeeds = len(seeds)

    # initialize streamline positions data
    vNorms = np.zeros([noSeeds,noIterations+1])
    streamlinePositions = np.zeros([noSeeds,noIterations+1,3])
    streamlinePositions[:,0,] = seeds[0:noSeeds]
    streamlinePositions[:,1,] = seeds[0:noSeeds]
    indexLastStreamlinePosition = noIterations * np.ones([noSeeds], dtype=np.intp)
    
    # interpolate data given these coordinates for each channel
    x = np.zeros([noSeeds,noX,noY,noZ,dw])
    x_,y_,z_ = dwi_tools._getCoordinateGrid(noX,noY,noZ,coordinateScaling)

    # prepare transformations to project a streamline point from IJK into RAS coordinate system to interpolate DWI data
    aff_ras_ijk = np.linalg.inv(affine) # aff: IJK -> RAS
    M = aff_ras_ijk[:3, :3]
    abc = aff_ras_ijk[:3, 3]
    abc = abc[:,None]
    
    
    start_time = time.time()    
    for iter in range(1,noIterations):
        #if((iter % 10) == 0):
        if(printProgress):
            print(str(iter-1) + "/" + str(noIterations) + " [" + str(time.time() - start_time) + "s]")
        curStreamlinePos_ras = streamlinePositions[:,iter,].T
        curStreamlinePos_ijk = (M.dot(curStreamlinePos_ras) + abc).T
        x = dwi_tools.interpolateDWIVolume(data, curStreamlinePos_ijk, x_,y_,z_, noX = noX, noY = noY, noZ = noZ)
        if(printfProfiling):
            print(" -> 1 " + str(time.time() - start_time) + "s]")
        lastDirections = (streamlinePositions[:,iter-1,] - streamlinePositions[:,iter,]) # previousPosition - currentPosition
        vecNorms = np.sqrt(np.sum(lastDirections ** 2 , axis = 1)) # make unit vector
        lastDirections = np.nan_to_num(lastDirections / vecNorms[:,None])
        
        if(bayesianModel):
            szX = lastDirections.shape
            noRepetitions = 50
            x = np.repeat(x,noRepetitions,axis=0)
            lastDirections = np.repeat(lastDirections,noRepetitions,axis=0)
        
        if(bitracker):
            predictedDirection = model.predict([x, lastDirections], batch_size = 2**12)
        else:
            predictedDirection = model.predict([x], batch_size = 2**12)
        
        if(len(predictedDirection) == 2):
            predictedDirection = predictedDirection[0]
            
        if(bayesianModel):
            bayesianDirections = np.reshape(predictedDirection, [ int(szX[0]), noRepetitions, szX[1] ] )
            
            for j in range(0,noSeeds):
                
                pv1 = lastDirections[j,]
                norm_pv1 = np.linalg.norm(pv1)
                
                for i in range(0,noRepetitions):
                    lv1 = bayesianDirections[j,i,]
                    theta = np.arcsin(np.dot(pv1,lv1) / (norm_pv1*np.linalg.norm(lv1)))            
                    if(theta < 0 and iter>1):
                        bayesianDirections[j,i,] = -bayesianDirections[j,i,]
            
            predictedDirection = np.mean(bayesianDirections,axis=1)
            
            vecNorms = np.sqrt(np.sum(predictedDirection ** 2 , axis = 1))
            predictedDirection = np.nan_to_num(predictedDirection / vecNorms[:,None])   
            # renormalize direction
            
            predictedDirectionStd = np.sum(np.std(bayesianDirections,axis=1),axis=1) / 3
        
        if(printfProfiling):
            print(" -> 2 " + str(time.time() - start_time) + "s]")
        # depending on the coordinates change different de-normalization approach
        if(useSph == True):
            #predictedDirection = dwi_tools.convAllFromSphToEuclCoords((2*np.pi)*predictedDirection + np.pi)
            vecNorms = np.sqrt(np.sum(predictedDirection ** 2 , axis = 1)) # should be unit in this case.. 
        else:
            # squash output to unit length
            vecNorms = np.sqrt(np.sum(predictedDirection ** 2 , axis = 1))
            #predictedDirection = np.nan_to_num(predictedDirection / vecNorms[:,None])   
        vNorms[:,iter,] = vecNorms
        
        if(bayesianModel):
            vNorms[:,iter,] = predictedDirectionStd
        
        if(printfProfiling):
            print(" -> 3 " + str(time.time() - start_time) + "s]")
        # update next streamline position, SLOWEST PART COMES HERE:
        for j in range(0,noSeeds):
            lv1 = predictedDirection[j,]
            pv1 = lastDirections[j,]
            #print(str(lv1) + '--' + str(pv1))
            #theta = np.arcsin(np.dot(pv1,lv1) / (np.linalg.norm(pv1)*np.linalg.norm(lv1)))   
            
            theta = np.dot(pv1,lv1)
            
            #thetaMinus = np.dot(pv1,-1*lv1)
            
            if(theta < 0 and iter>1):
            #if( (thetaMinus<theta) and (iter>1)):
                predictedDirection[j,] = -predictedDirection[j,]
        # SLOWEST PART ENDS HERE    
        if(printfProfiling):
            print(" -> 4 " + str(time.time() - start_time) + "s]")
        candidatePosition = streamlinePositions[:,iter,] - stepDirection * stepWidth * predictedDirection ### ### ### ### CHANGED - to + 11/19/18 !!! !!! !!! !!!
        candidatePosition_ijk = (M.dot(candidatePosition.T) + abc).T   #projectRAStoIJK(candidatePosition,M,abc)
        validPoints = areVoxelsValidStreamlinePoints(candidatePosition_ijk, mask, fa, fa_threshold)
        print("valid ratio %d / %d (%.2f)" % (sum(validPoints), noSeeds, float(sum(validPoints)) / float(noSeeds)))
        if(printfProfiling):
            print(" -> 5 " + str(time.time() - start_time) + "s]")
        for j in range(0,noSeeds):
            if(validPoints[j]):
                streamlinePositions[j,iter+1,] = candidatePosition[j,]
            else:
                indexLastStreamlinePosition[j] = np.min((indexLastStreamlinePosition[j],iter))
        
        stepDirection = 1
        
        
    streamlinePositions = streamlinePositions.tolist()
    
    vNorms = vNorms.tolist()
    
    for seedIdx in range(0,noSeeds):
        currentStreamline = np.array(streamlinePositions[seedIdx])
        currentStreamline = currentStreamline[0:indexLastStreamlinePosition[seedIdx],]
        
        currentNorm = np.array(vNorms[seedIdx])
        currentNorm = currentNorm[0:indexLastStreamlinePosition[seedIdx],]
        
        streamlinePositions[seedIdx] = currentStreamline
        vNorms[seedIdx] = currentNorm
        

    return streamlinePositions, vNorms


def makeStep(dwi,curPosition_ijk, model, lastDirections = None, x_ = [0], y_ = [0], z_ = [0], noX = 1, noY = 1, noZ = 1, batch_size = 2**8):
    dwi_at_curPosition = dwi_tools.interpolateDWIVolume(dwi, curPosition_ijk, x_,y_,z_, noX = noX, noY = noY, noZ = noZ)
    if(lastDirections is None):
        predictedDirection = model.predict([dwi_at_curPosition], batch_size = batch_size)
    else:
        predictedDirection = model.predict([dwi_at_curPosition, lastDirections], batch_size = batch_size)
        
        
    if(len(predictedDirection) == 2):
        predictedDirection = predictedDirection[0]
        
    return predictedDirection
            

def startWithStopping__(seeds, data, model, affine, mask, fa, printProgress = False, fa_threshold = 0.2, noX=3, noY=3,noZ=3,dw=288,coordinateScaling = 0.1, stepWidth = 0.1, useSph = False,inverseDirection = False, printfProfiling = False, noIterations = 200):   
    '''
    fibre tracking using neural networks
    '''    
    # the stepDirection is currently employed to track fibre's into the predicted as well as its opposite direction
    mask = mask.astype(np.float)
    stepDirection = 1
    
    noSeeds = len(seeds)

    # initialize streamline positions data
    vNorms = np.zeros([2*noSeeds,noIterations+1])
    streamlinePositions = np.zeros([2*noSeeds,noIterations+1,3])
    
    streamlinePositions[0:noSeeds,0,] = seeds[0:noSeeds] ## FORWARD
    streamlinePositions[noSeeds:,1,] = seeds[0:noSeeds] ## BACKWARD
    
    indexLastStreamlinePosition = noIterations * np.ones([2*noSeeds], dtype=np.intp)

    del seeds # its not supposed to use the seeds anymore
    
    # interpolate data given these coordinates for each channel
    x = np.zeros([2*noSeeds,noX,noY,noZ,dw])
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
    x = dwi_tools.interpolateDWIVolume(data, curStreamlinePos_ijk, x_,y_,z_, noX = noX, noY = noY, noZ = noZ)
    predictedDirection = makeStep(data, curPosition_ijk = curStreamlinePos_ijk, lastDirections = lastDirections, model = model)
    
    for j in range(0,noSeeds):
        lv1 = predictedDirection[j,]
        pv1 = lastDirections[j,]
        #print(str(lv1) + '--' + str(pv1))
        theta = np.arcsin(np.dot(pv1,lv1) / (np.linalg.norm(pv1)*np.linalg.norm(lv1)))   

        theta = np.dot(pv1,lv1)

        if(theta < 0 and iter>1):
            predictedDirection[j,] = -predictedDirection[j,]
            
    streamlinePositions[0:noSeeds,1,] = curStreamlinePos_ras - stepDirection * stepWidth * predictedDirection
    streamlinePositions[noSeeds:,0,] = streamlinePositions[0:noSeeds,1,]
    
    ### START ITERATION UNTIL NOITERATIONS REACHED ###
    start_time = time.time()    
    for iter in range(1,noIterations):
        # prepare data
        if(printProgress):
            print(str(iter-1) + "/" + str(noIterations) + " [" + str(time.time() - start_time) + "s]")
        curStreamlinePos_ras = streamlinePositions[:,iter,].T
        curStreamlinePos_ijk = (M.dot(curStreamlinePos_ras) + abc).T
        x = dwi_tools.interpolateDWIVolume(data, curStreamlinePos_ijk, x_,y_,z_, noX = noX, noY = noY, noZ = noZ)
        if(printfProfiling):
            print(" -> 1 " + str(time.time() - start_time) + "s]")
        lastDirections = (streamlinePositions[:,iter-1,] - streamlinePositions[:,iter,]) # previousPosition - currentPosition
        vecNorms = np.sqrt(np.sum(lastDirections ** 2 , axis = 1)) # make unit vector
        lastDirections = np.nan_to_num(lastDirections / vecNorms[:,None])
        
        
        # make step
        predictedDirection = makeStep(data, curPosition_ijk = curStreamlinePos_ijk, lastDirections = lastDirections, model = model)
        
        
        # compute some information about our step
        if(printfProfiling):
            print(" -> 2 " + str(time.time() - start_time) + "s]")
        # depending on the coordinates change different de-normalization approach
        if(useSph == True):
            #predictedDirection = dwi_tools.convAllFromSphToEuclCoords((2*np.pi)*predictedDirection + np.pi)
            vecNorms = np.sqrt(np.sum(predictedDirection ** 2 , axis = 1)) # should be unit in this case.. 
        else:
            # squash output to unit length
            vecNorms = np.sqrt(np.sum(predictedDirection ** 2 , axis = 1))
            #predictedDirection = np.nan_to_num(predictedDirection / vecNorms[:,None])   
        vNorms[:,iter,] = vecNorms
        

        
        # check predicted direction and flip it if necessary
        if(printfProfiling):
            print(" -> 3 " + str(time.time() - start_time) + "s]")
        # update next streamline position, SLOWEST PART COMES HERE:
        for j in range(0,2*noSeeds):
            lv1 = predictedDirection[j,]
            pv1 = lastDirections[j,]
            #print(str(lv1) + '--' + str(pv1))
            theta = np.arcsin(np.dot(pv1,lv1) / (np.linalg.norm(pv1)*np.linalg.norm(lv1)))   
            
            theta = np.dot(pv1,lv1)
            
            if(theta < 0 and iter>1):
                predictedDirection[j,] = -predictedDirection[j,]
        # SLOWEST PART ENDS HERE    
        if(printfProfiling):
            print(" -> 4 " + str(time.time() - start_time) + "s]")
        
        
        candidatePosition_ras = curStreamlinePos_ras.T - stepDirection * stepWidth * predictedDirection ### ### ### ### CHANGED - to + 11/19/18 !!! !!! !!! !!!
        candidatePosition_ijk = (M.dot(candidatePosition_ras.T) + abc).T   #projectRAStoIJK(candidatePosition,M,abc)
        validPoints = areVoxelsValidStreamlinePoints(candidatePosition_ijk, mask, fa, fa_threshold)
        if(printProgress):
            print("valid ratio %d / %d (%.2f)" % (sum(validPoints), 2*noSeeds, float(sum(validPoints)) / float(2*noSeeds)))
        if(printfProfiling):
            print(" -> 5 " + str(time.time() - start_time) + "s]")
        for j in range(0,2*noSeeds):
            if(validPoints[j]):
                streamlinePositions[j,iter+1,] = candidatePosition_ras[j,]
            else:
                #streamlinePositions[j,iter+1,] = candidatePosition_ras[j,]
                indexLastStreamlinePosition[j] = np.min((indexLastStreamlinePosition[j],iter))
        
        stepDirection = 1
        
        
    streamlinePositions = streamlinePositions.tolist()
    
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
    sl_backward = streamlinePositions[noSeeds+1:]

    # join directions
    streamlines = Streamlines(joinTwoAlignedStreamlineLists(sl_backward, sl_forward))
    
    return streamlines, vNorms



def startWithStoppingAnd2DProjection(seeds, data, model, affine, mask, fa, resamplingSphere, printProgress = False, fa_threshold = 0.2, stepWidth = 0.6, useSph = False,inverseDirection = False, bitracker = False, bayesianModel = False, printfProfiling = False, noIterations = 200):   
    '''
    fibre tracking using neural networks
    '''    
    # the stepDirection is currently employed to track fibre's into the predicted as well as its opposite direction
    mask = mask.astype(np.float)
    stepDirection = 1
    if(inverseDirection):
        stepDirection = -1
    
    noSeeds = len(seeds)
    dw = len(resamplingSphere.phi)

    # initialize streamline positions data
    vNorms = np.zeros([noSeeds,noIterations+1])
    streamlinePositions = np.zeros([noSeeds,noIterations+1,3])
    streamlinePositions[:,0,] = seeds[0:noSeeds]
    streamlinePositions[:,1,] = seeds[0:noSeeds]
    indexLastStreamlinePosition = noIterations * np.ones([noSeeds], dtype=np.intp)
    
    # interpolate data given these coordinates for each channel
    x = np.zeros([noSeeds,dw])
    
    x1 = np.zeros([noSeeds,24,24,1])
    
    x_,y_,z_ = dwi_tools._getCoordinateGrid(1,1,1,1)

    # prepare transformations to project a streamline point from IJK into RAS coordinate system to interpolate DWI data
    aff_ras_ijk = np.linalg.inv(affine) # aff: IJK -> RAS
    M = aff_ras_ijk[:3, :3]
    abc = aff_ras_ijk[:3, 3]
    abc = abc[:,None]
    
    
    start_time = time.time()    
    for iter in range(1,noIterations):
        if(printProgress):
            print(str(iter-1) + "/" + str(noIterations) + " [" + str(time.time() - start_time) + "s]")
        curStreamlinePos_ras = streamlinePositions[:,iter,].T
        curStreamlinePos_ijk = (M.dot(curStreamlinePos_ras) + abc).T
        x = dwi_tools.interpolateDWIVolume(data, curStreamlinePos_ijk, x_,y_,z_, noX = 1, noY = 1, noZ = 1)
        x1 = np.reshape(x, [noSeeds,10,10])
        x1 = x1[..., np.newaxis]
       
        if(printfProfiling):
            print(" -> 1 " + str(time.time() - start_time) + "s]")
        lastDirections = (streamlinePositions[:,iter-1,] - streamlinePositions[:,iter,]) # previousPosition - currentPosition
        vecNorms = np.sqrt(np.sum(lastDirections ** 2 , axis = 1)) # make unit vector
        lastDirections = np.nan_to_num(lastDirections / vecNorms[:,None])
        
        if(bayesianModel):
            szX = lastDirections.shape
            noRepetitions = 50
            x = np.repeat(x,noRepetitions,axis=0)
            lastDirections = np.repeat(lastDirections,noRepetitions,axis=0)
        
        if(bitracker):
            predictedDirection = model.predict([x1, lastDirections], batch_size = 2**12)
        else:
            predictedDirection = model.predict([x1], batch_size = 2**12)
            
        if(len(predictedDirection) == 2):
            predictedDirection = predictedDirection[0]
            
        if(bayesianModel):
            bayesianDirections = np.reshape(predictedDirection, [ int(szX[0]), noRepetitions, szX[1] ] )
            
            for j in range(0,noSeeds):
                
                pv1 = lastDirections[j,]
                norm_pv1 = np.linalg.norm(pv1)
                
                for i in range(0,noRepetitions):
                    lv1 = bayesianDirections[j,i,]
                    theta = np.arcsin(np.dot(pv1,lv1) / (norm_pv1*np.linalg.norm(lv1)))            
                    if(theta < 0 and iter>1):
                        bayesianDirections[j,i,] = -bayesianDirections[j,i,]
            
            predictedDirection = np.mean(bayesianDirections,axis=1)
            
            vecNorms = np.sqrt(np.sum(predictedDirection ** 2 , axis = 1))
            predictedDirection = np.nan_to_num(predictedDirection / vecNorms[:,None])   
            # renormalize direction
            
            predictedDirectionStd = np.sum(np.std(bayesianDirections,axis=1),axis=1) / 3
        
        if(printfProfiling):
            print(" -> 2 " + str(time.time() - start_time) + "s]")
        # depending on the coordinates change different de-normalization approach
        if(useSph == True):
            #predictedDirection = dwi_tools.convAllFromSphToEuclCoords((2*np.pi)*predictedDirection + np.pi)
            vecNorms = np.sqrt(np.sum(predictedDirection ** 2 , axis = 1)) # should be unit in this case.. 
        else:
            # squash output to unit length
            vecNorms = np.sqrt(np.sum(predictedDirection ** 2 , axis = 1))
            #predictedDirection = np.nan_to_num(predictedDirection / vecNorms[:,None])   
        vNorms[:,iter,] = vecNorms
        
        if(bayesianModel):
            vNorms[:,iter,] = predictedDirectionStd
        
        if(printfProfiling):
            print(" -> 3 " + str(time.time() - start_time) + "s]")
        # update next streamline position, SLOWEST PART COMES HERE:
        for j in range(0,noSeeds):
            lv1 = predictedDirection[j,]
            pv1 = lastDirections[j,]
            theta = np.arcsin(np.dot(pv1,lv1) / (np.linalg.norm(pv1)*np.linalg.norm(lv1)))   
            
            theta = np.dot(pv1,lv1)
            
            if(theta < 0 and iter>1):
                predictedDirection[j,] = -predictedDirection[j,]
        # SLOWEST PART ENDS HERE    
        if(printfProfiling):
            print(" -> 4 " + str(time.time() - start_time) + "s]")
        candidatePosition = streamlinePositions[:,iter,] - stepDirection * stepWidth * predictedDirection
        candidatePosition_ijk = (M.dot(candidatePosition.T) + abc).T   #projectRAStoIJK(candidatePosition,M,abc)
        validPoints = areVoxelsValidStreamlinePoints(candidatePosition_ijk, mask, fa, fa_threshold)
        print("valid ratio %d / %d (%.2f)" % (sum(validPoints), noSeeds, float(sum(validPoints)) / float(noSeeds)))
        if(printfProfiling):
            print(" -> 5 " + str(time.time() - start_time) + "s]")
        for j in range(0,noSeeds):
            if(validPoints[j]):
                streamlinePositions[j,iter+1,] = candidatePosition[j,]
            else:
                indexLastStreamlinePosition[j] = np.min((indexLastStreamlinePosition[j],iter))
        
        stepDirection = 1
        
        
    streamlinePositions = streamlinePositions.tolist()
    
    vNorms = vNorms.tolist()
    
    for seedIdx in range(0,noSeeds):
        currentStreamline = np.array(streamlinePositions[seedIdx])
        currentStreamline = currentStreamline[0:indexLastStreamlinePosition[seedIdx],]
        
        currentNorm = np.array(vNorms[seedIdx])
        currentNorm = currentNorm[0:indexLastStreamlinePosition[seedIdx],]
        
        streamlinePositions[seedIdx] = currentStreamline
        vNorms[seedIdx] = currentNorm
        

    return streamlinePositions, vNorms


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
