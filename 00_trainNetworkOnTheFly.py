from numpy.random import seed
seed(2342)
from tensorflow import set_random_seed
set_random_seed(4223)

import time
import nrrd
import os
import nibabel as nb
import numpy as np
import matplotlib.pyplot as plt
import h5py
import tensorflow as tf
import warnings

from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
from dipy.tracking.local import LocalTracking, ThresholdTissueClassifier
from dipy.tracking.utils import random_seeds_from_mask
from dipy.reconst.dti import TensorModel
from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel,
                                   auto_response)
from dipy.reconst.shm import CsaOdfModel
from dipy.data import default_sphere
from dipy.direction import peaks_from_model
from dipy.data import fetch_stanford_hardi, read_stanford_hardi, get_sphere
from dipy.segment.mask import median_otsu
from dipy.viz import actor, window
from dipy.io.image import save_nifti
from dipy.core import gradients
from dipy.tracking.streamline import Streamlines
from dipy.reconst.dti import fractional_anisotropy

from dipy.tracking import utils

import src.dwi_tools as dwi_tools
import src.nn_helper as nn_helper
from src.tractographydatagenerator import TractographyDataGenerator, TwoDimensionalTractographyDataGenerator, ThreeDimensionalTractographyDataGenerator
from src.nn_helper import swish, squared_cosine_proximity_2, weighted_binary_crossentropy, mse_directionInvariant
from src.SelectiveDropout import SelectiveDropout
from src.tied_layers1d import Convolution2D_tied

from keras.layers.advanced_activations import LeakyReLU, ReLU
from keras.layers import Activation
from keras.callbacks import TensorBoard, ReduceLROnPlateau
from keras.models import load_model

from dipy.core.gradients import gradient_table, gradient_table_from_bvals_bvecs
from keras.callbacks import LambdaCallback
import argparse
from src.state import TractographyInformation, TrainingInformation

def main():
    '''
    train the deep tractography network using previously generated training data
    '''
    # training parameters
    
    parser = argparse.ArgumentParser(description='Deep Learning Tractography -- Learning @ ISMRM w/ data generator')
    parser.add_argument('streamlines', help='path to streamlines')
    parser.add_argument('-nx', dest='nx', default=1, type=int, help='no of voxels in X plane for each streamline position')
    parser.add_argument('-ny', dest='ny',default=1, type=int, help='no of voxels in Y plane for each streamline position')
    parser.add_argument('-nz', dest='nz',default=1, type=int, help='no of voxels in Z plane for each streamline position')   
    parser.add_argument('-f', '--features', default=128, type=int, help='name of tracking case')
    parser.add_argument('-d', '--depth', default=3, type=int, help='name of tracking case')
    parser.add_argument('-dr', '--dilationRate', default=1, type=int, help='dilation rate in x and y plane for 3D models. ')
    parser.add_argument('-a', '--activationfunction', default='relu', help='relu, leakyrelu, swish')
    parser.add_argument('-m', '--modeltouse', default='mlp_single', help='mlp_single, mlp_doublein_single, cnn_special, cnn_special_pd, rcnn')
    parser.add_argument('-l', '--loss', default='sqCos2', help='cos, mse, sqCos2,sqCos2WEP')
    parser.add_argument('-b', '--batchsize', default=2**10, type=int, help='no. tracking steps')
    parser.add_argument('-e','--epochs', default=1000, type=int, help='no. epochs')
    parser.add_argument('-lr','--learningrate', type=float, default=1e-4, help='minimal length of a streamline [mm]')
    parser.add_argument('-sh', '--shOrder', type=int, default=4, dest='sh', help='order of spherical harmonics (if used)')
    parser.add_argument('-lm','--loadModel', dest='lm',default='', help='continue training with a pretrained model ')
    parser.add_argument('--repr', dest='repr',default='res100', help='data representation: [raw,sph,res100,2D]: raw, spherical harmonics, resampled to 100 directions, 16x16 2D resampling (256 directions) ')
    parser.add_argument('--unitTangent', help='unit tangent', dest='unittangent' , action='store_true')
    parser.add_argument('--dropout', help='dropout regularization', dest='dropout', action='store_true')
    parser.add_argument('--keepZeroVectors', help='keep zero vectors at the outer positions of streamline to indicate termination.', dest='keepzero' , action='store_true')
    parser.add_argument('-bn','--batchnormalization', help='batchnormalization', dest='dropout' , action='store_true')
    parser.add_argument('--bvalue',type=int, default=1000, help='b-value of our DWI data')
    parser.add_argument('--storeTemporaryData',help='store temporary data on disk to save computational time', action='store_true')
    parser.add_argument('-nt', '--noThreads', dest='noThreads', type=int, default=2, help='number of parallel threads of the data generator. Note: this also increases the memory demand.')
    parser.add_argument('--addRandomDataForEndpointPrediction', default='',
                        help='should we generate additional random data for endpoint prediction?', action='store_true')
    parser.add_argument('-sw', dest='sw', default=1.0, type=float, help='tracking step width [mm]')
    parser.add_argument('--rotateData', help='rotate data wrt. tangent [default]', dest='rotateData',
                        action='store_true')

    parser.set_defaults(rotateData=False)
    parser.set_defaults(unittangent=False)   
    parser.set_defaults(dropout=False)   
    parser.set_defaults(keepzero=False)
    parser.set_defaults(batchnormalization=False)   
    parser.set_defaults(storeTemporaryData=False)
    parser.set_defaults(addRandomDataForEndpointPrediction=False)
    args = parser.parse_args()

    activation_function = {
        'relu': lambda x: ReLU(),
        'leakyrelu': lambda x: LeakyReLU(),
        'swish': lambda x: Activation(swish)
    }[args.activationfunction](0)

    dilationRate = (args.dilationRate, args.dilationRate, 1)

    myState = TractographyInformation()
    myState.isISMRM = True
    myState.hcpID = ''
    myState.addRandomDataForEndpointPrediction = args.addRandomDataForEndpointPrediction
    myState.tensorModel = 'precomp'
    myState.dim = [args.nx, args.ny, args.nz]
    myState.repr = args.repr
    myState.b_value = args.bvalue
    myState.stepWidth = args.sw
    myState.shOrder = args.sh
    myState.faThreshold = 0
    myState.unitTangent = args.unittangent
    myState.rotateData = args.rotateData
    myState.pPrecomputedStreamlines = args.streamlines

    myTrainingState = TrainingInformation(pTrainData='', loss=args.loss, noFeatures=args.features,
                                          depth=args.depth, epochs=args.epochs,
                                          learningRate=args.learningrate, useDropout=args.dropout,
                                          useBatchNormalization=args.batchnormalization,
                                          model=args.modeltouse, keepZeroVectors=args.keepzero,
                                          activationFunction=activation_function, batch_size=args.batchsize,
                                          pPretrainedModel=args.lm, dilationRate=dilationRate
                                          )


    usePretrainedModel = False
    storeTemporaryData = args.storeTemporaryData
    endpointPrediction = False
    if(myTrainingState.loss == 'sqCos2WEP'):
        endpointPrediction = True

    
    pModelOutput = myState.pPrecomputedStreamlines.replace('.vtk','').replace('data/','')
    noOutputNeurons = 3 # euclidean coordinates
    
    # load streamlines
    streamlines = dwi_tools.loadVTKstreamlines(myState.pPrecomputedStreamlines)
    #streamlines = streamlines[0:30000]
    
    # load DWI dataset
    nameDWIDataset = 'ISMRM_2015_Tracto_challenge_data'
    useDenoising = False
    bvals,bvecs,gtab,dwi,aff,t1 = dwi_tools.loadISMRMData('data/%s' % (nameDWIDataset), denoiseData = useDenoising, resliceToHCPDimensions=False)
    b0_mask, binarymask = median_otsu(dwi[:,:,:,0], 2, 1)
    nameDWIDataset = 'ISMRM_2015_Tracto_challenge_data_denoised_preproc'
    
    dwi_subset, gtab_subset, bvals_subset, bvecs_subset = dwi_tools.cropDatsetToBValue(myState.b_value, bvals, bvecs, dwi)
    b0_idx = bvals < 10
    b0 = dwi[..., b0_idx].mean(axis=3)
    dwi_subset = dwi_tools.normalize_dwi(dwi_subset, b0)
    myState.b0 = b0
    myState.bvals = bvals_subset
    myState.bvecs = bvecs_subset

    dwi_singleShell = np.concatenate((dwi_subset, dwi[..., b0_idx]), axis=3)
    #    dwi_singleShell_norm = dwi_tools.normalize_dwi(dwi_singleShell, b0)
    bvals_singleShell = np.concatenate((bvals_subset, bvals[..., b0_idx]), axis=0)
    bvecs_singleShell = np.concatenate((bvecs_subset, bvecs[b0_idx,]), axis=0)
    gtab_singleShell = gradient_table(bvals=bvals_singleShell, bvecs=bvecs_singleShell, b0_threshold = 10)

    noD=-1

    ####################
    ####################
    if(myState.repr == '2D'):
        if(endpointPrediction):
            print('Endpoint prediction not implemented yet')
            return
        modelToUse = '2D_cnn'
        myState.repr = '2D'
        model = nn_helper.get_2Dcnn_fcn_singleOutput(myTrainingState, myState, inputShapeDWI=[16,16,args.nz])
        model.summary()
        # data generator
        training_generator = TwoDimensionalTractographyDataGenerator(dwi_subset,streamlines,aff,np.array(list(range(len(streamlines)-5000))), myState=myState, myTrainingState=myTrainingState, storeTemporaryData = storeTemporaryData)
        validation_generator = TwoDimensionalTractographyDataGenerator(dwi_subset,streamlines,aff,np.array(list(range(len(streamlines)-5000+1,len(streamlines)))), myState=myState, myTrainingState=myTrainingState, storeTemporaryData = storeTemporaryData)

        noTrainingSamples = len( list(range(len(streamlines)-1000)) )
        noValidationSamples = len( list(range(len(streamlines)-1000,len(streamlines))) )
        print('samples tra/val %d/%d' % (noTrainingSamples, noValidationSamples) )
    ####################
    ####################
    elif(myState.repr == '3D'):
        if(endpointPrediction):
            print('Endpoint prediction not implemented yet')
            return
        modelToUse = '3D_cnn_dr%d' % (dilationRate[0])
        #model = nn_helper.get_simple3DCNN(loss=loss, lr=lr, useDropout = useDropout, useBN = useBatchNormalization, inputShapeDWI=[noX*8,noY*8,noZ,1], outputShape = noOutputNeurons, activation_function = activation_function, features = noFeatures, depth = depth, noGPUs=noGPUs, dilationRate = dilationRate)
        model = nn_helper.get_3DCNN(myTrainingState, inputShapeDWI=[myState[0]*8,myState[1]*8,myState[2],1])
        model.summary()
        # data generator
        training_generator = ThreeDimensionalTractographyDataGenerator(dwi_subset,streamlines,aff,np.array(list(range(len(streamlines)-10000))), dim=myState.dim, batch_size=myTrainingState.batch_size, storeTemporaryData = storeTemporaryData)
        validation_generator = ThreeDimensionalTractographyDataGenerator(dwi_subset,streamlines,aff,np.array(list(range(len(streamlines)-10000,len(streamlines)))), dim=myState.dim , batch_size=myTrainingState.batch_size, storeTemporaryData = storeTemporaryData)
        noTrainingSamples = len( list(range(len(streamlines)-1000)) )
        noValidationSamples = len( list(range(len(streamlines)-1000,len(streamlines))) )
        print('samples tra/val %d/%d' % (noTrainingSamples, noValidationSamples) )
    ####################
    ####################
    else:
        ### ### ###
    ### MLP SINGLE ###
        ### ### ###
        if(endpointPrediction):
            print('wep model')
            model = nn_helper.get_mlp_singleOutputWEP(myTrainingState, inputShapeDWI=train_DWI.shape[1:])
        else:
            print('raw model')
            mlp_simple = nn_helper.get_mlp_singleOutput(myTrainingState, inputShapeDWI=train_DWI.shape[1:])

        model.summary()

        training_generator = TractographyDataGenerator(dwi_subset,streamlines,aff,np.array(list(range(len(streamlines)-1000))), dim=myState.dim, batch_size=myTrainingState.batch_size, storeTemporaryData = storeTemporaryData, endpointPrediction = endpointPrediction)
        validation_generator = TractographyDataGenerator(dwi_subset,streamlines,aff,np.array(list(range(len(streamlines)-1000,len(streamlines)))), dim=myState.dim, batch_size=myTrainingState.batch_size, storeTemporaryData = storeTemporaryData, endpointPrediction = endpointPrediction)

        noTrainingSamples = len( list(range(len(streamlines)-1000)) )
        noValidationSamples = len( list(range(len(streamlines)-1000,len(streamlines))) )
        print('samples tra/val %d/%d' % (noTrainingSamples, noValidationSamples) )
    
    if(myTrainingState.pPretrainedModel != ''):
        model = load_model(myTrainingState.pPretrainedModel, custom_objects={'tf':tf, 'swish':Activation(swish), 'squared_cosine_proximity_2': squared_cosine_proximity_2, 'Convolution2D_tied': Convolution2D_tied, 'weighted_binary_crossentropy': weighted_binary_crossentropy, 'mse_directionInvariant': mse_directionInvariant})
    
    print('\n**************')
    print('** Training **')
    print('**************\n')
    print('model ' + str(myTrainingState.modelToUse) + ' loss ' + myTrainingState.loss)
    print('dx ' + str(myState.dim[0]) + ' dy ' + str(myState.dim[1]) + ' dz  ' + str(myState.dim[2]))
    print('features ' + str(myTrainingState.noFeatures) + ' depth ' + str(myTrainingState.depth) + ' lr ' + str(myTrainingState.lr) + '\ndropout ' +
          str(myTrainingState.useDropout) + ' bn  ' + str(myTrainingState.useBatchNormalization) + ' batch size ' + str(myTrainingState.batch_size))
    print('**************\n')
   
    # train simple MLP
    params = "%s_%s_dx_%d_dy_%d_dz_%d_dd_%d_%s_feat_%d_depth_%d_output_%d_lr_%.4f_dropout_%d_bn_%d_unitTangent_%d_wz_%d" % \
             (myTrainingState.modelToUse,myTrainingState.loss,myState.dim[0],myState.dim[1],myState.dim[2],noD,myTrainingState.activationFunction.__class__.__name__,myTrainingState.noFeatures,
              myTrainingState.depth,3,myTrainingState.lr,myTrainingState.useDropout,myTrainingState.useBatchNormalization,myState.unitTangent,myTrainingState.keepZeroVectors)

    pModel = "results/" + pModelOutput + '/models/' + params + ".h5"  # "-{val_loss:.6f}.h5"
    pCSVLog = "results/" + pModelOutput + '/logs/' + params + ".csv"
   
    newpath = r'results/' + pModelOutput + '/models/'
    if not os.path.exists(newpath):
        os.makedirs(newpath)
        
    newpath = r'results/' + pModelOutput + '/logs/'
    if not os.path.exists(newpath):
        os.makedirs(newpath)



    # Train model on dataset
    checkpoint = ModelCheckpoint(pModel, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    csv_logger = CSVLogger(pCSVLog)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=1e-5, verbose=1)
    
    callbacks=[checkpoint, csv_logger]
    
    model.fit_generator(generator=training_generator,
        validation_data=validation_generator,
        use_multiprocessing=True,
        workers=args.noThreads,
        steps_per_epoch=int(noTrainingSamples/myTrainingState.batch_size),
        epochs=myTrainingState.epochs,
        verbose=1,
        validation_steps=int(noValidationSamples/myTrainingState.batch_size),
        callbacks=callbacks,
        max_queue_size=args.noThreads)
            

        
if __name__ == "__main__":
    main()
