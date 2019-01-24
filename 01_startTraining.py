from numpy.random import seed
seed(2342)
from tensorflow import set_random_seed
set_random_seed(4223)
import argparse
import os
import numpy as np
import h5py
import warnings
import src.dwi_tools as dwi_tools
import src.nn_helper as nn_helper
from src.nn_helper import swish
from keras.layers.advanced_activations import LeakyReLU, ReLU
from keras.layers import Activation
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
from keras.models import load_model
from src.state import TractographyInformation, TrainingInformation


def main():
    '''
    train the deep tractography network using previously generated training data
    '''
    # training parameters
    
    parser = argparse.ArgumentParser(description='Deep Learning Tractography -- Learning')
    parser.add_argument('data', help='path to training data')
    parser.add_argument('-f', '--features', default=128, type=int, help='name of tracking case')
    parser.add_argument('-d', '--depth', default=3, type=int, help='name of tracking case')
    parser.add_argument('-a', '--activationfunction', default='relu', help='relu, leakyrelu, swish')
    parser.add_argument('-m', '--modeltouse', default='mlp_single', help='mlp_single, mlp_doublein_single, 2Dcnn, 3Dcnn, cnn_special, rcnn')
    parser.add_argument('-l', '--loss', default='sqCos2WEP', help='cos, mse, sqCos2, sqCos2WEP')
    parser.add_argument('-b', '--batchsize', default=2**12, type=int, help='no. tracking steps')
    parser.add_argument('-e','--epochs', default=1000, type=int, help='no. epochs')
    parser.add_argument('-lr','--learningrate', type=float, default=1e-4, help='minimal length of a streamline [mm]')
    parser.add_argument('-sh', '--shOrder', type=int, default=8, help='order of spherical harmonics (if used)')
    parser.add_argument('--unitTangent', help='unit tangent', dest='unittangent' , action='store_true')
    parser.add_argument('--dropout', help='dropout regularization', dest='dropout' , action='store_true')
    parser.add_argument('--keepZeroVectors', help='keep zero vectors at the outer positions of streamline to indicate termination.', dest='keepzero' , action='store_true')
    parser.add_argument('-bn','--batchnormalization', help='batchnormalization', dest='dropout' , action='store_true')
    parser.add_argument('--bvalue',type=int, default=1000, help='b-value of our DWI data')
        
    parser.set_defaults(unittangent=False)   
    parser.set_defaults(dropout=False)   
    parser.set_defaults(keepzero=False)
    parser.set_defaults(batchnormalization=False)   
    args = parser.parse_args()

    myState = TractographyInformation(unitTangent=args.unittangent)



    activation_function = {
          'relu': lambda x: ReLU(),
          'leakyrelu': lambda x: LeakyReLU(),
          'swish': lambda x: Activation(swish)
        }[args.activationfunction](0)

    myTrainingState = TrainingInformation(pTrainData = args.data, loss = args.loss, noFeatures = args.features, depth = args.depth, epochs = args.epochs,
                                          learningRate = args.learningrate, useDropout = args.dropout, useBatchNormalization = args.batchnormalization,
                                          model = args.modeltouse, keepZeroVectors = args.keepzero, activationFunction = activation_function, batch_size = args.batchsize)


    if(myTrainingState.loss == 'sqCos2WEP'):
        print('Keeping zero vectors as the neural network shall predict endpoints.')
        myTrainingState.keepZeroVectors = True
    
    useSphericalCoordinates = False
    pModelOutput = myTrainingState.pTrainData.replace('.h5','').replace('data/','')


    # load training data
    f = h5py.File(myTrainingState.pTrainData, "r")
    train_DWI = np.array(f["train_DWI"].value)
    train_prevDirection = np.array(f["train_curPosition"].value)
    train_nextDirection = np.array(f["train_NextFibreDirection"].value)
    f.close()
    
    indices = np.arange(len(train_DWI))
    np.random.shuffle(indices)
    train_DWI = train_DWI[indices,]
    train_nextDirection = train_nextDirection[indices,]
    train_prevDirection = train_prevDirection[indices,]
       
    vN = np.sqrt(np.sum(train_nextDirection ** 2 , axis = 1))
    idx1 = np.where(vN > 0)[0]
    vN = np.sqrt(np.sum(train_prevDirection ** 2 , axis = 1))
    idx2 = np.where(vN > 0)[0]
    s2 = set(idx2)
    idxNoZeroVectors = [val for val in idx1 if val in s2]
    
    if(myTrainingState.keepZeroVectors == False):
        train_DWI = train_DWI[idxNoZeroVectors,...]
        train_nextDirection = train_nextDirection[idxNoZeroVectors,]
        train_prevDirection = train_prevDirection[idxNoZeroVectors,]
    
    noSamples,noX,noY,noZ,noD = train_DWI.shape

    myState.dim = [noX,noY,noZ,noD]
    
    print('\n**************')
    print('** Training **')
    print('**************\n')
    print('model ' + str(myTrainingState.modelToUse) + ' loss ' + myTrainingState.loss)
    print('dx ' + str(noX) + ' dy ' + str(noY) + ' dz  ' + str(noZ) + ' dd ' + str(noD))
    print('features ' + str(myTrainingState.noFeatures) + ' depth ' + str(myTrainingState.depth) + ' lr ' + str(myTrainingState.lr) + '\ndropout ' +
          str(myTrainingState.useDropout) + ' bn  ' + str(myTrainingState.useBatchNormalization) + ' batch size ' + str(myTrainingState.batch_size))
    print('dataset ' + str(myTrainingState.pTrainData) + " " + str(noSamples))
    print('**************\n')

    # train simple MLP
    params = "%s_%s_dx_%d_dy_%d_dz_%d_dd_%d_%s_feat_%d_depth_%d_output_%d_lr_%.4f_dropout_%d_bn_%d_unitTangent_%d_wz_%d" % \
             (myTrainingState.modelToUse,myTrainingState.loss,noX,noY,noZ,noD,myTrainingState.activationFunction.__class__.__name__,myTrainingState.noFeatures,
              myTrainingState.depth,3,myTrainingState.lr,myTrainingState.useDropout,myTrainingState.useBatchNormalization,myState.unitTangent,myTrainingState.keepZeroVectors)
    pModel = "results/" + pModelOutput + '/models/' + params + "-{val_loss:.6f}.h5"
    pCSVLog = "results/" + pModelOutput + '/logs/' + params + ".csv"
    
    newpath = r'results/' + pModelOutput + '/models/'
    if not os.path.exists(newpath):
        os.makedirs(newpath)
        
    newpath = r'results/' + pModelOutput + '/logs/'
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    
    if(myTrainingState.noOutputNeurons == 2):
        # spherical coordinates
        warnings.warn('conversion into spherical coordinates seems to be flawed atm')
        
        print('-> projecting dependent value into spherical coordinates')
        train_prevDirection, train_nextDirection = dwi_tools.convertIntoSphericalCoordsAndNormalize(train_prevDirection, train_nextDirection)    
    
    if (myTrainingState.usePretrainedModel):
        checkpoint = ModelCheckpoint(pModel, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        csv_logger = CSVLogger(pCSVLog)
        ptmodel = load_model(myTrainingState.pPretrainedModel)
        ptmodel.fit([train_DWI], [train_prevDirection, train_nextDirection], batch_size=myTrainingState.batch_size, epochs=myTrainingState.epochs, verbose=2,validation_split=0.2, callbacks=[checkpoint,csv_logger])
        return
    
    checkpoint = ModelCheckpoint(pModel, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    csv_logger = CSVLogger(pCSVLog)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=1e-5, verbose=1)

    ####################
    ####################
    if (myTrainingState.modelToUse == '1Dcnn'):
        train_DWI = np.reshape(train_DWI,[noSamples,myState.dim[0]*myState.dim[1]*myState.dim[2],noD])
        cnn = nn_helper.get_1DCNN(myTrainingState, inputShapeDWI = train_DWI.shape[1:])
        cnn.summary()
        cnn.fit([train_DWI], [train_nextDirection], batch_size=myTrainingState.batch_size, epochs=myTrainingState.epochs, verbose=2,validation_split=0.2, callbacks=[checkpoint,csv_logger])
    ####################
    ####################
    if (myTrainingState.modelToUse == '2Dcnn'):
        print(str(train_DWI.shape))
        cnn = nn_helper.get_2DCNN(myTrainingState, inputShapeDWI = train_DWI.shape[1:])
        cnn.summary()
        cnn.fit([train_DWI], [train_nextDirection], batch_size=myTrainingState.batch_size, epochs=myTrainingState.epochs, verbose=2,validation_split=0.2, callbacks=[checkpoint,csv_logger])
    ####################
    ####################
    elif (myTrainingState.modelToUse == '3Dcnn'):
        cnn = nn_helper.get_3DCNN(myTrainingState, inputShapeDWI = train_DWI.shape[1:])
        cnn.summary()
        cnn.fit([train_DWI], [train_nextDirection], batch_size=myTrainingState.batch_size, epochs=myTrainingState.epochs, verbose=2,validation_split=0.2, callbacks=[checkpoint,csv_logger])
    ###
    elif (myTrainingState.modelToUse == 'rcnn'):
        print(str(train_DWI.shape))
        train_DWI = np.reshape(train_DWI, [noSamples,16,16])
        train_DWI = train_DWI[..., np.newaxis]
        print(str(train_DWI.shape))
        cnn = nn_helper.get_rcnn(myTrainingState, inputShapeDWI = train_DWI.shape[1:])
        cnn.summary()
        cnn.fit([train_DWI], [train_nextDirection], batch_size=myTrainingState.batch_size, epochs=myTrainingState.epochs, verbose=2,validation_split=0.2, callbacks=[checkpoint,csv_logger])
    ###
        ### ### ###
    ### MLP SINGLE ###
        ### ### ###
    elif (myTrainingState.modelToUse == 'mlp_single'):
        class_weight = None
        
        if(myTrainingState.loss == 'sqCos2WEP'):
            noSamples = len(train_DWI)
            labels = np.zeros((noSamples,1))
            labels[idx1] = 1
            loss = 'sqCos2'
            
            noPosSamples = len(np.where(labels == 1)[0])
            noNegSamples = len(np.where(labels == 0)[0])
            class_weight = {1: (noPosSamples+noNegSamples) / noPosSamples,
                            0: (noPosSamples+noNegSamples) / noNegSamples}
            class_weight = { 'signLayer': {1: (noPosSamples+noNegSamples) / noPosSamples, 0: (noPosSamples+noNegSamples) / noNegSamples} }
            print(class_weight)
            print(str(train_nextDirection.shape))
            print(str(labels.shape))
        
            mlp_simple = nn_helper.get_mlp_singleOutputWEP(myTrainingState, inputShapeDWI = train_DWI.shape[1:])
            mlp_simple.summary()
            mlp_simple.fit([train_DWI], [train_nextDirection, labels], batch_size=myTrainingState.batch_size, epochs=myTrainingState.epochs, verbose=2,validation_split=0.2, callbacks=[checkpoint,csv_logger], class_weight=class_weight)
        else:
            mlp_simple = nn_helper.get_mlp_singleOutput(myTrainingState, inputShapeDWI = train_DWI.shape[1:])
            mlp_simple.summary()
            mlp_simple.fit([train_DWI], [train_nextDirection], batch_size=myTrainingState.batch_size, epochs=myTrainingState.epochs, verbose=2,validation_split=0.2, callbacks=[checkpoint,csv_logger])
    ###
        ### ### ###
    ### 2MLP SINGLE ###
        ### ### ###    
    elif (myTrainingState.modelToUse == '2mlp_single'):
        # load aggregated previous dwi coeffs
        f = h5py.File(myTrainingState.pTrainData, "r")
        train_DWI_pastAgg = np.array(f["train_DWI_pastAgg"].value)
        f.close()

        if(myTrainingState.loss == 'sqCos2WEP'):
            noSamples = len(train_DWI)
            labels = np.zeros((noSamples,1))
            labels[idx1] = 1
            loss = 'sqCos2'
        
        mlp_simple = nn_helper.get_2mlp_singleOutputWEP(myTrainingState)
        
        noPosSamples = len(np.where(labels == 1)[0])
        noNegSamples = len(np.where(labels == 0)[0])
        class_weight = {1: (noPosSamples+noNegSamples) / noPosSamples,
                        0: 10* (noPosSamples+noNegSamples) / noNegSamples}
        print(class_weight)

        class_weight = { 'signLayer': {1: (noPosSamples+noNegSamples) / noPosSamples, 0: (noPosSamples+noNegSamples) / noNegSamples} }
        mlp_simple.summary()
        mlp_simple.fit([train_DWI, train_DWI_pastAgg], [train_nextDirection, labels], batch_size=myTrainingState.batch_size, epochs=myTrainingState.epochs, verbose=2,validation_split=0.2, callbacks=[checkpoint,csv_logger])#, class_weight=class_weight)
    ###
    
    elif (myTrainingState.modelToUse == 'unet'):
        cnn_simple = nn_helper.get_3Dunet_simpleTracker(myTrainingState)
        cnn_simple.fit([train_DWI], [train_prevDirection, train_nextDirection], batch_size=myTrainingState.batch_size, epochs=myTrainingState.epochs, verbose=2,validation_split=0.2, callbacks=[checkpoint,csv_logger])

        
if __name__ == "__main__":
    main()
