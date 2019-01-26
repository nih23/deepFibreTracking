import numpy as np


class TractographyInformation:

    def __init__(self, model = 'MLP', representation = 'raw', tensormodel = 'dti', stepwidth = 1, b_value = 1000, dim = [1,1,1],
                 unitTangent = 0, faThreshold = 0.15, shOrder = 4, pPrecomputedStreamlines = '', rotateData = False, addRandomDataForEndpointPrediction = False,
                 gridSpacing = 1, noCrossingFibres = 1, nameDWIdataset = '', isISMRM = False, hcpID = '', usePreviousDirection = False, use2DProjection = False,
                 magicModel = False, pStopTracking = 0.5):
        ## 01/24/19: changed the default value to 0.25 as this is required by the rotation approach
        self.dim = dim
        self.model = model
        self.repr = representation
        self.tensorModel = tensormodel
        self.stepWidth = stepwidth
        self.b_value = b_value
        self.unitTangent = unitTangent
        self.faThreshold = faThreshold
        self.shOrder = shOrder
        self.pPrecomputedStreamlines = pPrecomputedStreamlines
        self.rotateData = rotateData
        self.addRandomDataForEndpointPrediction = addRandomDataForEndpointPrediction
        self.gridSpacing = gridSpacing
        self.noCrossingFibres = noCrossingFibres
        self.nameDWIdataset = nameDWIdataset
        self.isISMRM = isISMRM
        self.hcpID = hcpID
        self.referenceOrientation = np.array([0, 0, 1])
        self.usePreviousDirection = usePreviousDirection
        self.use2DProjection = use2DProjection
        self.magicModel = magicModel
        self.bvecs = None
        self.bvals = None
        self.b0 = None
        self.pStopTracking = pStopTracking

    def getReferenceOrientation(self):
        return self.referenceOrientation

    def parseModel(self, tracker):
        if (self.model.find("mlp_single") > 0):
            self.usePreviousDirection = False
            if (self.usePreviousDirection):
                noSamples, noX, noY, noZ, noC = tracker.get_input_shape_at(0)[0]
            else:
                noSamples, noX, noY, noZ, noC = tracker.get_input_shape_at(0)

        if ((self.model.find("cnn_special") > 0) or (self.model.find("rcnn") > 0)):
            self.use2DProjection = True
            self.usePreviousDirection = (self.model.find("cnn_special_pd") > 0) or (self.model.find("rcnn_pd") > 0)
            noSamples, noX, noY, noZ = tracker.get_input_shape_at(0)
            noC = noZ
            noX, noY, noZ = (1, 1, 1)

        elif (self.model.find("res1002D") > 0):
            self.use2DProjection = True
            if (self.usePreviousDirection):
                noSamples, noX, noY, noZ, noC = tracker.get_input_shape_at(0)[0]
            else:
                noSamples, noX, noY, noZ, noC = tracker.get_input_shape_at(0)
        elif (self.model.find("res100_") > 0):
            self.use2DProjection = False
            self.repr = 'res100'
            if (self.usePreviousDirection):
                noSamples, noX, noY, noZ, noC = tracker.get_input_shape_at(0)[0]
            else:
                noSamples, noX, noY, noZ, noC = tracker.get_input_shape_at(0)

        if (self.model.find("2Dcnn") > 0):
            self.repr = '2Dcnn'
            noSamples, noX, noY, noZ = tracker.get_input_shape_at(0)
            noC = -1
            noX = 1
            noY = 1
            noZ = 5 
            self.usePreviousDirection = False
            self.use2DProjection = False
            self.rotateData = False

        if (self.model.find("1Dcnn") > 0):
            self.repr = '1Dcnn'
            noSamples, noX, noY, noZ = tracker.get_input_shape_at(0)
            noC = -1
            noX = 1
            noY = 1
            self.usePreviousDirection = False
            self.use2DProjection = False
            self.rotateData = False

        elif (self.model.find("3Dcnn") > 0):
            self.repr = '3Dcnn'
            self.repr = 'raw'
            noSamples, noX, noY, noZ, noC = tracker.get_input_shape_at(0)
            self.usePreviousDirection = False
            self.use2DProjection = False

        elif (self.model.find("3DcnnProj") > 0):
            self.repr = '3DcnnProj'
            noSamples, noX, noY, noZ, noC = tracker.get_input_shape_at(0)
            noX = int(noX / 8)
            noY = int(noY / 8)
            self.usePreviousDirection = False
            self.use2DProjection = True

        self.magicModel = False

        self.dim = [noX,noY,noZ,noC]

        if (self.model.find("sqCos2WEP") > 0):
            print("Magic model :)")
            self.magicModel = True

        if (noC == 15 or noC == 45):
            print('Spherical Harmonics activated due to 15C or 45C.')
            self.repr = 'sh'

        #if (self.model.find("_raw_") > 0):
            #self.repr = 'raw'

        return self


class TrainingInformation:

    def __init__(self, pTrainData, loss, noFeatures, learningRate, depth, batch_size, epochs,
                 activationFunction, useDropout, useBatchNormalization, model, keepZeroVectors,
                 noGPUs = 1, usePretrainedModel = False, noOutputNeurons = 3, pDropout = 0.5, pPretrainedModel = '', dilationRate = 1):

        self.noGPUs = noGPUs
        self.pTrainData = pTrainData
        self.loss = loss
        self.noFeatures = noFeatures
        self.depth = depth
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = learningRate
        self.useDropout = useDropout
        self.useBatchNormalization = useBatchNormalization
        self.usePretrainedModel = usePretrainedModel
        self.modelToUse = model
        self.keepZeroVectors = keepZeroVectors
        self.activationFunction = activationFunction
        self.noOutputNeurons = noOutputNeurons
        self.pDropout = pDropout
        self.pPretrainedModel = pPretrainedModel
        self.dilationRate = dilationRate
