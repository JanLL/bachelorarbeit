import numpy as np
import vigra
from vigra import graphs
import copy
import pylab
import time
import sys
from scipy import stats


import inferno



#### Ground truth via watershed and seeds ####

def getGroundTruthWatershed(groundTruth, threshold=120):

    groundTruthThreshold = copy.deepcopy(groundTruth)
    groundTruthThreshold[groundTruthThreshold < threshold] = 0
    groundTruthThreshold[groundTruthThreshold >= threshold] = 255

    groundTruthLabels = vigra.analysis.labelImageWithBackground(groundTruthThreshold)

    ### Remove Mini-Label-Areas
    for i in range(1, groundTruthLabels.max()+1):
        if (len(groundTruthLabels[groundTruthLabels==i]) < 4):
            groundTruthLabels[groundTruthLabels==i] = 0
            
    groundTruthLabels = vigra.analysis.labelImageWithBackground(groundTruthLabels)
    
   
    numLabels = groundTruthLabels.max()
    

    groundTruthCont = copy.deepcopy(groundTruthThreshold)
    groundTruthCont[groundTruthCont==0] = 1
    groundTruthCont[groundTruthCont==255] = 0
    groundTruthCont = vigra.filters.gaussianSmoothing(groundTruthCont, 0.1)


    seeds = np.zeros((groundTruth.shape[0],groundTruth.shape[1]), dtype=np.uint32)
    for i in range(1, numLabels+1):
        labelArea = np.where(groundTruthLabels==i)
        x = labelArea[0][0]
        y = labelArea[1][0]
        seeds[x][y] = i

    watershedLabels = np.transpose(graphs.nodeWeightedWatersheds(rag.baseGraph, groundTruthCont, seeds))
    
    return watershedLabels

def getSuperpixelLabelList(rag, watershedLabels):
   
    numLabels = watershedLabels.max()
    
    superpixelLabelList = []
    for l in range(0, rag.baseGraphLabels.max()+1):
        nodeCoords = np.where(rag.baseGraphLabels==l)
        nodeNumberSP = len(nodeCoords[0])

        labelCounter = np.zeros(numLabels+1)
        for i in range(nodeNumberSP):
            x = nodeCoords[0][i]
            y = nodeCoords[1][i]

            labelCounter[watershedLabels[y, x]] += 1

        maxVoteLabel = labelCounter.argmax()
        superpixelLabelList.append(maxVoteLabel)   
        
    return np.uint32(superpixelLabelList)
        
        
def getGroundTruthSol(rag, superpixelLabelList):
    
    solGroundTruth = []
    for e in rag.edgeIter():
        if (superpixelLabelList[e.u.id] != superpixelLabelList[e.v.id]):
            solGroundTruth.append(1)
        else:
            solGroundTruth.append(0)

    return np.uint32(solGroundTruth)


def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def getNodeCoordsAroundEdge(rag, edge, numNodesAroundEdge):
    
    if (numNodesAroundEdge == 0):   # return Node coords of whole superpixels
        uCoords = np.where(rag.labels == edge.u.id)
        vCoords = np.where(rag.labels == edge.v.id)

        coordsU = []
        coordsV = []
        for i in range(len(uCoords[0])):
            coordsU.append(tuple((uCoords[0][i], uCoords[1][i])))
        for i in range(len(vCoords[0])):
            coordsV.append(tuple((vCoords[0][i], vCoords[1][i])))
        
        return coordsU, coordsV
    
    nodesCoordsU = []
    nodesCoordsV = []
    for i in range(len(rag.edgeUVCoordinates(edge)[0])):
        u = rag.edgeUVCoordinates(edge)[0][i]
        v = rag.edgeUVCoordinates(edge)[1][i]
        nodesCoordsU.append(tuple(u))
        nodesCoordsV.append(tuple(v))
        for prox in range(1, numNodesAroundEdge):
            if (u[0] == v[0]):
                tmpNode = copy.deepcopy(u)
                tmpNode[1] = (tmpNode[1] + prox)%rag.baseGraph.shape[1]
                nodesCoordsU.append(tuple(tmpNode))

                tmpNode = copy.deepcopy(v)
                tmpNode[1] = (tmpNode[1] - prox)%rag.baseGraph.shape[1]
                nodesCoordsV.append(tuple(tmpNode))

            if (u[1] == v[1]):
                tmpNode = copy.deepcopy(u)
                tmpNode[0] = (tmpNode[0] + prox)%rag.baseGraph.shape[0]
                nodesCoordsU.append(tuple(tmpNode))

                tmpNode = copy.deepcopy(v)
                tmpNode[0] = (tmpNode[0] - prox)%rag.baseGraph.shape[0]
                nodesCoordsV.append(tuple(tmpNode))
    
    return nodesCoordsU, nodesCoordsV



def getEdgeWeightsFromNodesAround3(rag, img, numNodesAroundEdge, variance=False, mean=False, meanRatio=False, 
                                                                 medianRatio=False, skewness=False, kurtosis=False):

    if (len(img.shape)==3):
        channels = img.shape[2]
    else:
        channels = 1

    tempFeatureList = []

    if (variance == True):
        variances = np.zeros((rag.edgeNum, channels))
        tempFeatureList.append(variances)

    if (mean == True):
        means = np.zeros((rag.edgeNum, channels))
        tempFeatureList.append(means)
    
    if (meanRatio == True):
        meanRatios = np.zeros((rag.edgeNum, channels))        
        tempFeatureList.append(meanRatios)

    if (medianRatio == True):
        medianRatios = np.zeros((rag.edgeNum, channels))
        tempFeatureList.append(medianRatios)

    if (skewness == True):
        skewnessFeature = np.zeros((rag.edgeNum, channels))
        tempFeatureList.append(skewnessFeature)

    if (kurtosis == True):
        kurtosisFeature = np.zeros((rag.edgeNum, channels))
        tempFeatureList.append(kurtosisFeature)
        
    
    for i, e in enumerate(rag.edgeIter()):
        nodeCoords = getNodeCoordsAroundEdge(rag, e, numNodesAroundEdge)
        nNodesU = len(nodeCoords[0])
        nNodesV = len(nodeCoords[1])
        

        valuesU = np.array(np.concatenate([img[ n[0], n[1], :].reshape((channels, 1)) for n in nodeCoords[0]], axis=1)).transpose()
        valuesV = np.array(np.concatenate([img[ n[0], n[1], :].reshape((channels, 1)) for n in nodeCoords[1]], axis=1)).transpose()
    
        
        if (variance == True):
            variances[i, :] = [ch.var() for ch in np.concatenate((valuesU, valuesV), axis=0).transpose()]

        if (mean == True):
            means[i, :] = [ch.mean() for ch in np.concatenate((valuesU, valuesV), axis=0).transpose()]
                
        if (meanRatio == True):
            for ch in range(channels):
                meanU = valuesU[:,ch].mean()
                meanV = valuesV[:,ch].mean()
                meanRatios[i, ch] = max(meanU, meanV) / (min(meanU, meanV) + 1e-8)
                    
                
        if (medianRatio == True):
            for ch in range(channels):
                medianU = np.median(valuesU[:,ch])
                medianV = np.median(valuesV[:,ch])
                medianRatios[i, ch] = max(medianU, medianV) / (min(medianU, medianV) + 1e-8)

        if (skewness == True):
            skewnessFeature[i, :] = [stats.skew(ch) for ch in np.concatenate((valuesU, valuesV), axis=0).transpose()]

        if (kurtosis == True):
            kurtosisFeature[i, :] = [stats.kurtosis(ch) for ch in np.concatenate((valuesU, valuesV), axis=0).transpose()]

    # normalize to [0, 1]
    for ch in range(channels):
        if (variance == True):
            variances[:,ch] /= variances[:,ch].max()
        if (mean == True):
            means[:,ch] /= means[:,ch].max()
        if (meanRatio == True):
            meanRatios[:,ch] /= meanRatios[:,ch].max()
        if (medianRatio == True):
            medianRatios[:,ch] /= medianRatios[:,ch].max()
        if (skewness == True):
            skewnessFeature[:,ch] /= skewnessFeature[:,ch].max()
        if (kurtosis == True):
            kurtosisFeature[:,ch] /= kurtosisFeature[:,ch].max()
       
    
    edgeWeightsFromNodesAround = np.concatenate(tempFeatureList, axis=1)
    
    return edgeWeightsFromNodesAround


def getFeatures(rag, img, imgId):

    featureNames = ['1-Feature']
    ############################## Filter ###################################################
    filters = []
    ### Gradient Magnitude ###
    imgLab = vigra.colors.transform_RGB2Lab(img)
    imgLabBig = vigra.resize(imgLab, [imgLab.shape[0]*2-1, imgLab.shape[1]*2-1])  ##### was ist der Vorteil hiervon? #####
    filters.append(vigra.filters.gaussianGradientMagnitude(imgLabBig, 1.))
    featureNames.append('GradMag1')
    filters.append(vigra.filters.gaussianGradientMagnitude(imgLabBig, 2.))
    featureNames.append('GradMag2')
    filters.append(vigra.filters.gaussianGradientMagnitude(imgLabBig, 5.))
    featureNames.append('GradMag5')

    
    ### Hessian of Gaussian Eigenvalues ###
    sigmahoG     = 2.0
    hoG = vigra.filters.hessianOfGaussianEigenvalues(rgb2gray(imgLab), sigmahoG)  
    filters.append(hoG[:,:,0])
    featureNames.append('HessGauss1')
    filters.append(hoG[:,:,1])
    featureNames.append('HessGauss2')

    
    ### Laplacian of Gaussian ###
    loG = vigra.filters.laplacianOfGaussian(imgLab)
    loG = loG[:,:,0]  # es gilt hier: loG[:,:,i] = loG[:,:,j], i,j = 1,2,3
    filters.append(loG)
    featureNames.append('LoG')

    ### Canny Filter ###
    scaleCanny = 2.0
    thresholdCanny = 2.0
    markerCanny = 1
    canny = vigra.VigraArray(vigra.analysis.cannyEdgeImage(rgb2gray(img), scaleCanny, 
                                                           thresholdCanny, markerCanny), 
                             dtype=np.float32)
    filters.append(canny)
    featureNames.append('Canny')

    
    ### Structure Tensor Eigenvalues ###
    strucTens = vigra.filters.structureTensorEigenvalues(imgLab, 0.7, 0.7)
    filters.append(strucTens[:,:,0])
    featureNames.append('StrucTensor1')
    filters.append(strucTens[:,:,1])
    featureNames.append('StrucTensor2')
    

    n4 = vigra.impex.readImage('images/edgeDetectors/n4/' + imgId + '.png')
    filters.append(n4)
    featureNames.append('N4')

    dollar = vigra.impex.readImage('images/edgeDetectors/dollar/' + imgId + '.png')
    filters.append(dollar)
    featureNames.append('Dollar')
    
        
        
    ##########################################################################################
    ############# Edge Weights Calculation #############
    edgeWeightsList = []
    featureSpace = np.ones((rag.edgeNum, 1))
    
    
    for i in range(len(filters)):
        gridGraphEdgeIndicator = graphs.edgeFeaturesFromImage(rag.baseGraph, filters[i]) 
        edgeWeights = rag.accumulateEdgeFeatures(gridGraphEdgeIndicator)
        #edgeWeights /= edgeWeights.max()
        edgeWeights = edgeWeights.reshape(edgeWeights.shape[0], 1)
        featureSpace = np.concatenate((featureSpace, edgeWeights), axis=1)
                
          
    pos = np.where(np.array(featureNames)=='N4')[0][0]
    edgeWeights = featureSpace[:,pos] * rag.edgeLengths()
    edgeWeights /= edgeWeights.max()
    edgeWeights = edgeWeights.reshape(edgeWeights.shape[0], 1)    
    featureSpace = np.concatenate((featureSpace, edgeWeights), axis=1)
    featureNames.append('N4_EdgeLengthWeighted')
                
    pos = np.where(np.array(featureNames)=='Dollar')[0][0]
    edgeWeights = featureSpace[:,pos] * rag.edgeLengths()
    edgeWeights /= edgeWeights.max()
    edgeWeights = edgeWeights.reshape(edgeWeights.shape[0], 1)    
    featureSpace = np.concatenate((featureSpace, edgeWeights), axis=1)
    featureNames.append('Dollar_EdgeLengthWeighted')



    rgbDummy = np.array(n4)
    rgbDummy = rgbDummy.reshape(rgbDummy.shape[0], rgbDummy.shape[1], 1)
    edgeWeights = getEdgeWeightsFromNodesAround3(rag, rgbDummy, 1, variance=True, mean=True, meanRatio=True, medianRatio=True, skewness=True, kurtosis=True)
    featureSpace = np.concatenate((featureSpace, edgeWeights), axis=1)
    featureNames.extend(('N4_Variance_1', 'N4_Mean_1', 'N4_MeanRatio_1', 'N4_MedianRatio_1', 'N4_Skewness_1', 'N4_Kurtosis_1'))
    
    edgeWeights = getEdgeWeightsFromNodesAround3(rag, rgbDummy, 3, variance=True, mean=True, meanRatio=True, medianRatio=True, skewness=True, kurtosis=True)
    featureSpace = np.concatenate((featureSpace, edgeWeights), axis=1)
    featureNames.extend(('N4_Variance_3', 'N4_Mean_3', 'N4_MeanRatio_3', 'N4_MedianRatio_3', 'N4_Skewness_3', 'N4_Kurtosis_3'))


    rgbDummy = np.array(dollar)
    rgbDummy = rgbDummy.reshape(rgbDummy.shape[0], rgbDummy.shape[1], 1)
    edgeWeights = getEdgeWeightsFromNodesAround3(rag, rgbDummy, 1, variance=True, mean=True, meanRatio=True, medianRatio=True, skewness=True, kurtosis=True)
    featureSpace = np.concatenate((featureSpace, edgeWeights), axis=1)
    featureNames.extend(('Dollar_Variance_1', 'Dollar_Mean_1', 'Dollar_MeanRatio_1', 'Dollar_MedianRatio_1', 'Dollar_Skewness_1', 'Dollar_Kurtosis_1'))
    
    edgeWeights = getEdgeWeightsFromNodesAround3(rag, rgbDummy, 3, variance=True, mean=True, meanRatio=True, medianRatio=True, skewness=True, kurtosis=True)
    featureSpace = np.concatenate((featureSpace, edgeWeights), axis=1)
    featureNames.extend(('Dollar_Variance_3', 'Dollar_Mean_3', 'Dollar_MeanRatio_3', 'Dollar_MedianRatio_3', 'Dollar_Skewness_3', 'Dollar_Kurtosis_3'))


            
    edgeWeights = getEdgeWeightsFromNodesAround3(rag, imgLab, 1, variance=True, mean=True, meanRatio=True, medianRatio=True, skewness=True, kurtosis=True)
    featureSpace = np.concatenate((featureSpace, edgeWeights), axis=1)
    featureNames.extend(('Variance_1_R', 'Variance_1_G', 'Variance_1_B',
                         'Mean_1_R', 'Mean_1_G', 'Mean_1_B', 
                         'MeanRatio_1_R', 'MeanRatio_1_G', 'MeanRatio_1_B',
                         'MedianRatio_1_R', 'MedianRatio_1_G', 'MedianRatio_1_B',
                         'Skewness_1_R', 'Skewness_1_G', 'Skewness_1_B', 
                         'Kurtosis_1_R', 'Kurtosis_1_G', 'Kurtosis_1_B'))
    edgeWeights = getEdgeWeightsFromNodesAround3(rag, imgLab, 3, variance=True, mean=True, meanRatio=True, medianRatio=True, skewness=True, kurtosis=True)
    featureSpace = np.concatenate((featureSpace, edgeWeights), axis=1)
    featureNames.extend(('Variance_3_R', 'Variance_3_G', 'Variance_3_B',
                         'Mean_3_R', 'Mean_3_G', 'Mean_3_B', 
                         'MeanRatio_3_R', 'MeanRatio_3_G', 'MeanRatio_3_B',
                         'MedianRatio_3_R', 'MedianRatio_3_G', 'MedianRatio_3_B',
                         'Skewness_3_R', 'Skewness_3_G', 'Skewness_3_B', 
                         'Kurtosis_3_R', 'Kurtosis_3_G', 'Kurtosis_3_B'))
    
    featureSpace = featureSpace.astype(np.float64)

    ### Normalize to [-1, 1]
    '''for edgeWeights in featureSpace.transpose():
        if (edgeWeights.min() < 0):
            edgeWeights -= edgeWeights.min()
        maximum = edgeWeights.max() 
        edgeWeights *= 2
        edgeWeights /= maximum
        edgeWeights -= 1
    '''

    ### Normalize to [0, 1]
    for edgeWeights in featureSpace.transpose():
        if (edgeWeights.min() < 0):
            edgeWeights -= edgeWeights.min()
        edgeWeights /= edgeWeights.max()
    
    return featureSpace, featureNames

def buildRandomForest(edgeFeatures, gtSols, filename=None):
    
    RF = vigra.learning.RandomForest()

    RF_labels = np.concatenate(gtSols)
    RF_labels = RF_labels.reshape(RF_labels.shape[0], 1)

    RF_features = np.concatenate(edgeFeatures).astype(np.float32)

    oob = RF.learnRF(RF_features, RF_labels)

    if (filename != None):
        RF.writeHDF5(filename)
    
    return RF

def getProbsFromRF(features, RF=None, filename=None):
        
    cuts = []
    for i in range(len(features)-1):
        if (len(cuts) == 0):
            cuts.append(np.int32(features[i].shape[0]))
        else:
            cuts.append(np.int32(features[i].shape[0] + cuts[-1]))
    RF_features = np.concatenate(features).astype(np.float32)
    
    probList = []
    if (RF != None):
        probs = RF.predictProbabilities(RF_features)
        for prob in np.split(probs, cuts):
            probList.append(prob)
        return probList
        
    elif (filename != None):
        RF_local = vigra.learning.RandomForest(filename)
        probs = RF_local.predictProbabilities(RF_features)
        for prob in np.split(probs, cuts):
            probList.append(prob)
        return probList
        
    else:
        print "No Random Forest source given!"
        return -1

def performLearning(trainingFeatureSpaces, trainingRags, trainingEdges, gtLabels, loss, 
    regularizerStr=1., learnerParameter=None, start=None, weightConstraints=None):

    validLosses = ['partitionHamming', 'variationOfInformation']
    if (loss not in validLosses):
        raise NameError('Chosen Loss invalid! Valid Losses are in ' + validLosses.__str__())
        return
    
    ParaMcModel = inferno.models.ParametrizedMulticutModel

    nTrainSamples = len(trainingFeatureSpaces)
    nFeatures = trainingFeatureSpaces[0].shape[1]
    
    modelVec = ParaMcModel.modelVector(nTrainSamples)
    if (loss == 'partitionHamming'):
        lossFctVec = ParaMcModel.lossFunctionVector2('partitionHamming',nTrainSamples)
    
    elif (loss == 'variationOfInformation'):
        lossFctVec = ParaMcModel.lossFunctionVector2('variationOfInformation2',nTrainSamples)
    gtVec = ParaMcModel.groundTruthVector(nTrainSamples)

    if (start == None):
        weightVector = inferno.learning.WeightVector(nFeatures, 0.0)
    else:
        weightVector = start

    if (weightConstraints == None):
        weightConstraints = inferno.learning.WeightConstraints(nFeatures)
    


    for n in range(nTrainSamples):

        nVar = trainingRags[n].nodeNum
        modelVec[n]._assign(nVar, trainingEdges[n]-1, trainingFeatureSpaces[n], weightVector) # -1 at edges that nodes start at 0

        if (loss == 'partitionHamming'):
            lossFctVec[n] = inferno.learning.loss_functions.partitionHamming(modelVec[n], rescale=1.0, overseg=1.0, underseg=1.5)
        
        elif (loss == 'variationOfInformation'):
            sizeMap = modelVec[n].variableMap('float64', 1.0)
            sizeMapView = sizeMap.view()
            for l in range(1, trainingRags[n].maxNodeId+1):
                sizeMapView[l-1] = np.count_nonzero(np.array(trainingRags[n].baseGraphLabels==l, dtype=np.int8))
            lossFctVec[n] = inferno.learning.loss_functions.variationOfInformation2(model=modelVec[n], variableSizeMap=sizeMap)

        gt = gtVec[n]
        gt.assign(modelVec[n])
        gtView = gt.view()
        gtView[:] = gtLabels[n][1:]

    regType = inferno.learning.RegularizerType.L2
    regularizer = inferno.learning.Regularizer(regularizerType=regType, c=regularizerStr)

    dataSet = inferno.learning.dataset.defaultDataset(modelVec, lossFctVec, gtVec, weightConstraints, regularizer=regularizer)

    mcFac = inferno.inference.multicutFactory(ParaMcModel)
    lossAugMcFac = inferno.inference.multicutFactory(ParaMcModel.lossAugmentedModelClass2('partitionHamming'))


    if (loss == 'partitionHamming'):
        if (learnerParameter == None):
            learnerParameter = dict(maxIterations=70)
        learnerSubGrad = inferno.learning.learners.subGradient(dataSet, **learnerParameter)
        t1 = time.time()
        print "Begin weight learning..."
        sys.stdout.flush()
        learnerSubGrad.learn(lossAugMcFac, weightVector, mcFac)
        t2 = time.time()
    
    elif (loss == 'variationOfInformation'):
        if (learnerParameter == None):
            learnerParameter = dict(maxIterations=30, nPertubations=7, sigma=1.5)
        learnerStochGrad = inferno.learning.learners.stochasticGradient(dataSet, **learnerParameter)
        t1 = time.time()
        print "Begin weight learning..."
        sys.stdout.flush()
        learnerStochGrad.learn(mcFac, weightVector)
        t2 = time.time()
        
    print "Learning Duration: ", t2-t1, "sec"
    
    return weightVector


def performTesting(testImgs, testRags, testEdges, testFeatureSpaces, testIds, weightVector, resultsPath):

    ParaMcModel = inferno.models.ParametrizedMulticutModel

    nTestSamples = len(testImgs)

    modelVec = ParaMcModel.modelVector(nTestSamples)

    for n in range(nTestSamples):
        nVar = testRags[n].nodeNum
        modelVec[n]._assign(nVar, testEdges[n]-1, testFeatureSpaces[n], weightVector)



    for i in range(nTestSamples):

        solver = inferno.inference.multicut(modelVec[i])

        visitor = inferno.inference.verboseVisitor(modelVec[i])
        solver.infer(visitor.visitor())

        conf = solver.conf()

        arg = conf.view().astype('uint32')
        arg = np.array([0] + list(arg), dtype=np.uint32) + 1

        fig = pylab.figure(frameon=False)
        
        # make figure without frame
        ax = pylab.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        
        testRags[i].show(testImgs[i], labels=arg, edgeColor=(1,0,0), alpha=0.)

        
        fig.savefig(resultsPath + str(testIds[i]) + '.tif')
        
        del fig, ax


