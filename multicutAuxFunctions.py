import numpy as np
import vigra
from vigra import graphs
import copy
import pylab
import time
import sys

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


def getEdgeWeightsFromNodesAround2(rag, img, numNodesAroundEdge, variance=False, meanRatio=False, medianRatio=False):

    if (variance == True):
        varianceR = np.zeros((rag.edgeNum,1))
        varianceG = np.zeros((rag.edgeNum,1))
        varianceB = np.zeros((rag.edgeNum,1))
    
    if (meanRatio == True):
        meanRatioR = np.zeros((rag.edgeNum,1))
        meanRatioG = np.zeros((rag.edgeNum,1))
        meanRatioB = np.zeros((rag.edgeNum,1))
        
    if (medianRatio == True):
        medianRatioR = np.zeros((rag.edgeNum,1))
        medianRatioG = np.zeros((rag.edgeNum,1))
        medianRatioB = np.zeros((rag.edgeNum,1))
    
    for i, e in enumerate(rag.edgeIter()):
        nodeCoords = getNodeCoordsAroundEdge(rag, e, numNodesAroundEdge)
        nNodesU = len(nodeCoords[0])
        nNodesV = len(nodeCoords[1])
        
        if (variance == True):
            currVarR = np.zeros(nNodesU + nNodesV)
            currVarG = np.zeros(nNodesU + nNodesV)
            currVarB = np.zeros(nNodesU + nNodesV)
            
        if (meanRatio == True or medianRatio):
            valuesUR = np.zeros(nNodesU)
            valuesUG = np.zeros(nNodesU)
            valuesUB = np.zeros(nNodesU)
            
            valuesVR = np.zeros(nNodesV)
            valuesVG = np.zeros(nNodesV)
            valuesVB = np.zeros(nNodesV)
            
        for j, n in enumerate(nodeCoords[0]):
            if (variance == True):
                currVarR[j] = img[ n[0], n[1], 0 ]
                currVarG[j] = img[ n[0], n[1], 1 ]
                currVarB[j] = img[ n[0], n[1], 2 ]
                
            if (meanRatio == True or medianRatio == True):
                valuesUR[j] = img[ n[0], n[1], 0 ]
                valuesUG[j] = img[ n[0], n[1], 1 ]
                valuesUB[j] = img[ n[0], n[1], 2 ]
                
        for j, n in enumerate(nodeCoords[1]):
            if (variance == True):
                currVarR[j+nNodesU] = img[ n[0], n[1], 0 ]
                currVarG[j+nNodesU] = img[ n[0], n[1], 1 ]
                currVarB[j+nNodesU] = img[ n[0], n[1], 2 ]
                
            if (meanRatio == True or medianRatio == True):
                valuesVR[j] = img[ n[0], n[1], 0 ]
                valuesVG[j] = img[ n[0], n[1], 1 ]
                valuesVB[j] = img[ n[0], n[1], 2 ]
    
        if (variance == True):
            varianceR[i] = currVarR.var()
            varianceG[i] = currVarG.var()
            varianceB[i] = currVarB.var()
            
        if (meanRatio == True):
            meanUR = valuesUR.mean()
            meanVR = valuesVR.mean()
            meanRatioR[i] = max(meanUR, meanVR) / (min(meanUR, meanVR) + 1e-8)
            meanUG = valuesUG.mean()
            meanVG = valuesVG.mean()
            meanRatioG[i] = max(meanUG, meanVG) / (min(meanUG, meanVG) + 1e-8)
            meanUB = valuesUB.mean()
            meanVB = valuesVB.mean()
            meanRatioB[i] = max(meanUB, meanVB) / (min(meanUB, meanVB) + 1e-8)
            
        if (medianRatio == True):
            medianUR = np.median(valuesUR)
            medianVR = np.median(valuesVR)
            medianRatioR[i] = max(medianUR, medianVR) / (min(medianUR, medianVR) + 1e-8)
            medianUG = np.median(valuesUG)
            medianVG = np.median(valuesVG)
            medianRatioG[i] = max(medianUG, medianVG) / (min(medianUG, medianVG) + 1e-8)
            medianUB = np.median(valuesUB)
            medianVB = np.median(valuesVB)
            medianRatioB[i] = max(medianUB, medianVB) / (min(medianUB, medianVB) + 1e-8)
            
    varianceR /= varianceR.max()
    varianceG /= varianceG.max()
    varianceB /= varianceB.max()
    
    meanRatioR /= meanRatioR.max()
    meanRatioG /= meanRatioG.max()
    meanRatioB /= meanRatioB.max()
    
    medianRatioR /= medianRatioR.max()
    medianRatioG /= medianRatioG.max()
    medianRatioB /= medianRatioB.max()
    
    edgeWeightsFromNodesAround = np.concatenate((varianceR, varianceG, varianceB, 
                                                 meanRatioR, meanRatioG, meanRatioB,
                                                 medianRatioR, medianRatioG, medianRatioB), axis=1)
    
    return edgeWeightsFromNodesAround

            


def getFeatures(rag, img, imgId):

    featureNames = ['1-Feature']
    ############################## Filter ###################################################
    filters = []
    ### Gradient Magnitude ###
    sigmaGradMag = 2.0       # sigma Gaussian gradient
    imgLab = vigra.colors.transform_RGB2Lab(img)
    imgLabBig = vigra.resize(imgLab, [imgLab.shape[0]*2-1, imgLab.shape[1]*2-1])  ##### was ist der Vorteil hiervon? #####
    filters.append(vigra.filters.gaussianGradientMagnitude(imgLabBig, 1.))
    featureNames.append('GradMag1')
    filters.append(vigra.filters.gaussianGradientMagnitude(imgLabBig, 2.))
    featureNames.append('GradMag2')
    filters.append(vigra.filters.gaussianGradientMagnitude(imgLabBig, 5.))
    featureNames.append('GradMag5')

    
    ### Hessian of Gaussian ###
    sigmahoG     = 2.0
    hoG = vigra.filters.hessianOfGaussian2D(rgb2gray(imgLab), sigmahoG)   # hoG[i]:0 vertical part, 1 diagonal, 2 horizontal
    filters.append(hoG[:,:,0])
    featureNames.append('HessGaussY')
    filters.append(hoG[:,:,1])
    featureNames.append('HessGaussXY')
    filters.append(hoG[:,:,2])
    featureNames.append('HessGaussX')
    
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


    
    strucTens = vigra.filters.structureTensorEigenvalues(imgLab, 0.7, 0.7)
    filters.append(strucTens[:,:,0])
    featureNames.append('StrucTensor1')
    filters.append(strucTens[:,:,1])
    featureNames.append('StrucTensor2')
    
    filters.append(vigra.impex.readImage('images/edgeDetectors/n4/' + imgId + '.png'))
    featureNames.append('N4')
    filters.append(vigra.impex.readImage('images/edgeDetectors/dollar/' + imgId + '.png'))
    featureNames.append('Dollar')
        
        
        
    ##########################################################################################
    ############# Edge Weights Calculation #############
    edgeWeightsList = []
    featureSpace = np.ones((rag.edgeNum, 1))
    
    
    for i in range(len(filters)):
        gridGraphEdgeIndicator = graphs.edgeFeaturesFromImage(rag.baseGraph, filters[i]) 
        edgeWeights = rag.accumulateEdgeFeatures(gridGraphEdgeIndicator)
        edgeWeights /= edgeWeights.max()
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
            
            
    edgeWeights = getEdgeWeightsFromNodesAround2(rag, imgLab, 1, variance=True, meanRatio=True, medianRatio=True)
    featureSpace = np.concatenate((featureSpace, edgeWeights), axis=1)
    featureNames.extend(('Variance_1_R', 'Variance_1_G', 'Variance_1_B',
                         'MeanRatio_1_R', 'MeanRatio_1_G', 'MeanRatio_1_B',
                         'MedianRatio_1_R', 'MedianRatio_1_G', 'MedianRatio_1_B'))
    edgeWeights = getEdgeWeightsFromNodesAround2(rag, imgLab, 3, variance=True, meanRatio=True, medianRatio=True)
    featureSpace = np.concatenate((featureSpace, edgeWeights), axis=1)
    featureNames.extend(('Variance_3_R', 'Variance_3_G', 'Variance_3_B',
                         'MeanRatio_3_R', 'MeanRatio_3_G', 'MeanRatio_3_B',
                         'MedianRatio_3_R', 'MedianRatio_3_G', 'MedianRatio_3_B'))
    
    featureSpace = featureSpace.astype(np.float64)
    
    return featureSpace, featureNames

def buildRandomForest(edgeFeatures, gtSols, filename='RF.hdf5'):
    
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

def performLearning(trainingFeatureSpaces, trainingRags, trainingEdges, gtLabels, loss, LearnerParameter=None):

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
    gtVec = ParaMcModel.groundTruthVector(nTrainSamples)e nur von platinblonden Rave-Kids getragen, nun ero

    weightVector = inferno.learning.WeightVector(nFeatures, 0.0)
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


    dataSet = inferno.learning.dataset.defaultDataset(modelVec, lossFctVec, gtVec, weightConstraints)

    mcFac = inferno.inference.multicutFactory(ParaMcModel)
    lossAugMcFac = inferno.inference.multicutFactory(ParaMcModel.lossAugmentedModelClass2('partitionHamming'))

    if (loss == 'partitionHamming'):
        paramSubGrad = dict(maxIterations=70)
        learnerSubGrad = inferno.learning.learners.subGradient(dataSet, **paramSubGrad)
        t1 = time.time()
        print "Begin weight learning..."
        sys.stdout.flush()
        learnerSubGrad.learn(lossAugMcFac, weightVector, mcFac)
        t2 = time.time()
    
    elif (loss == 'variationOfInformation'):
        paramStochGrad = dict(maxIterations=30, nPertubations=7, sigma=1.5)
        learnerStochGrad = inferno.learning.learners.stochasticGradient(dataSet, **paramStochGrad)
        t1 = time.time()
        print "Begin weight learning..."
        sys.stdout.flush()
        learnerStochGrad.learn(mcFac, weightVector)
        t2 = time.time()
        
    print "Learning Duration: ", t2-t1, "sec"
    
    return weightVector

