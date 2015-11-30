import numpy as np
import matplotlib.pyplot as plt
import vigra
from vigra import graphs
import copy
import os
import pylab
import time
from scipy.io import loadmat
import sys

import inferno
import multicutAuxFunctions as maf


resultsPath = 'results/151130_runTest/'

if not os.path.exists(resultsPath):
    os.makedirs(resultsPath)
    os.makedirs(resultsPath + 'featureSpaces/training')
    os.makedirs(resultsPath + 'featureSpaces/testing')
    os.makedirs(resultsPath + 'partitionHamming/segmentedImages')
    os.makedirs(resultsPath + 'VOI/segmentedImages')

################################### Training Data ########################################

trainSetPath = 'trainingSet/'
#trainSetPath = 'fewImages/'


path = os.walk(trainSetPath)

trainingIds = []
trainingImgs = []

trainingGtLabels = []
trainingGtSols = []

trainingRags = []

superpixelDiameter = 20      # super-pixel size
slicWeight         = 25.0    # SLIC color - spatial weight
beta               = 0.5     # node vs edge weight
nodeNumStop        = 50      # desired num. nodes in result
minSize            = 15

print "Loading Training Data..."
############# load images and convert to LAB #############
for root, dirs, files in path:
    jpgFiles = [filename for filename in files if filename.endswith('.jpg')]
    T = len(jpgFiles)
    for i, filename in enumerate(jpgFiles):
        
        sys.stdout.write('\r')
        sys.stdout.write("[%-50s] %d%%" % ('='*int(float(i+1)/T*50), int(float(i+1)/T*100)))
        sys.stdout.flush()

        fileId = filename[:-4]
        trainingIds.append(fileId)
        
        img = vigra.impex.readImage(root + '/' + filename)
        trainingImgs.append(img)
        imgLab = vigra.colors.transform_RGB2Lab(img)
            
        gridGraph = graphs.gridGraph(img.shape[0:2])

        slicLabels = vigra.analysis.labelImage(vigra.analysis.slicSuperpixels(imgLab, slicWeight, superpixelDiameter, minSize=minSize)[0])
        rag = graphs.regionAdjacencyGraph(gridGraph, slicLabels)
        trainingRags.append(rag)
        
        gtWatershed = loadmat('trainingSet/groundTruth/' + fileId + '.mat')['groundTruth'][0,0][0][0][0]
        gtLabel = maf.getSuperpixelLabelList(rag, gtWatershed)
        trainingGtLabels.append(gtLabel)
        
        trainingGtSols.append(maf.getGroundTruthSol(rag, gtLabel))

### Feature Spaces
# Training

testingFeatureSpacesPath = resultsPath + 'featureSpaces/training/'

trainingFeatureSpaces = []
trainingEdges = []
t1 = time.time()
T = len(trainingImgs)
print "\nBuilding up Training Feature Space..."
sys.stdout.flush()
for i, (rag, img, trainId) in enumerate(zip(trainingRags, trainingImgs, trainingIds)):
    trainingEdges.append(rag.uvIds().astype('uint64'))
    
    if (os.path.isfile(testingFeatureSpacesPath + trainId + '.npy') == True):
        features = np.load(testingFeatureSpacesPath + trainId + '.npy')
        if (os.path.isfile(resultsPath + 'featureSpaces/featureNames.npy') == True):
            featureNames = list(np.load(resultsPath + 'featureSpaces/featureNames.npy'))
        
    else:
        features, featureNames = maf.getFeatures(rag, img, trainId)
        np.save(testingFeatureSpacesPath + trainId + '.npy', features)
        np.save(resultsPath + 'featureSpaces/featureNames.npy', featureNames)
        
    
    sys.stdout.write('\r')
    sys.stdout.write("[%-50s] %d%%" % ('='*int(float(i+1)/T*50), int(float(i+1)/T*100)))
    sys.stdout.flush()
    
    trainingFeatureSpaces.append(features)
t2 = time.time()

print "\nTime to built up Training Feature Space:", t2-t1, "sec"


############################################# Testing Data #################################################

testSetPath = 'testSet/'
#testSetPath = 'fewImages/'

path = os.walk(testSetPath)

testImgs = []
testIds = []

testRags = []

testingGtLabels = []
testingGtSols = []


superpixelDiameter = 20       # super-pixel size
slicWeight         = 25.0     # SLIC color - spatial weight
beta               = 0.5     # node vs edge weight
nodeNumStop        = 50         # desired num. nodes in result
minSize            = 15

print "Loading Testing Data...\n"
############# load images and convert to LAB #############
T = len(os.listdir(testSetPath))
for root, dirs, files in path:
    for i, filename in enumerate(files):
        if (filename.endswith('.jpg')):
            
			sys.stdout.write('\r')
			sys.stdout.write("[%-50s] %d%%" % ('='*int(float(i+1)/T*50), int(float(i+1)/T*100)))
			sys.stdout.flush()

			fileId = filename[:-4]
			testIds.append(fileId)
			img = vigra.impex.readImage(root + filename)
			imgLab = vigra.colors.transform_RGB2Lab(img)
			testImgs.append(img)

			gridGraph = graphs.gridGraph(img.shape[0:2])

			slicLabels = vigra.analysis.labelImage(vigra.analysis.slicSuperpixels(imgLab, slicWeight, superpixelDiameter, minSize=minSize)[0])
			rag = graphs.regionAdjacencyGraph(gridGraph, slicLabels)
			testRags.append(rag) 

			gtWatershed = loadmat('trainingSet/groundTruth/' + fileId + '.mat')['groundTruth'][0,0][0][0][0]
			gtLabel = maf.getSuperpixelLabelList(rag, gtWatershed)
			testingGtLabels.append(gtLabel)

			testingGtSols.append(maf.getGroundTruthSol(rag, gtLabel))       

            
testFeatureSpaces = []
testEdges = []

testingFeatureSpacesPath = resultsPath + 'featureSpaces/testing/'
if not os.path.exists(testingFeatureSpacesPath):
    os.makedirs(testingFeatureSpacesPath)

t1 = time.time()
print "\nBuilding up Testing Feature Space..."
T = len(testImgs)
for i, (rag, img, testId) in enumerate(zip(testRags, testImgs, testIds)):
    testEdges.append(rag.uvIds().astype('uint64'))
    
    if (os.path.isfile(testingFeatureSpacesPath + testId + '.npy') == True):
        features = np.load(testingFeatureSpacesPath + testId + '.npy')
        
    else:
        features = maf.getFeatures(rag, img, testId)[0]
        np.save(testingFeatureSpacesPath + testId + '.npy', features)

    sys.stdout.write('\r')
    sys.stdout.write("[%-50s] %d%%" % ('='*int(float(i+1)/T*50), int(float(i+1)/T*100)))
    sys.stdout.flush()

    testFeatureSpaces.append(features)
t2 = time.time()

print "\nTime to built up Testing Feature Space:", t2-t1, "sec"



########################## PartitionHamming Learner ########################################



### PartitionHamming

nFeatures = trainingFeatureSpaces[0].shape[1]
weightConstraints = inferno.learning.WeightConstraints(nFeatures)
weightConstraints.addBound(1, -1.01, -0.99)

subGradParameter = dict(maxIterations=30, nThreads=4, n=0.1)
weightVector = maf.performLearning(trainingFeatureSpaces, trainingRags, trainingEdges, trainingGtLabels,
                                   loss='partitionHamming', learnerParameter=subGradParameter, 
                                   weightConstraints=weightConstraints, regularizerStr=1.)

np.save(resultsPath + 'partitionHamming/weights.npy', weightVector)

maf.performTesting2(testImgs, testRags, testEdges, testFeatureSpaces, testIds, testingGtLabels, 
					featureNames, weightVector, resultsPath + 'partitionHamming/')


########################## Add Random Forest Feature ###################################

rfPath = resultsPath + 'RF.hdf5'
if (os.path.isfile(rfPath)):
    RF = vigra.learning.RandomForest(rfPath)
else:
    RF = maf.buildRandomForest(trainingFeatureSpaces, trainingGtSols, rfPath)

trainingRfProbs = maf.getProbsFromRF(trainingFeatureSpaces, RF)

trainingFeatureSpaces[:] = [np.concatenate((featureSpace, (prob[:,1]).reshape(prob.shape[0],1)), axis=1) for featureSpace, prob in zip(trainingFeatureSpaces, trainingRfProbs)]
featureNames.append('RF_Prob')








