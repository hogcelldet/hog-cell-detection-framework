# -*- coding: utf-8 -*-

# Libraries
import os
import glob
import numpy as np
import cv2
import pickle
import datetime
import time

# Files
import ImportData
import SVM
import HardExamples
import DetectionProcess
import BivariateSimilarityIndex

# ----------------------------------------------------------------
 
# Returns a list of paths to subfolders
def Listdirs(folder):
    return [
        d for d in (os.path.join(folder, d1) for d1 in os.listdir(folder))
        if os.path.isdir(d)
    ] 
        
# ----------------------------------------------------------------
          
if __name__ == '__main__':
    
    defaultHOG = cv2.HOGDescriptor()

    myParams = dict(
            _winSize           = (32,32),
            _blockSize         = (8,8),
            _blockStride       = (4,4),
            _cellSize          = (4,4),
            _nbins             = 9,
            _derivAperture     = defaultHOG.derivAperture,
            _winSigma          = defaultHOG.winSigma,
            _histogramNormType = defaultHOG.histogramNormType,
            _L2HysThreshold    = defaultHOG.L2HysThreshold,
            _gammaCorrection   = defaultHOG.gammaCorrection,
            _nlevels           = 1 # Max number of HOG window scales
        )

    hog = cv2.HOGDescriptor(**myParams)
    
    # Import data and extract HOG features
    trainData, trainClasses, labels, groundTruth = ImportData. \
    ImportDataAndExtractHOGFeatures(hog=hog,
                                    days=["day1","day2","day3"],
                                    saveAnnotations=False,
                                    thisManySamples=2000)
                      
    # Shuffle and maintain the same order in every array
    np.random.seed(666) #222
    shuffledOrder = range(trainData.shape[0])
    np.random.shuffle(shuffledOrder)
    trainData = np.asarray( [trainData[i,:] for i in shuffledOrder] )
    trainClasses = np.asarray( [trainClasses[i] for i in shuffledOrder] )
    labels = np.asarray( [labels[i] for i in shuffledOrder] )
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Find hard examples
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    searchForHardExamples = True
    loadHardExamplesFromFile = False
    
    if searchForHardExamples:
        hE, hELabels, ROCResults = HardExamples.Search(hog, trainData, \
        trainClasses, labels, groundTruth, amountToInitialTraining=0.5, \
        saveImagesWithDetections=True, saveHardExampleImages=True,
        maxIters=10, maxHardExamples=2000)
        
        trainData = np.concatenate((trainData,hE))
        trainClasses = np.concatenate((trainClasses, [0]*hE.shape[0]))
        labels  = np.concatenate((labels,hELabels))
    
    # Load from file
    if loadHardExamplesFromFile:
        with open('savedVariables/hardExamples.pickle') as f: 
            hE, hELabels = pickle.load(f)
            
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Learn the data
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    learnNewModel = True
    
    if learnNewModel:
        cost = 0.1
        model = SVM.Train(trainData, trainClasses, cost=cost)
        
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Detect
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
    detectionParams = dict(
                           hitThreshold         = -model.intercept_[0],
                           winStride            = (2,2),
                           padding              = (8,8),
                           scale                = 1.05,
                           finalThreshold       = 2,
                           useMeanshiftGrouping = False
                           )
    
    print "\nEstimating growth curve:"
    print "------------------------------------------------------------" 
    
    hog.setSVMDetector( model.coef_[0] )     
    params = dict(
                 hitThreshold         = -model.intercept_[0],
                 winStride            = (4,4),
                 padding              = (8,8),
                 scale                = 1.05,
                 finalThreshold       = 2,
                 useMeanshiftGrouping = False
                 )
    
    searchMethod = "detectMultiScale"
    
    #evaluate = ("singleImage", r".\testWithThese\day3\Tile002496.bmp")
    evaluate = ("folder",      r".\testWithThese")
    
    outputFolder = r".\\" + searchMethod + "_" + datetime.datetime. \
                   fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S') 
                   
    parametersToStudy = []
                           
    searchResults = DetectionProcess.OneOrMultipleImageSearch(hog, model, 
                    searchMethod, params, 
                    [], evaluate, outputFolder, saveIm=True,
                    checkForBreakCondition1=False,
                    checkForBreakCondition2=False,
                    saveDetectionBinaryImages=True)

 
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Visualize the growth curve
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    # Average results from each day
    avgDetectedCells = []
    avgTruthCells = []
    avgTP = []
    avgFP = []
    avgFN = []
    index = 0
    for ix,day in enumerate(searchResults):
        imagesPerDay = 0
        avgDetectedCells.append(0)
        avgTruthCells.append(0)
        avgTP.append(0)
        avgFP.append(0)
        avgFN.append(0)
        # Sum results from this day
        for im in day:
            avgDetectedCells[ix] += im["TP"][0] + im["FP"][0]
            avgTruthCells[ix]    += im["nrOfCellsTruth"][0]
            imagesPerDay += 1
            avgTP[ix] += im["TP"][0]
            avgFP[ix] += im["FP"][0]
            avgFN[ix] += im["FN"][0]
        avgDetectedCells[-1] = avgDetectedCells[-1] / imagesPerDay
        avgTruthCells[-1] = avgTruthCells[-1] / imagesPerDay
        avgTP[-1] = avgTP[-1] / imagesPerDay
        avgFP[-1] = avgFP[-1] / imagesPerDay
        avgFN[-1] = avgFN[-1] / imagesPerDay

    avgNormDetectedCells = np.array(avgDetectedCells) / np.array(avgTruthCells[0] * np.ones(len(avgDetectedCells)))
    avgNormTruthCells    = np.array(avgTruthCells)    / np.array(avgTruthCells[0] * np.ones(len(avgTruthCells)))
    avgNormTP            = np.array(avgTP)            / np.array(avgTruthCells[0] * np.ones(len(avgTP)))
    avgNormFP            = np.array(avgFP)            / np.array(avgTruthCells[0] * np.ones(len(avgFP)))
    avgNormFN            = np.array(avgFN)            / np.array(avgTruthCells[0] * np.ones(len(avgFN)))
    
    # Visualize
    import pylab as pl
    #pl.close("all")
    fig = pl.figure(figsize=(14, 7), facecolor='white')
    ax1 = fig.add_subplot(1,2,1)
    ax1.grid(color='black', linestyle='-.', linewidth=1, alpha=0.2)
    ax1.plot(avgNormTruthCells, color="c", linestyle="-", marker="o", label="Manual", linewidth=3)
    ax1.plot(avgNormDetectedCells, color="m", linestyle="-", marker='o', label="HOG TP+FP", linewidth=3)
    ax1.plot(avgNormTP, "g--", label="HOG TP", linewidth=2)
    ax1.plot(avgNormFP, "r--", label="HOG FP", linewidth=2)
    ax1.plot(avgNormFN, "b--", label="HOG FN", linewidth=2)
    #ax1.set_xlim([0, max(uniqueDaysInteger)-1])
    #ax1.set_xticks(np.arange(1,7,1))
    ax1.set_xticklabels(["1","2","3","4","5","6"])
    ax1.set_xlabel('Experiment time (days)')
    ax1.set_ylabel('Relative number of cells ')
    ax1.legend(loc = 'upper left')
    pl.draw()
    
    relativeError = (np.abs(np.array(avgTruthCells) - np.array(avgDetectedCells))) / np.array(avgTruthCells)
    relativeError = relativeError[1:]
    np.max(relativeError)
    np.mean(relativeError)


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Calculate and visualize BSI
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ax2 = fig.add_subplot(1,2,2)
    ax2.grid(color='black', linestyle='-.', linewidth=1, alpha=0.2)
    
    colors = ["b","g","r","c","m","y","k"]
    truthDir = r".\groundTruth"
    estimDir = outputFolder + "\\detectionBinaries"
    fileTypes = ["bmp", 'jpg', 'png']    
    
    listOfDirTruth = Listdirs(truthDir)
    listOfDirEstim = Listdirs(estimDir)
    
    totalBSIres         = {}
    totalBSIres["TET"]  = []
    totalBSIres["TEE"]  = []
    totalBSIres["dSeg"] = []    
    
    # Loop through input folders
    for i in range(len(listOfDirTruth)):
        
        # Get list of images in the folder
        imageListTruth = []
        imageListEstim = []
        for fileType in fileTypes:
            imageListTruth = imageListTruth + glob.glob(listOfDirTruth[i] +  "\*." + fileType)
            imageListEstim = imageListEstim + glob.glob(listOfDirEstim[i] +  "\*." + fileType)
       
        # Loop through the images
        for j in range(len(imageListTruth)):
            
            if len(imageListTruth) > j and len(imageListEstim) > j:
                pathToTruthIm = imageListTruth[j]
                pathToEstimIm = imageListEstim[j]
                
                TET,TEE,dSeg = BivariateSimilarityIndex.BSI(pathToTruthIm,
                                                            pathToEstimIm)
                                                            
                totalBSIres["TET"].append(TET)
                totalBSIres["TEE"].append(TEE)
                totalBSIres["dSeg"].append(dSeg)
    
                # Draw label only once            
                if j==0:
                    ax2.scatter(TET, TEE, s=120.0, c=colors[i], label="Day " +str(i+1))
                else:
                    ax2.scatter(TET, TEE, s=120.0, c=colors[i])
                
        # Legend
        ax2.legend(loc="upper left", fontsize=12, scatterpoints=1)
        
    ax2.set_xlim(0,1)
    ax2.set_ylim(0,1)
    ax2.set_xlabel("TET")
    ax2.set_ylabel("TEE")
    pl.draw()
    
    print np.mean(totalBSIres["dSeg"])
    
