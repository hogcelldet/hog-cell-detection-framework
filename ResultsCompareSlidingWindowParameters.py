# -*- coding: utf-8 -*-

# Libraries
import cv2
import numpy as np
import pickle
import pylab as pl
import datetime
import time
from mpl_toolkits.mplot3d import Axes3D

# Files
import ImportData
import SVM
import HardExamples
import DetectionProcess

if __name__ == '__main__':
    
    defaultHOG = cv2.HOGDescriptor()

    myParams = dict(
            _winSize           = (32,32), # Window size
            _blockSize         = (16,16), # Block size
            _blockStride       = (8,8),   # Block step size
            _cellSize          = (4,4),   # Cell size
            _nbins             = 9,       # Number of orientation bins
            _derivAperture     = defaultHOG.derivAperture,
            _winSigma          = defaultHOG.winSigma,
            _histogramNormType = defaultHOG.histogramNormType,
            _L2HysThreshold    = defaultHOG.L2HysThreshold,
            _gammaCorrection   = defaultHOG.gammaCorrection,
            _nlevels           = 1 # Max number of HOG window increases
        )

    hog = cv2.HOGDescriptor(**myParams)
    
    # Import data and extract HOG features
    days=["day1","day2","day3"]
    saveAnnotations=False
    thisManySamples=100#"all"
    trainData, trainClasses, labels, groundTruth = ImportData. \
    ImportDataAndExtractHOGFeatures(hog, days, saveAnnotations,thisManySamples)
    
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
        trainClasses, labels, groundTruth, amountToInitialTraining=1.0, \
        saveImagesWithDetections=True, saveHardExampleImages=True,
        maxIters=1000, maxHardExamples=200000, calculateROC=False,
        ROCforThisManyFirstIters=5)
        
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
    saveModelToHardDrive = False
    
    if learnNewModel:
        cost = 10.0**(np.arange(-2,3,1))
        model = SVM.Train(trainData, trainClasses,
                          cost=cost, CVtype="lolo", labels=labels)
        #cost = 0.1
        #model = SVM.Train(trainData, trainClasses, cost=cost)

    else:
        with open('savedVariables/trainData,trainClasses,labels,model.pickle')\
        as f:
            trainData, trainClasses, labels, model = pickle.load(f)
            
    if saveModelToHardDrive:
        with open('savedVariables/trainData,trainClasses,model_30_1', 'w')as f:
            pickle.dump([trainData, trainClasses, model], f)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Vary detection parameters
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  
    
    searchMethod = "detectMultiScale"
    #searchMethod = "detect"
    
    evaluate = ("singleImage", r".\testWithThese\day2\Tile002496.bmp")
    #evaluate = ("folder",      r".\testWithThese")
    #evaluate = ("folder",      r".\trainWithThese")
    
    #parametersToStudy = ["scale", "winStride"]
    parametersToStudy = ["nlevels"]
    
    allParameterNamesAndRanges = dict(
    hitThreshold   = [0.1,0.0,-0.1],#np.arange(5.0,-5.0,-0.1),
    winStride      = [(s,s) for s in range(5,10,2)],#[(s,s) for s in range(1,9,1)],
    padding        = [(s,s) for s in range(1,33,1)],
    scale          = [1.0,1.05,1.1],#np.arange(1.00,1.09,0.01),
    finalThreshold = np.arange(1.2,2.8,0.2),
    nlevels        = [1,10,20,64]#np.arange(1,11,1)
    )
    
    # Set default search parameters
    if searchMethod == "detectMultiScale":
        defaultParams = dict(
                             hitThreshold         = -model.intercept_[0],
                             winStride            = (4,4),
                             padding              = (8,8),
                             scale                = 1.05,
                             finalThreshold       = 2,
                             useMeanshiftGrouping = False
                     )
    elif searchMethod == "detect":
        defaultParams = dict(
                             hitThreshold = -model.intercept_[0],
                             winStride    = (4,4),
                             padding      = (8,8)
                             )
                     
    # Pick up ranges for selected parameters
    if len(parametersToStudy) == 1:
        paramsAndTheirRangesToBeVaried = (parametersToStudy[0], 
                           allParameterNamesAndRanges[parametersToStudy[0]])
    elif len(parametersToStudy) == 2:
        paramsAndTheirRangesToBeVaried = (parametersToStudy[0], 
                           allParameterNamesAndRanges[parametersToStudy[0]],
                           parametersToStudy[1], 
                           allParameterNamesAndRanges[parametersToStudy[1]])
                           
    saveIm = True
    checkForBreakCondition1 = False
    checkForBreakCondition2 = False
    if "hitThreshold" in parametersToStudy:
        checkForBreakCondition1=True
        
        
    outputFolder = r".\\" + searchMethod + "_" + datetime.datetime. \
                   fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S') 
    
    searchResults = DetectionProcess.OneOrMultipleImageSearch(hog, model, 
                    searchMethod, defaultParams, 
                    paramsAndTheirRangesToBeVaried, evaluate, outputFolder,
                    saveIm,
                    checkForBreakCondition1,
                    checkForBreakCondition2)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Analyze results
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   
        
    if len(parametersToStudy) == 1:
        drawF1 = 1
        drawROC = 0
        
        
        if evaluate[0] == "singleImage":
            
            if drawROC:
                fig = pl.figure(figsize=(7, 7), facecolor='white')
                pl.plot(searchResults["FPR"], searchResults["TPR"], lw=2)
                pl.xlim(0,1)
                pl.ylim(0,1)
                
            if drawF1:
                fig = pl.figure(figsize=(7, 7), facecolor='white')
                ax1 = fig.add_subplot(1,1,1)
                ax1.plot(range(len(searchResults["F1"])),
                         searchResults["F1"], lw=2)
                ax1.set_xlabel(parametersToStudy[0])
                ax1.set_ylabel("F1-score")
                ax1.set_xticks(range(len(paramsAndTheirRangesToBeVaried[1])))
                ax1.set_xticklabels(
                [str(x) for x in paramsAndTheirRangesToBeVaried[1]])
                pl.ylim(0,1)
                pl.tight_layout()
                #pl.draw()
                pl.show()
                
        
        if evaluate[0] == "folder":
            
            if drawROC:
                fig = pl.figure(figsize=(7, 7), facecolor='white')
                for folder in searchResults:
                    for image in folder:
                        pl.plot(image["FPR"], image["TPR"], lw=2)
                        pl.xlim(0,1)
                        pl.ylim(0,1)
            
            # Plot F1-score as a function of varied parameter
            elif drawF1:
                pl.close("all")
                fig = pl.figure(figsize=(6, 6), facecolor='white')
                ax1 = fig.add_subplot(1,1,1)
                ax1.set_ylim(0,1)
                #ax1.set_xlabel("SVM sensitivity")
                ax1.set_xlabel(parametersToStudy[0])
                ax1.set_ylabel("F1-score")
                
                if parametersToStudy[0] == "hitThreshold":
                    # Determine which img had shortest range of F1-values
                    # searched, so that equally short results can be drawn
                    # between images from each day, according to the shortest
                    # range of F1-values searched. Different images have
                    # different search lengths because of "checkForBreakCondition1"
                    # premature search break. We did not always search the whole
                    # range of hitThreshold values to save some time. We stopped
                    # the search short after reaching peak F1 scores.
                    F1Lengths = []                                                     
                    for i,folder in enumerate(searchResults):
                        for j,image in enumerate(folder):
                            F1Lengths.append(len(image["F1"]))
                    shortestF1 = np.min(F1Lengths)
                    xRange = allParameterNamesAndRanges[parametersToStudy[0]] \
                             [0:shortestF1]
                    ax1.set_xlim(np.min(xRange),np.max(xRange)) 
                else:
                    # Other than hitThreshold parameters: use the full range
                    # of parameter values.
                    xRange = range(len(paramsAndTheirRangesToBeVaried[1]))
                    shortestF1 = len(xRange)
                    ax1.set_xticks(xRange)
                    ax1.set_xticklabels(
                    [str(x) for x in paramsAndTheirRangesToBeVaried[1]])
                
                # Average for each varied parameter value (on x-axis)
                # for each day
                meanF1res = []
                for i,folder in enumerate(searchResults):
                    # Initialize list for the mean scores
                    meanF1res.append(np.zeros(shortestF1))
                    # Add results together
                    for j,image in enumerate(folder):
                        for k in range(shortestF1):
                            meanF1res[i][k] += image["F1"][k]
                
                # Average and plot
                colors = ["b","g","r","c","m","y","k"]
                bestMeanF1 = []
                bestMeanXVal = []
                for i in range(len(meanF1res)):
                    # Average results by dividing by the number of images
                    # taken each day
                    for j in range(len(meanF1res[i])):
                        meanF1res[i][j] /= len(searchResults[i])
                    # Plot
                    ax1.plot(xRange, meanF1res[i], lw=3, color=colors[i],
                             label="Day " + str(i+1))
                    ax1.legend(loc="best", fontsize=12)
                    # Stem best F1 score
                    bestMeanF1.append(np.max(meanF1res[i]))
                    
                    if parametersToStudy[0] == "hitThreshold":
                        bestMeanXVal.append(xRange[np.argmax(meanF1res[i])])
                    else:
                        bestMeanXVal.append( \
                        allParameterNamesAndRanges[parametersToStudy[0]] \
                        [xRange[np.argmax(meanF1res[i])]])
                        
                    markerline = pl.stem([xRange[np.argmax(meanF1res[i])]],
                                         [np.max(meanF1res[i])],colors[i]+'-.')
                    pl.setp(markerline, 'markerfacecolor', colors[i])
                    
                if parametersToStudy[0] == "hitThreshold":
                    ax1.annotate("Mean\n"+str(round(np.mean(bestMeanXVal), 1)),
                            xy=(np.mean(bestMeanXVal), 0),  xycoords='data',
                            xytext=(0.4, 0.15), textcoords='axes fraction', 
                            arrowprops=dict(facecolor='black', width=2,
                            headwidth=10, shrink=0.05),
                            horizontalalignment='right',
                            verticalalignment='top')
                else:    
                    ax1.annotate("Mean\n"+str(round(np.mean(bestMeanXVal), 1)),
                            xy=(np.where(
                            allParameterNamesAndRanges[parametersToStudy[0]]==
                            np.mean(bestMeanXVal))[0], 0),  xycoords='data',
                            xytext=(0.4, 0.15), textcoords='axes fraction', 
                            arrowprops=dict(facecolor='black', width=2,
                            headwidth=10, shrink=0.05),
                            horizontalalignment='right',
                            verticalalignment='top')
                
                #pl.draw()
                pl.show()
                
                
                # Calculate mean of best mean F1 scores
                bestF1res = []#np.zeros(len(searchResults))           
                for i,folder in enumerate(searchResults):
                    for j,image in enumerate(folder):
                        bestF1res.append(np.max(image["F1"]))
                print np.mean(bestF1res)
    
    
    
    
    # Draw 3D barplot
    elif len(parametersToStudy) == 2:
        
        # Rows: parametersToStudy[0]
        # Cols: parametersToStudy[1]
        meanRes = np.zeros(( len(paramsAndTheirRangesToBeVaried[1]),
                             len(paramsAndTheirRangesToBeVaried[3]) ))
        meanTimeTaken = np.zeros(( len(paramsAndTheirRangesToBeVaried[1]),
                                   len(paramsAndTheirRangesToBeVaried[3]) ))
        
        
        if evaluate[0] == "folder":
            
            # Sum the results of all images
            nrOfImages = 0
            for folder in searchResults:
                for image in folder:
                    nrOfImages += 1
                    for i in range(meanRes.shape[0]):
                        for j in range(meanRes.shape[1]):
                            meanRes[i,j] += image["F1"][i+j]
                            meanTimeTaken[i,j] += image["timeTaken"][i+j]
            # Take average: divide the values by the number of images
            for i in range(meanRes.shape[0]):
                for j in range(meanRes.shape[1]):
                    meanRes[i,j] /= nrOfImages
                    meanTimeTaken[i,j] /= nrOfImages
                    
        if evaluate[0] == "singleImage":
            
            for i in range(meanRes.shape[0]):
                for j in range(meanRes.shape[1]):
                    meanRes[i,j] = searchResults["F1"][i+j]
                    meanTimeTaken[i,j] = searchResults["timeTaken"][i+j]
                
        # Determine the locations at which the bars start
        xpos = np.repeat(
        range(len(allParameterNamesAndRanges[parametersToStudy[0]])),
              len(allParameterNamesAndRanges[parametersToStudy[1]]))
        ypos = np.tile(
        range(len(allParameterNamesAndRanges[parametersToStudy[1]])),
              len(allParameterNamesAndRanges[parametersToStudy[0]]))
        zpos = np.zeros(len(xpos)) # height
        
        # Determine step sizes: width, depth, height
        dx = np.ones(len(xpos))    
        dy = np.ones(len(ypos))
        dz = meanRes.flatten()
        dzTime = meanTimeTaken.flatten()
        
        # Normalize dz to [0,1] for colormap 
        dzNormalized = dz
        maxF1,minF1 = np.max(dzNormalized), np.min(dzNormalized)
        dzNormalized = (dzNormalized - minF1)/(maxF1 - minF1)
        colors = pl.cm.cool(dzNormalized)
        
        # Normalize dzTime to [0,1] for colormap 
        dzTimeNormalized = dzTime
        timeMax,timeMin = np.max(dzTimeNormalized), np.min(dzTimeNormalized)
        dzTimeNormalized = (dzTimeNormalized - timeMin)/(timeMax - timeMin)
        colorsTime = pl.cm.cool(dzTimeNormalized)
        
        # Plot scores
        fig = pl.figure(figsize=(13,6), facecolor='white')
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d') # Plot time taken
        ax1.bar3d(xpos, ypos, zpos, dx, dy, dz,     color=colors,     alpha=0.6)
        ax2.bar3d(xpos, ypos, zpos, dx, dy, dzTime, color=colorsTime, alpha=0.6)
        ax1.set_xlabel(parametersToStudy[0],fontsize=14, fontweight='bold', color='k')
        ax2.set_xlabel(parametersToStudy[0],fontsize=14, fontweight='bold', color='k')
        ax1.set_ylabel(parametersToStudy[1],fontsize=14, fontweight='bold', color='k')
        ax2.set_ylabel(parametersToStudy[1],fontsize=14, fontweight='bold', color='k')
        ax1.set_zlabel('F1-score',                   fontsize=14, fontweight='bold', color='k')
        ax2.set_zlabel('Computation time (seconds)', fontsize=14, fontweight='bold', color='k')
        ax1.set_zlim(0, 1) 
        #ax1.set_xlim(xLimit)
        #ax1.set_ylim(yLimit)
        #ax2.set_xlim(xLimit)
        #ax2.set_ylim(yLimit)
        ax1.set_xticks(np.unique(xpos))
        ax1.set_yticks(np.unique(ypos))
        ax2.set_xticks(np.unique(xpos))
        ax2.set_yticks(np.unique(ypos))
        ax1.set_xticklabels([str(x) for x in allParameterNamesAndRanges[parametersToStudy[0]]])
        ax1.set_yticklabels([str(x) for x in allParameterNamesAndRanges[parametersToStudy[1]]])
        ax2.set_xticklabels([str(x) for x in allParameterNamesAndRanges[parametersToStudy[0]]])
        ax2.set_yticklabels([str(x) for x in allParameterNamesAndRanges[parametersToStudy[1]]])
        # Set viewing angles, 1st argument: z-height, 2nd: angle
        ax1.view_init(26,60)
        ax2.view_init(26,60)
        pl.tight_layout()
        #pl.draw()
        pl.show()
    
    