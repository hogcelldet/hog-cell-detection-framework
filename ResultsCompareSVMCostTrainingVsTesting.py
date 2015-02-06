# -*- coding: utf-8 -*-


# Libraries
import cv2
import numpy as np
import pylab as pl
import pickle
import datetime
import time
from scipy.integrate import trapz #,simps

# Files
import ImportData
import SVM
import DetectionProcess
import HardExamples


def add_subplot_axes(ax,rect,axisbg='w'):
    fig = pl.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position  = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)    
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]  # <= Typo was here
    subax = fig.add_axes([x,y,width,height],axisbg=axisbg)
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2]**0.5
    y_labelsize *= rect[3]**0.5
    subax.xaxis.set_tick_params(labelsize=0)
    subax.yaxis.set_tick_params(labelsize=0)
    return subax
    

if __name__ == '__main__':
    
    parameterToBeVaried = "Cost"    
    
    defaultHOG = cv2.HOGDescriptor()
    
    myParams = dict(
            _winSize           = (32,32),
            _blockSize         = (16,16),
            _blockStride       = (8,8),
            _cellSize          = (8,8),
            _nbins             = 9,
            _derivAperture     = defaultHOG.derivAperture,
            _winSigma          = defaultHOG.winSigma,
            _histogramNormType = defaultHOG.histogramNormType,
            _L2HysThreshold    = defaultHOG.L2HysThreshold,
            _gammaCorrection   = defaultHOG.gammaCorrection,
            _nlevels           = 1
            )
            
    hog = cv2.HOGDescriptor(**myParams)
    
    # Import data and extract HOG features
    trainData, trainClasses, labels, groundTruth = ImportData. \
    ImportDataAndExtractHOGFeatures(hog=hog,
                                    days=["day1","day2","day3"],
                                    saveAnnotations=False,
                                    thisManySamples=4000)

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
        maxIters=100, maxHardExamples=200000)
        
        trainData = np.concatenate((trainData,hE))
        trainClasses = np.concatenate((trainClasses, [0]*hE.shape[0]))
        labels  = np.concatenate((labels,hELabels))
    
    # Load from file
    if loadHardExamplesFromFile:
        with open('savedVariables/hardExamples.pickle') as f: 
            hE, hELabels = pickle.load(f)
    
    # Shuffle and maintain the same order in every array
    np.random.seed(666) #222
    shuffledOrder = range(hE.shape[0])
    np.random.shuffle(shuffledOrder)
    hE = np.asarray( [hE[i,:] for i in shuffledOrder] )
    hELabels = np.asarray( [hELabels[i] for i in shuffledOrder] )
    

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Divide into training and testing data
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # 50 % of HE to training and 50 % of HE to testing
    forTesting = 2000
    forTraining = 2000
    trainD, trainC, testD, testC = [],[],[],[]
    
    posExRowIndices = np.where(trainClasses==1)[0]
    negExRowIndices = np.where(trainClasses==0)[0]
    testD = np.concatenate((trainData[posExRowIndices[-forTesting:]],
                            trainData[negExRowIndices[-forTesting:]],
                            hE[np.int(np.floor(hE.shape[0]/2)):,:] ))
    trainD = np.concatenate((trainData[posExRowIndices[0:forTraining]],
                             trainData[negExRowIndices[0:forTraining]],
                             hE[:np.int(np.floor(hE.shape[0]/2)),:] ))
    trainL = np.concatenate((labels[posExRowIndices[0:forTraining]],
                             labels[negExRowIndices[0:forTraining]],
                             hELabels[:np.int(np.floor(hE.shape[0]/2))] ))
    testC = np.concatenate((trainClasses[posExRowIndices[-forTesting:]],
                            trainClasses[negExRowIndices[-forTesting:]],
                            ([0] * np.int(np.floor(hE.shape[0]/2))) ))
    trainC = np.concatenate((trainClasses[posExRowIndices[0:forTraining]],
                             trainClasses[negExRowIndices[0:forTraining]],
                             ([0] * np.int(np.floor(hE.shape[0]/2))) ))
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build classifiers with different values and calculate their ROC
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                
    totalResultsTraining = []
    totalResultsTesting = []
    
    cost = 10.0**(np.arange(-4,5,1))
    for C in cost:
        
        # Initialize dictionary of lists where resutls will be saved
        ROCResultsTraining = {}
        ROCResultsTraining["FPR"]               = [] # False Positive Rate
        ROCResultsTraining["TPR"]               = [] # True Positive Rate
        ROCResultsTraining["AUC"]               = [] # Area Under ROC
        ROCResultsTraining["fvl"]               = [] # feature vector length
        ROCResultsTraining["models"]            = []
        ROCResultsTraining[parameterToBeVaried] = []
								
        #cost = 10.0**(np.arange(-2,3,1))
        model = SVM.Train(trainD, trainC, cost=C)
        
								
								
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ROC in training
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        fpr, tpr, roc_auc = SVM.ROC(model, trainD, trainC, testD, testC)
        
        ROCResultsTraining["FPR"].append(fpr)
        ROCResultsTraining["TPR"].append(tpr)
        ROCResultsTraining["AUC"].append(roc_auc)
        ROCResultsTraining["fvl"].append(trainData.shape[1]) 
        ROCResultsTraining["models"].append(model)
        ROCResultsTraining[parameterToBeVaried].append(C)
								
        totalResultsTraining.append(ROCResultsTraining)
        
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ROC in testing
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~					
    
        searchMethod = "detectMultiScale"
        #searchMethod = "detect"
        
        #evaluate = ("singleImage", r".\testWithThese\day3\Tile002496.bmp")
        evaluate = ("folder",      r".\testWithThese")
        
        parametersToStudy = ["hitThreshold"]
        
        allParameterNamesAndRanges = dict(
        hitThreshold   = np.arange(5.0,-3.05,-0.05), 
        winStride      = [(s,s) for s in range(1,33,1)],
        padding        = [(s,s) for s in range(1,33,1)],
        scale          = np.arange(1.03,1.22,0.02),
        finalThreshold = np.arange(1,3.1,0.01),
        nlevels        = np.arange(1,11,1)#np.concatenate((np.arange(1,5,1),np.arange(14,65,10)))
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
        
        outputFolder = r".\\" + searchMethod + "_" + datetime.datetime. \
                       fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')
                               
        searchResults = DetectionProcess.OneOrMultipleImageSearch(hog, model, 
                        searchMethod, defaultParams, 
                        paramsAndTheirRangesToBeVaried, evaluate, outputFolder,
                        saveIm=False,
                        checkForBreakCondition1=True,
                        checkForBreakCondition2=False)
                        
        # Append current cost value ROC curve result in testing
        totalResultsTesting.append(searchResults)
                     
                     
                     
            
            
    if evaluate[0] == "singleImage":
        # Visualize results
        width = 3
        lineStyles = ["b","g","r","c","m","y","k","#FF00FF", "#00FFFF", "#FFFF00"]
        pl.close("all")
        fig = pl.figure(figsize=(13,6), facecolor='none')	
        ax1 = fig.add_subplot(1,2,1)
        ax2 = fig.add_subplot(1,2,2)	
        ax1.set_xlim(0,1)
        ax1.set_ylim(0,1)
        ax2.set_xlim(0,1)
        ax2.set_ylim(0,1)
        for i in range(len(totalResultsTraining)):
            
            ax1.plot(totalResultsTraining[i]["FPR"][0],
                     totalResultsTraining[i]["TPR"][0],
                     label="Cost " + str(totalResultsTraining[i]["Cost"][0]) + \
                     "  " * (8-len(str(totalResultsTraining[i]["Cost"][0]))) + \
                     "AUC " + str(totalResultsTraining[i]["AUC"][0])[0:5],
                     linewidth=width, color=lineStyles[i])
                    
            # Make sure that (0,0) is the first value
            totalResultsTesting[i]["FPR"] = [0.0] + totalResultsTesting[i]["FPR"]
            totalResultsTesting[i]["TPR"] = [0.0] + totalResultsTesting[i]["TPR"]
            # Delete points which take ROC curve backwards.
            # In other words, make sure that FPR values are increasing.
            t = 0
            while t+1 < len(totalResultsTesting[i]["FPR"]):
                if totalResultsTesting[i]["FPR"][t] >= totalResultsTesting[i]["FPR"][t+1]:
                    del totalResultsTesting[i]["FPR"][t+1]
                    del totalResultsTesting[i]["TPR"][t+1]
                t = t+1
            # FPR values can be >1.0 because of special TN calculation style.
            # Thus, interpolate TPR value at FPR=1.0.
            interpolatedLastValue = np.interp(1.0, totalResultsTesting[i]["FPR"], totalResultsTesting[i]["TPR"])
            totalResultsTesting[i]["FPR"][-1] = 1.0
            totalResultsTesting[i]["TPR"][-1] = interpolatedLastValue
            # Approximate AUC
            testingAUC = trapz(totalResultsTesting[i]["TPR"], totalResultsTesting[i]["FPR"])
            ax2.plot(totalResultsTesting[i]["FPR"],
                     totalResultsTesting[i]["TPR"],
                     label="Cost " + str(totalResultsTraining[i]["Cost"][0]) + \
                     "  " * (8-len(str(totalResultsTraining[i]["Cost"][0]))) + \
                     "AUC " + str(testingAUC)[0:5],
                     linewidth=width, color=lineStyles[i])
                     
        ax1.legend(loc="lower right", fontsize=12)        
        ax2.legend(loc="lower right", fontsize=12)
        ax1.set_xlabel('False positive rate (FPR)')
        ax1.set_ylabel('True positive rate (TPR)')
        ax2.set_xlabel('False positive rate (FPR)')
        ax2.set_ylabel('True positive rate (TPR)')
        
        pl.tight_layout()
        pl.draw()

        
        
        
    elif evaluate[0] == "folder":
        totalRes={}
        for ci, costIter in enumerate(totalResultsTesting):
            totalRes[ci] = {}
            totalRes[ci]["meanInterpCostFPRs"] = np.zeros(len(np.arange(0.0,1.01,0.01)))
            totalRes[ci]["meanInterpCostTPRs"] = np.zeros(len(np.arange(0.0,1.01,0.01)))
            for dayIter in costIter:
                for imIter in dayIter:
                    # Make sure that (0,0) is the first value
                    imIter["FPR"] = [0.0] + imIter["FPR"]
                    imIter["TPR"] = [0.0] + imIter["TPR"]
                    # Delete points which take ROC curve backwards.
                    # In other words, make sure that FPR values are increasing.
                    t = 0
                    while t+1 < len(imIter["FPR"]):
                        if imIter["FPR"][t] >= imIter["FPR"][t+1]:
                            del imIter["FPR"][t+1]
                            del imIter["TPR"][t+1]
                            continue
                        t += 1
                    # Interpolate
                    interpolatedVals = np.interp(np.arange(0.0,1.01,0.01), imIter["FPR"], imIter["TPR"])
                    imIter["FPR"] = np.arange(0.0,1.01,0.01)
                    imIter["TPR"] = interpolatedVals

                    totalRes[ci]["meanInterpCostFPRs"] = [x+y for x,y in zip(totalRes[ci]["meanInterpCostFPRs"], list(imIter["FPR"]))]
                    totalRes[ci]["meanInterpCostTPRs"] = [x+y for x,y in zip(totalRes[ci]["meanInterpCostTPRs"], list(imIter["TPR"]))]
            totalRes[ci]["meanInterpCostFPRs"] = [totalRes[ci]["meanInterpCostFPRs"][y]/12 for y in range(len(totalRes[ci]["meanInterpCostFPRs"]))]
            totalRes[ci]["meanInterpCostTPRs"] = [totalRes[ci]["meanInterpCostTPRs"][y]/12 for y in range(len(totalRes[ci]["meanInterpCostTPRs"]))]
                  
                  
        pl.close("all")
        width = 3
        lineStyles = ["b","g","r","c","m","y","k","#FF00FF", "#00FFFF", "#FFFF00"]
        fig = pl.figure(figsize=(13,6), facecolor='none')	
        ax1 = fig.add_subplot(1,2,1)
        ax2 = fig.add_subplot(1,2,2)	
        ax1.set_xlim(0,1)
        ax1.set_ylim(0,1)
        ax2.set_xlim(0,1)
        ax2.set_ylim(0,1)
        for ci in range(len(totalRes)):

            # Plot
            ax2.plot(totalRes[ci]["meanInterpCostFPRs"],
                    totalRes[ci]["meanInterpCostTPRs"],
                    label="Cost " + str(totalResultsTraining[ci]["Cost"][0]) + \
                    "  " * (8-len(str(totalResultsTraining[ci]["Cost"][0]))),
                    linewidth=width, color=lineStyles[ci])
            ax2.legend(loc="lower right", fontsize=12)  
    
    
