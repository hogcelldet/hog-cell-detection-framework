# -*- coding: utf-8 -*-

# Libraries
import os
import glob
import time
import datetime
import numpy as np
import cv2
import scipy
import uuid

# Files
import SVM
import DetectionProcess
import Filters
import MeasurePerformance
        
# ----------------------------------------------------------------

# Returns a list of paths to subfolders
def Listdirs(folder):
    return [
        d for d in (os.path.join(folder, d1) for d1 in os.listdir(folder))
        if os.path.isdir(d)
    ] 
    
# ----------------------------------------------------------------

def Search(hog, trainData, trainClasses, labels,
         groundTruth, amountToInitialTraining=1.0,
         saveImagesWithDetections=False, saveHardExampleImages=True,
         maxIters=1000, maxHardExamples=200000, calculateROC=False,
         ROCforThisManyFirstIters=5):
    
    # Initialize dictionary of lists where resutls will be saved
    ROCResults = {}
    ROCResults["FPR"]               = [] # False Positive Rate
    ROCResults["TPR"]               = [] # True Positive Rate
    ROCResults["AUC"]               = [] # Area Under ROC
    ROCResults["iter"]              = []    
    ROCResults["nrOfIterHE"]        = []
    ROCResults["F1"]                = []
    
    totalHardExamples = []
    totalHardExampleLabels = []
    
    # find hard examples using smaller amount of data.
    # else, use all the data.
    if amountToInitialTraining != 1.0:
        negExRowIndices = np.where(trainClasses==0)[0]
        posExRowIndices = np.where(trainClasses==1)[0]
        
        tDNegInd = negExRowIndices[:int(amountToInitialTraining*len(negExRowIndices))]
        tDPosInd = posExRowIndices[:int(amountToInitialTraining*len(posExRowIndices))]
    
        trainData = np.concatenate((trainData[tDNegInd],trainData[tDPosInd]))
        trainClasses = np.concatenate((trainClasses[tDNegInd],trainClasses[tDPosInd]))
        labels  = np.concatenate((labels[tDNegInd],labels[tDPosInd]))

    if saveHardExampleImages or saveImagesWithDetections:
        # Output folder name
        parentFolder = "hardExamples_" +  \
            datetime.datetime.fromtimestamp(time.time()). \
            strftime('%Y-%m-%d_%H-%M-%S')
        # Create parent output folder if does not exist yet
        if not os.path.exists(parentFolder):
            os.makedirs(parentFolder)
            
    if calculateROC:
        ROCResults, cost = ROC(trainData, trainClasses, labels, ROCResults)
        ROCResults["iter"].append(1) # First iteration
        ROCResults["nrOfIterHE"].append(0) # Zero hard examples
    
    for i in np.arange(2,maxIters,1):

        iterHardExamples = []    
        iterHardExampleLabels = []
        
        # Search and build SVM model.
        # If ROC was calculated last on last iteration, we already have
        # cross-validated cost.
        if calculateROC and i<=ROCforThisManyFirstIters:
            model = SVM.Train(trainData, trainClasses, cost=cost)
        else: # Else, cross-validate new cost value
            cost = 10.0**(np.arange(-2,3,1))
            model = SVM.Train(trainData, trainClasses,
                              cost=cost, CVtype="lolo", labels=labels)
                
        # Use the model to detect cells from already seen images
        # and compare them to ground truth in order to find false positives
        w = model.coef_[0]
        hog.setSVMDetector( w ) 
        
        searchMethod = "detectMultiScale"
        
        params = dict(
                     hitThreshold         = -model.intercept_[0],
                     winStride            = (2,2), # IMPORTANT! if same as blockStride, no detections will be produced
                     padding              = (0,0), # IMPORTANT! if bigger than (0,0), detections can have minus values which cropping does not like
                     scale                = 1.05,
                     finalThreshold       = 2,
                     useMeanshiftGrouping = False
                     )
        
        # Input folder location and possible image types
        listOfDayDirs = Listdirs(r".\trainWithThese")
        fileTypes = ["bmp", 'jpg', 'png']

        # Loop through input folders (days)
        for ii, directory in enumerate(listOfDayDirs):
            
            # Search hard examples only from images from these days,
            # because subsequent day images are not annotated 100% correctly
            # and thus "false positives" might be actual positives
            if not "day1" in directory and \
               not "day2" in directory and \
               not "day3" in directory:
                continue
            
            # Get list of images in the folder
            imageList = []
            for fileType in fileTypes:
                imageList = imageList + glob.glob(directory +  "\*." + fileType)
           
            # Loop through images
            for j,imFileName in enumerate(imageList):
                
                # Image name without file type extension
                imName = imFileName[imFileName.rfind("\\")+1 : \
                imFileName.rfind(".")]                
                
                print "\nProcessing " + directory[directory.rfind("\\")+1 : ] + \
                " image " + str(j+1) + "/" + str(len(imageList)) + "..."                  
                
                # Open current image
                img = cv2.imread(imFileName, cv2.CV_LOAD_IMAGE_GRAYSCALE)
                
                # Second copy that remains untouched for cropping
                imgOrig = cv2.imread(imFileName, cv2.CV_LOAD_IMAGE_GRAYSCALE)
                
                # Detect
                found, timeTaken, w = DetectionProcess.SlidingWindow(hog, img, \
                searchMethod, params, filterDetections=False)
                
                if saveHardExampleImages:
                    # Create the folder if it does not exist already
                    childFolder = parentFolder + "\\" + "hardExamples_iter" + str(i)
                    if not os.path.exists(childFolder):
                        os.makedirs(childFolder)
                    
                # Find hard examples
                for ri,r in enumerate(found):
                    for qi,q in enumerate(groundTruth[directory][imName]["positiveExamples"]):

                        # True if overlaps at least x %...
                        if Filters.Overlap(r, q) > 0.0000001:
                            break
                        
                        # This example is false positive if it overlaps
                        # less than x % with any of the true positives
                        elif (qi == len(groundTruth[directory][imName]["positiveExamples"])-1):
                            
                            # You can set minimum weight/confidence threshold
                            # for hard examples here.
                            if w[ri] > 0.0:

                                cropped = imgOrig[ r[1]:r[1]+r[3], r[0]:r[0]+r[2] ]
                                
                                # Crop & resize
                                cropped = cv2.resize(cropped, hog.winSize)
        
                                # Generate feature
                                feature = hog.compute(cropped)[:,0]
                                
                                iterHardExamples.append(feature)
                                iterHardExampleLabels.append(ii)
           
                                # Save the image  
                                if saveHardExampleImages:
                                    cv2.imwrite(childFolder + "\\" + imName + "_" + \
                                    str(uuid.uuid4().fields[-1])[:5] + ".png", cropped)
                                    
                # Save the results in .INI.
                # Create the folder where at least detections.INI files will 
                # be saved. Images with detections will be saved over there
                # as well later, if input argument of this function says so.
                childFolder = parentFolder + "\\" + \
                directory[directory.rfind("\\")+1 : ] + "_imagesWithDetections"
                # Create the folder if it does not exist already
                if not os.path.exists(childFolder):
                    os.makedirs(childFolder)
                    
                pathToDetectionsIni = childFolder+"\\"+imName+"_iter" + str(i)+".ini"
                
                DetectionProcess.SaveIni(found, pathToDetectionsIni,
                         searchMethod, imFileName, hog)

                # Analyze the results, build confusion matrix
                pathToTruthIni = imFileName[:imFileName.rfind(".")]+"_annotations.ini"
                
                TP, FP, FN, TPR, FPR, F1, F05, F09, nrOfCellsTruth, imWithDetections = \
                MeasurePerformance.Measure \
                (pathToDetectionsIni, pathToTruthIni, imFileName)
                
                ROCResults["F1"].append(F1)
                
                if saveImagesWithDetections:                   
                    # Save the image with detections
                    scipy.misc.imsave(childFolder + "\\" + \
                    imFileName[imFileName.rfind("\\")+1 : imFileName.rfind(".")] \
                    + "_iter" + str(i) + ".png", imWithDetections)
                    
        # If no hard examples were found, draw ROC for the last time and exit
        if len(iterHardExamples) == 0:
            ROCResults, cost = ROC(trainData, trainClasses, labels, ROCResults)
            ROCResults["iter"].append(i)
            ROCResults["nrOfIterHE"].append(0)
            break
        
        # Concatenate
        totalHardExamples = totalHardExamples + iterHardExamples
        totalHardExampleLabels = totalHardExampleLabels + iterHardExampleLabels
        
        # List to array
        iterHardExampleLabels = np.asarray(iterHardExampleLabels)
        iterHardExamples = np.asarray(iterHardExamples)
        
        # Append
        trainData = np.concatenate((trainData,iterHardExamples))
        trainClasses = np.concatenate((trainClasses,([0] * iterHardExamples.shape[0])))
        labels  = np.concatenate((labels, iterHardExampleLabels))
        
        # Save the number of hard examples on first iteration
        if i == 2:
            nrOfHeFirstIter = iterHardExamples.shape[0]
        
        # If the search is not complete, print number of HE and calculate 
        # ROC if needed
        if iterHardExamples.shape[0] >= (0.05 * nrOfHeFirstIter):
            print "\nHard examples found: " + str(iterHardExamples.shape[0]) + "\n"
            if calculateROC and i < ROCforThisManyFirstIters:
                ROCResults, cost = ROC(trainData, trainClasses, labels, ROCResults)
                ROCResults["iter"].append(i)
                ROCResults["nrOfIterHE"].append(iterHardExamples.shape[0])
        # Search is complete, calculate ROC for the last time if needed and
        # exit the search
        else:
            print "\n|--------------------------------------------------"
            print "| < 5 % hard examples found from the initial amount!"
            print "| Exiting the search..."
            print "|--------------------------------------------------"
            if calculateROC:
                ROCResults, cost = ROC(trainData, trainClasses, labels, ROCResults)
                ROCResults["iter"].append(i)
                ROCResults["nrOfIterHE"].append(iterHardExamples.shape[0])
            break
    
    
    totalHardExamples = np.asarray(totalHardExamples)
    totalHardExampleLabels = np.asarray(totalHardExampleLabels)
            
    return (totalHardExamples, totalHardExampleLabels, ROCResults)






def ROC(trainData, trainClasses, labels, ROCResults):
    
    # Shuffle and maintain the same order in every array
    np.random.seed(666) #222
    shuffledOrder = range(trainData.shape[0])
    np.random.shuffle(shuffledOrder)
    trainData = np.asarray( [trainData[zz,:] for zz in shuffledOrder] )
    trainClasses = np.asarray( [trainClasses[zz] for zz in shuffledOrder] )
    labels = np.asarray( [labels[zz] for zz in shuffledOrder] )
    
    # Find pos & neg indices
    posExRowIndices = np.where(trainClasses==1)[0]
    negExRowIndices = np.where(trainClasses==0)[0]
    
    # Take 75 % for training and 25 % for testing
    forTrainingPos = np.int(0.75 * len(posExRowIndices))
    forTestingPos = len(posExRowIndices) - forTrainingPos
    forTrainingNeg = np.int(0.75 * len(negExRowIndices))
    forTestingNeg = len(negExRowIndices) - forTrainingNeg

    # Partition the data
    trainD, trainC, trainL, testD, testC = [],[],[],[],[]
    testD = np.concatenate((trainData[posExRowIndices[-forTestingPos:]],
                            trainData[negExRowIndices[-forTestingNeg:]]))
    trainD = np.concatenate((trainData[posExRowIndices[0:forTrainingPos]],
                             trainData[negExRowIndices[0:forTrainingNeg]]))
    trainL = np.concatenate((labels[posExRowIndices[0:forTrainingPos]],
                             labels[negExRowIndices[0:forTrainingNeg]]))
    testC = np.concatenate((trainClasses[posExRowIndices[-forTestingPos:]],
                            trainClasses[negExRowIndices[-forTestingNeg:]]))
    trainC = np.concatenate((trainClasses[posExRowIndices[0:forTrainingPos]],
                             trainClasses[negExRowIndices[0:forTrainingNeg]]))
    # Determine best C
    cost = 10.0**(np.arange(-2,3,1))
    model = SVM.Train(trainD, trainC, cost=cost, CVtype="lolo", labels=trainL)
    
    # Calculate ROC with the best C
    fpr, tpr, roc_auc = SVM.ROC(model, trainD, trainC, testD, testC)
    
    # Save results
    ROCResults["FPR"].append(fpr)
    ROCResults["TPR"].append(tpr)
    ROCResults["AUC"].append(roc_auc)
    
    return (ROCResults, model.C)
    