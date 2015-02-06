# -*- coding: utf-8 -*-





# Libraries
import os
import glob
import time
import numpy as np
import cv2
import scipy.io 
import ConfigParser
from PIL import Image, ImageDraw

# Files
import MeasurePerformance
import Filters





"""

Returns a list of paths to subfolders

"""

def Listdirs(folder):
    return [
        d for d in (os.path.join(folder, d1) for d1 in os.listdir(folder))
        if os.path.isdir(d)
    ] 





"""

This function takes care of looping through image(s).

"""
def OneOrMultipleImageSearch(hog, model, searchMethod, defaultParams, 
                             paramsAndTheirRangesToBeVaried, evaluate,
                             outputFolder, saveIm,
                             checkForBreakCondition1=False,
                             checkForBreakCondition2=False,
                             saveDetectionBinaryImages=False):
    
    # Set SVM weights as detector.
    # Bias will be added to hitThreshold parameter.
    w = model.coef_[0]  
    hog.setSVMDetector( w )
    
    # Create output folder
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)   
    
    if evaluate[0] == "singleImage":
        
        # Open image
        imFileName = evaluate[1]
        img = cv2.imread(imFileName, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        
        searchResultsForSingleIm = OneOrTwoParameterSearch(hog, model, 
                                                searchMethod,
                                                defaultParams,
                                                paramsAndTheirRangesToBeVaried,
                                                outputFolder,
                                                imFileName, img, saveIm,
                                                checkForBreakCondition1,
                                                checkForBreakCondition2,
                                                saveDetectionBinaryImages)
             
        return searchResultsForSingleIm

    elif evaluate[0] == "folder":
        
        totalResults = []
        
        # Input folder location and possible image types
        listOfDayDirs = Listdirs(evaluate[1])
        fileTypes = ["bmp", 'jpg', 'png']
        
        # Loop through folders (days)
        for day,directory in enumerate(listOfDayDirs):
            
            # Get list of images in the folder
            imageList = []
            for fileType in fileTypes:
                imageList = imageList+glob.glob(directory + "\*." + fileType)
            
            folderResults = []
            
            # Loop through images
            for i,imFileName in enumerate(imageList):
                
                print "\nProcessing " + directory[directory.rfind("\\")+1:]+\
                " image " + str(i+1) + "/" + str(len(imageList)) + "..."
                print "----------------------------"
                
                # Open image
                img = cv2.imread(imFileName, cv2.CV_LOAD_IMAGE_GRAYSCALE)
                
                
                searchResultsForSingleIm = OneOrTwoParameterSearch(hog, model, 
                                               searchMethod, defaultParams,
                                               paramsAndTheirRangesToBeVaried,
                                               outputFolder,
                                               imFileName, img, saveIm,
                                               checkForBreakCondition1,
                                               checkForBreakCondition2,
                                               saveDetectionBinaryImages)
                                               
                folderResults.append(searchResultsForSingleIm)            
                    
            totalResults.append(folderResults)        

        return totalResults





"""

This function takes care of looping through values of given parameter(s).

"""

def OneOrTwoParameterSearch(hog, model, searchMethod, defaultParams,
                            paramsAndTheirRangesToBeVaried, outputFolder,
                            imFileName, img, saveIm,
                            checkForBreakCondition1,
                            checkForBreakCondition2,
                            saveDetectionBinaryImages=False):
                                
    # Initialize dictionary of lists where results will be saved
    res                       = {}
    res["imName"]             = [] # Name of image
    res["nrOfCellsTruth"]     = [] # True number of cells
    res["TP"]                 = [] # Number of True Positives
    res["FP"]                 = [] # Number of False Positives
    res["FN"]                 = [] # Number of False Negatives
    res["FPR"]                = [] # False Positive Rate
    res["TPR"]                = [] # True Positive Rate
    res["F1"]                 = [] # F1 score
    res["F05"]                = [] # F05 score
    res["F09"]                = [] # F09 score
    res["timeTaken"]          = [] # Time taken to detect cells
    res["variedParam1AndVal"] = [] # Parameter which is varied
    res["variedParam2AndVal"] = [] # Parameter which stays steady (grid-search)
    res["w"]                  = [] # Confidence for detections
    res["allParams"]          = [] # All detection parameters
    
    # If no parameter values to be searched,
    # one only wants to use predefined values
    if len(paramsAndTheirRangesToBeVaried) == 0:
        
        res,stopSearching = ConductSingleSearch(hog, model, res, [],
                                  searchMethod, defaultParams,
                                  [], [],
                                  outputFolder,
                                  imFileName, img, saveIm,
                                  checkForBreakCondition1,
                                  checkForBreakCondition2,
                                  saveDetectionBinaryImages)

    # If one parameter values are searched
    elif len(paramsAndTheirRangesToBeVaried) == 2:   
        
        # Vary detection parameter and save results
        for i in range(len(paramsAndTheirRangesToBeVaried[1])):
    
            print "\nCurrent " + paramsAndTheirRangesToBeVaried[0] + \
                  ":", paramsAndTheirRangesToBeVaried[1][i]
                  
            variedParam1AndVal = [paramsAndTheirRangesToBeVaried[0],
                                 paramsAndTheirRangesToBeVaried[1][i]]
            variedParam2AndVal = []
            
            res,stopSearching = ConductSingleSearch(hog, model, res, i,
                                      searchMethod, defaultParams,
                                      variedParam1AndVal, variedParam2AndVal,
                                      outputFolder,
                                      imFileName, img, saveIm,
                                      checkForBreakCondition1,
                                      checkForBreakCondition2)
                                      
            if stopSearching:
                break
        
    
    # If two parameter values are searched (grid-search)
    elif len(paramsAndTheirRangesToBeVaried) == 4:
        
        for i in range(len(paramsAndTheirRangesToBeVaried[3])):
            
            variedParam2AndVal = [paramsAndTheirRangesToBeVaried[2],
                                 paramsAndTheirRangesToBeVaried[3][i]]
            
            print "\nSetting " + variedParam2AndVal[0] + \
            " to " + str(variedParam2AndVal[1]) +  \
            " and looping through " + \
            paramsAndTheirRangesToBeVaried[0]
            
            # Vary detection parameter and save results
            for j in range(len(paramsAndTheirRangesToBeVaried[1])):
                
                variedParam1AndVal = [paramsAndTheirRangesToBeVaried[0],
                                     paramsAndTheirRangesToBeVaried[1][j]]
        
                print "\nCurrent " + variedParam1AndVal[0] + \
                      ":", variedParam1AndVal[1]
                
                res, stopSearching = ConductSingleSearch(hog, model, res, j,
                                          searchMethod, defaultParams,
                                          variedParam1AndVal, variedParam2AndVal,
                                          outputFolder,
                                          imFileName, img, saveIm,
                                          checkForBreakCondition1,
                                          checkForBreakCondition2)
                                          
                if stopSearching:
                    break
                
    else:
        print "!!!"
        print """!!! Warning: Invalid paramsAndTheirRangesToBeVaried parameter
              value. It must have length of 0, 2 or 4."""
        
    # Return the results in either of the cases
    return res





"""

This function is responsible for single detection task with given parameters.

"""

def ConductSingleSearch(hog, model, res, i,
                        searchMethod, defaultParams,
                        variedParam1AndVal, variedParam2AndVal,
                        outputFolder,
                        imFileName, img, saveIm,
                        checkForBreakCondition1,
                        checkForBreakCondition2,
                        saveDetectionBinaryImages=False):
    
    stopSearching = False
    
    
    # If parameter1 is to be varied, set its value in default parameter values
    if variedParam1AndVal != []:                     
        hog, defaultParams = SetSearchParams(hog, model,
                                             defaultParams,
                                             searchMethod,
                                             variedParam1AndVal)
    
    # If parameter2 is to be varied, set its value in default parameter values
    if variedParam2AndVal != []:
        
        hog, defaultParams = SetSearchParams(hog, model,
                                             defaultParams,
                                             searchMethod,
                                             variedParam2AndVal)
                                             
    
    # Generate settings string to be used in file name
    if searchMethod == "detectMultiScale":
        
        settings = "_iter_"    + str(i) + \
                   "_nlev_"    + str(hog.nlevels) + \
                   "_hT_"      + str(defaultParams["hitThreshold"]) + \
                   "_wS_"      + str(defaultParams["winStride"]) + \
                   "_padding_" + str(defaultParams["padding"]) + \
                   "_scale_"   + str(defaultParams["scale"]) + \
                   "_fT_"      + str(defaultParams["finalThreshold"]) + \
                   "_mS_"      + str(defaultParams["useMeanshiftGrouping"])
    else:
        settings = "_iter_"         + str(i) + \
                   "_hitThreshold_" + str(defaultParams["hitThreshold"]) + \
                   "_winStride_"    + str(defaultParams["winStride"])

    # Run sliding window procedure
    found, timeTaken, w = SlidingWindow(hog, img, searchMethod, 
                          defaultParams, filterDetections=False)
                          
    # Image name withouth path or file extension
    imName = imFileName[imFileName.rfind("\\")+1:imFileName.rfind(".")]
    
    if saveDetectionBinaryImages:
        # Create new black image where detections are marked
        im = Image.new("L", [img.shape[1],img.shape[0]], "black") 
        #im = np.zeros((img.shape))
        for x, y, w, h in found:
            # Mark detections with white
            draw = ImageDraw.Draw(im)
            draw.rectangle([x, y, x+w, y+h],  fill=255)
        # Determine the image day
        day = imFileName[imFileName[:imFileName.rfind("\\")].rfind("\\")+4:
        imFileName.rfind("\\")]
        # Save classification image
        estimDir = outputFolder + "\\detectionBinaries"
        childOfEstimDir = estimDir + "\\" + day +  "\\"
        if not os.path.exists(childOfEstimDir):
            os.makedirs(childOfEstimDir)
        scipy.misc.imsave(childOfEstimDir + imName + "_day" + \
        day + ".png", im)
    
    # Save the results in .INI
    pathToDetectionsIni = outputFolder+"\\"+imName+settings+".ini"
    SaveIni(found, pathToDetectionsIni, searchMethod, imFileName, hog)
    
    # Determine ground truth .INI file name
    pathToTruthIni = imFileName[:imFileName.rfind(".")]+"_annotations.ini"
    
    # Analyze the results, build confusion matrix
    TP, FP, FN, TPR, FPR, F1, F05, F09, nrOfCellsTruth, imWithDetections = \
    MeasurePerformance.Measure \
    (pathToDetectionsIni, pathToTruthIni, imFileName)
    
    # Save image with annotations
    if saveIm:
        whichDay = imFileName[:imFileName.rfind("\\")][-4:]
        scipy.misc.imsave \
        (outputFolder + "\\" + imName + settings + "_" + whichDay + \
        ".png", imWithDetections)

    # Save results
    res["imName"].append(imFileName)
    res["nrOfCellsTruth"].append(nrOfCellsTruth)
    res["TP"].append(TP)
    res["FP"].append(FP)
    res["FN"].append(FN)
    res["FPR"].append(FPR)
    res["TPR"].append(TPR)
    res["F1"].append(F1)
    res["F05"].append(F05)
    res["F09"].append(F09)
    res["timeTaken"].append(timeTaken)
    res["variedParam1AndVal"].append(variedParam1AndVal)
    res["variedParam2AndVal"].append(variedParam2AndVal)
    res["w"].append(w)
    res["allParams"].append(defaultParams)
    
    if checkForBreakCondition1:
        if TPR >= 1.0:
            print "Ending the search because TPR >= 1.0"
            stopSearching = True
        elif FPR >= 1.0:
            print "Ending the search because FPR >= 1.0"
            stopSearching = True
        elif len(res["F1"]) > 3:
            # Make sure that we have gone beyond max F1
            if np.max(res["F1"]) > 0.6:
                if (res["F1"][-1] < res["F1"][-3]) and (res["F1"][-1] <= 0.6):
                    print "\nEnding the search because 3 iterations " + \
                    "in a row number F1-score has been decreasing and " + \
                    "F1-score is now <= 0.6"
                    stopSearching = True
        elif len(res["F1"]) > 3:
            if res["F1"][-1] == 0.0 and \
               res["F1"][-2] == 0.0 and \
               res["F1"][-3] == 0.0:
                print "\nEnding the search because 3 iterations " + \
                "in a row number F1-score has been 0.0"
                stopSearching = True
    
    if checkForBreakCondition2:
        nrOfIters = 4
        if len(res["FN"])>nrOfIters and \
           res["FN"][-(nrOfIters+1)] < res["FN"][-nrOfIters] and \
           res["FN"][-nrOfIters] < res["FN"][-(nrOfIters-1)] and \
           res["FN"][-(nrOfIters-1)] < res["FN"][-(nrOfIters-2)]:
            print "\nEnding the search because " + str(nrOfIters) + \
            " iterations " + \
            "in a row number of false negatives has been increasing."
            # Delete last three results from every variable
            for key in res.keys():
                res[key] = res[key][:-nrOfIters]
            stopSearching = True
    
    return (res, stopSearching)





"""

This function initializes parameters correctly for hog.detect or 
hog.detectMultiScalefor function.

"""

def SetSearchParams(hog, model, defaultParams,
                    searchMethod,setThisParamAndVal):
    
    parameterName, parameterValue = setThisParamAndVal
    
    # In case of hitThreshold being set, add value to bias.
    # hitThreshold parameter is included in both searchMethods.
    # In scikit-learn's SVM, bias is -model.intercept_[0]
    if parameterName == "hitThreshold":
        defaultParams[parameterName] = -model.intercept_[0]+parameterValue
    
    # Set parameters for detectMultiScale -function
    elif searchMethod == "detectMultiScale":
        
        # Nlevels parameter has to be changed when initializing 
        # HOG class instance
        if parameterName == "nlevels":
            myParams = dict(
                            _winSize           = hog.winSize,
                            _blockSize         = hog.blockSize,
                            _blockStride       = hog.blockStride,
                            _cellSize          = hog.cellSize,
                            _nbins             = hog.nbins,
                            _derivAperture     = hog.derivAperture,
                            _winSigma          = hog.winSigma,
                            _histogramNormType = hog.histogramNormType,
                            _L2HysThreshold    = hog.L2HysThreshold,
                            _gammaCorrection   = hog.gammaCorrection,
                            _nlevels           = parameterValue
                            )

            hog = cv2.HOGDescriptor(**myParams)
            
            w = model.coef_[0]
            hog.setSVMDetector( w )
			
        # Other parameters can be changed without initializing new HOG
        # class instance (like it has to initialized with nlevels)
        else:
            defaultParams[parameterName] = parameterValue
    
    # Set parameters for detect -function
    elif searchMethod == "detect":
        
        # Nlevels parameter has to be changed when initializing 
        # HOG class instance
        if parameterName == "nlevels":
            myParams = dict(
                            _winSize           = hog.winSize,
                            _blockSize         = hog.blockSize,
                            _blockStride       = hog.blockStride,
                            _cellSize          = hog.cellSize,
                            _nbins             = hog.nbins,
                            _derivAperture     = hog.derivAperture,
                            _winSigma          = hog.winSigma,
                            _histogramNormType = hog.histogramNormType,
                            _L2HysThreshold    = hog.L2HysThreshold,
                            _gammaCorrection   = hog.gammaCorrection,
                            _nlevels           = parameterValue
                            )
            hog = cv2.HOGDescriptor(**myParams)
            w = model.coef_[0]
            hog.setSVMDetector( w )
            
        else:
            defaultParams[parameterName] = parameterValue
                   
    return (hog, defaultParams)





"""

This function saves detections to .ini file

Inputs:
found = List of detections returned by hog.detect or hog.detectMultiScale.
pathToDetectionsIni = String describing the path and filename of the ini file.
searchMethod = "detect" or "detectMultiScale", specifying the function type,
                which produced detections.
imFileName = String of image name from which objects were detected.
             Will be usedin .ini section names.
hog = OpenCV hogDescriptor.

"""

def SaveIni(found, pathToDetectionsIni, searchMethod, imFileName, hog):
    
    # Create empty file & open it
    config = ConfigParser.RawConfigParser()
    if not os.path.isfile(pathToDetectionsIni):
        # If you receive IOError: [Errno 2] here, it most probably
        # means you have too long file name
        open(pathToDetectionsIni, "a").close()
        config.read(pathToDetectionsIni)
    
    # Use index in secName instead of datetime because datetime is 
    # too inaccurate/slow to generate unique names for sections.
    index = 1
    
    for detection in found:
        # Both detectMultiScale & detect return
        # upper left corner coordinates of detections
        ulc = detection[0]
        ulr = detection[1]
        # detectMultiScale returns also width & height of detections
        if searchMethod == "detectMultiScale":
            w,h = detection[2:4]
        # detect does not return width & height -->
        # set them the same as winSize
        elif searchMethod == "detect":
            w, h = hog.winSize
        # Name of section
        secName = os.path.basename(imFileName) + \
        "_positiveExample_" + str(index)
        # Create section
        if not config.has_section(secName):
            config.add_section(secName)
        # Set section keys (properties)
        config.set(secName, "date", time.asctime())
        config.set(secName, "ulc", ulc)
        config.set(secName, "ulr", ulr)
        config.set(secName, "lrc", ulc+w)
        config.set(secName, "lrr", ulr+h)
        index += 1
    
    # Write result to file
    with open(pathToDetectionsIni, "w") as outfile:
        config.write(outfile)




     
"""

This function runs the sliding window procedure and filters out detections.

Inputs:
hog = OpenCV HOGDescriptor class instance.
img = Source image. Preferably opened with OpenCV to ensure compatibility.
searchMethod = "detect" or "detectMultiScale", specifying function type.
params = Dictionary of function parameters.
filterDetections = Boolean, specifying whether to filter detections or not.

Outputs:
detections, computationTime, detectionWeights

"""

def SlidingWindow(hog, img, searchMethod, params, filterDetections):    
    
    # Start the clock
    start_time = time.time()
    
    if searchMethod == "detectMultiScale":
        found, w = hog.detectMultiScale(img, **params)
    
    elif searchMethod == "detect":
        found, w = hog.detect(img, **params)
    
    else:
        print "!!!"
        print "!!! Error: searchMethod argument not recognized!"
        print "!!!"
        return
    
    # Stop the clock
    timeTaken = time.time()-start_time # In seconds
    print "Time taken: %i min %.1f seconds" % (np.floor((timeTaken)/60), \
    timeTaken-(60*np.floor((timeTaken)/60)))
    
    print "Nr of initial detections: %i" % len(found)
    
    if (filterDetections):
        foundFiltered1 = []
        foundFiltered2 = []
        #print "Filtering detections (inside each other & too much overlap)..."
        
        # Filter detections inside each other
        for ri, r in enumerate(found):
            for qi, q in enumerate(found):
                # Filter out detection which has detection inside
                if ri != qi and Filters.Inside(q,r,[]):
                    # Delete also its weight
                    np.delete(w, ri)
                    break
                # Pass a detection through this filter if it is does not
                # have any detections inside 
                elif (qi == len(found)-1):
                    foundFiltered1.append(r)
        #print "len(foundFiltered1): %i" % len(foundFiltered1)
        
        # Filter overlapping detections            
        for ri, r in enumerate(foundFiltered1):
            for qi, q in enumerate(foundFiltered1):
                # Filter out detection which overlap too much
                if ri != qi and Filters.Overlap(r, q, [], None) > 0.8:
                    # Delete also its weight
                    np.delete(w, ri)
                    break
                # Pass a detection through this filter if it is not
                # overlapping too much with any of the other detections 
                elif (qi == len(foundFiltered1)-1):
                    foundFiltered2.append(r)   
        #print "len(foundFiltered2): %i" % len(foundFiltered2)
        
        print "Nr of detections after filtering: %i" % len(foundFiltered2)
        return (foundFiltered2, timeTaken, w)
        
    else:
        return (found, timeTaken, w)


