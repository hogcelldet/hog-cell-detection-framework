# -*- coding: utf-8 -*-

"""
Imports training data and generates features

Arguments:
hog = OpenCV HOG class instance.
days = List of strings of folder names indicating that negative annotations
       can be automatically collected from the images inside these folders.
saveAnnotations = Boolean value determining if user wants to save all the
                  annotations as images to the hard drive after resizing.
thisManySamples = Integer or string.
                  If integer: This many positive examples and 
                              this many negative examples will be imported.
                  If string (e.g. "all"): All examples are imported.
"""

import cv2
import os
import glob
import numpy as np  
import ConfigParser
import uuid
import time
import scipy.io 

# ----------------------------------------------------------------

# Clears the screen
def Cls():
    os.system(['clear','cls'][os.name == 'nt'])

# ----------------------------------------------------------------
 
# Returns a list of paths to subfolders
def Listdirs(folder):
    return [
        d for d in (os.path.join(folder, d1) for d1 in os.listdir(folder))
        if os.path.isdir(d)
    ] 

# ----------------------------------------------------------------

def ImportDataAndExtractHOGFeatures(hog, days, saveAnnotations, thisManySamples):
    
    winSizeAspectRatio = hog.winSize[0]/hog.winSize[1]
    
    timeOfExecution = str(time.time())
    timeOfExecution = timeOfExecution[:timeOfExecution.rfind(".")]
    
    Cls() # Clear screen
    totalNrOfPos = 0
    totalNrOfNeg = 0
    annotationWidths  = []
    annotationHeights = []
    aspectRatios = []
    trainingExamples  = {} # Dictionary for features
    # Structure:
    # Key: Dir (string)
    #  Key: positiveExamples
    #  Key: negativeExamples
    groundTruth       = {} # Dictionary for annotation coordinates
    # Structure:
    # Key: Dir
    #  Key: Image
    #   Key: positiveExamples
    #   Key: negativeExamples

    print "Importing annotations:"
    print "------------------------------------------------------------"    
    
    # Input folder location and possible image types
    listOfDayDirs = Listdirs(r".\trainWithThese")
    fileTypes = ["bmp", "jpg", "png"]
    
    # Loop through input folders (days)
    for k,directory in enumerate(listOfDayDirs):
        
        # Initialize new dictionaries
        trainingExamples[directory] = {}
        groundTruth[directory]      = {}
        # Initialize new lists
        trainingExamples[directory]["positiveExamples"] = []
        trainingExamples[directory]["negativeExamples"] = []
        
        # Get list of images in the folder
        imageList = []
        for fileType in fileTypes:
            imageList = imageList + glob.glob(directory +  "\*." + fileType)
        
        # Loop through images
        for i,imFileName in enumerate(imageList):
            
            # Image name without file type extension
            imName = imFileName[imFileName.rfind("\\")+1 : \
            imFileName.rfind(".")]           
            
            # Initialize new dictionaries
            groundTruth[directory][imName] = {}     
            # Initialize new lists
            groundTruth[directory][imName]["positiveExamples"] = []
            groundTruth[directory][imName]["negativeExamples"] = []
            
            print "\n" + imFileName + ":"

            # Open current image
            img = cv2.imread(imFileName, 0); 
            height, width = img.shape

            # Path to image's annotation file
            pathToAnnotationFile = os.path.splitext(imFileName)[0] + \
            "_annotations.ini"
            # If file does not exist --> skip to next image
            if not os.path.isfile(pathToAnnotationFile):
                continue
            # If file does exist --> open it
            else:
                config = ConfigParser.RawConfigParser()
                config.read(pathToAnnotationFile)  
            
            # Initialize
            positiveExamples = []
            negativeExamples = []
        
            # Create new black image where annotations are marked in white
            # so that negative examples can be gathered from the rest
            # of the locations
            markedAnnotations = np.zeros((img.shape))
            
            # -----------------------------------------------
            # Gather annotated positive and negative examples
            # -----------------------------------------------
            
            # Go through every annotation and extract it from original image
            for section in config.sections():
                
                # Get coordinates
                ulc = int(float(config.get(section, "ulc")))
                ulr = int(float(config.get(section, "ulr")))
                lrc = int(float(config.get(section, "lrc")))
                lrr = int(float(config.get(section, "lrr")))
                
                # Make sure the section options is not not NaN or INF
                if np.isnan(ulc) or np.isnan(ulr) or \
                   np.isnan(lrc) or np.isnan(lrr):
                    break
                
                annHeight = lrr-ulr
                annWidth = lrc-ulc
                
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Crop and resize annotation
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                annotationAspectRatio = (annHeight/annWidth)
                                         
                # If aspect ratio is correct, simply crop image as it is
                if annotationAspectRatio == winSizeAspectRatio:
                    annotation = img[ulr:lrr, ulc:lrc]
                    annotation = cv2.resize(annotation, hog.winSize)
                
                # If more width than height --> select height accordingly
                # Aspect ratio is preserved
                elif annotationAspectRatio < winSizeAspectRatio:
                    extendThisMuch = (winSizeAspectRatio * annWidth)/2
                    # If no room to extend height, resize and lose aspect ratio
                    if ulr < np.floor(extendThisMuch) or lrr > height-np.ceil(extendThisMuch):
                        annotation = img[ulr:lrr, ulc:lrc]
                        annotation = cv2.resize(annotation, hog.winSize)
                    # If there is room to extend, then extend height
                    else:
                        annotation = img[ulr-np.int(np.floor(extendThisMuch)):lrr+np.int(np.ceil(extendThisMuch)), ulc:lrc]
                        annotation = cv2.resize(annotation, hog.winSize)
                # If more height than width --> select width accordingly
                # Aspect ratio is preserved
                elif annHeight > annWidth:
                    extendThisMuch = (winSizeAspectRatio * annHeight)/2
                    # If no room to extend width, resize and lose aspect ratio
                    if ulc < np.floor(extendThisMuch) or lrc > width-np.ceil(extendThisMuch):
                        annotation = img[ulr:lrr, ulc:lrc]
                        annotation = cv2.resize(annotation, hog.winSize)
                    # If there is room to extend, then extend width
                    else:
                        annotation = img[ulr:lrr, ulc-np.int(np.floor(extendThisMuch)):lrc+np.int(np.ceil(extendThisMuch))]
                        annotation = cv2.resize(annotation, hog.winSize)
                        
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Generate feature
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    
                feature = hog.compute(annotation)[:,0]

                # Save the feature
                if "positiveExample" in section:
                    
                    # Save size & aspect ratio
                    annotationHeights.append(annHeight)
                    annotationWidths.append(annWidth)
                    aspectRatios.append(annotationAspectRatio)
                    
                    positiveExamples.append(feature)
                    
                    # Save coordinates to dictionary
                    groundTruth[directory][imName]["positiveExamples"].append \
                    ([ulc,ulr,lrc-ulc,lrr-ulr])
                    
                    if (saveAnnotations):
                        
                        folder = r".\annotations_"+timeOfExecution+"\\positive" 
                        # Create the folder if it does not exist already
                        if not os.path.exists(folder):
                            os.makedirs(folder)            
                        
                        # Save the image      
                        cv2.imwrite(folder + "\\" + \
                        directory[directory.rfind("\\")+1 :] + "_" + \
                        imName + "_" + \
                        str(uuid.uuid4().fields[-1]) + ".bmp", annotation)

                elif "negativeExample" in section:
                    
                    negativeExamples.append(feature)
                    
                    # Save coordinates to dictionary
                    groundTruth[directory][imName]["negativeExamples"].append \
                    ([ulc,ulr,lrc-ulc,lrr-ulr])
                    
                    if (saveAnnotations):
                        
                        folder = r".\annotations_"+timeOfExecution+"\\negative"
                        # Create the folder if it does not exist already
                        if not os.path.exists(folder):
                            os.makedirs(folder)     
                            
                        # Save the image                
                        cv2.imwrite(folder + "\\" + \
                        directory[directory.rfind("\\")+1 :] + "_" + \
                        imName + "_" + \
                        str(uuid.uuid4().fields[-1]) + ".bmp", annotation)
                
                else:
                    print "WARNING: Section without positive/negative identifier"
       
                # Mark handled annotation with white
                markedAnnotations[ulr:lrr, ulc:lrc] = 1
                
            # ---------------------------------------------------------------
            # Gather more negative examples automatically from the background
            # ---------------------------------------------------------------
            
            # Gather only from images from these days, because on subsequent
            # day images do not have enough background space anyway and they
            # are not fully annotated, which could cause some of positive
            # examples ending up being marked as negative example.
            if any(day in directory for day in days):
                
                # Loop through image with step size = winSize
                for y in xrange(hog.winSize[0],width-hog.winSize[0],hog.winSize[0]+10):
                    for x in xrange(hog.winSize[1],height-hog.winSize[1],hog.winSize[1]+10):
                        
                        # Crop area from binary image
                        currentArea = markedAnnotations[x:x+hog.winSize[1], \
                        y:y+hog.winSize[0]]
                        
                        # Check if currentArea is completely black,
                        # i.e. it is background
                        if np.sum(currentArea) == 0:
                            
                            # Crop the same area from original image
                            cropped = img[x:x+hog.winSize[1], y:y+hog.winSize[0]]

                            # Extract feature
                            feature = hog.compute(cropped)
                            # Save the feature
                            negativeExamples.append(feature)
                            
                            # Save coordinates to dictionary
                            groundTruth[directory][imName]["negativeExamples"]\
                            .append([x,y,hog.winSize[1],hog.winSize[0]])
                            
                            if (saveAnnotations):
                                folder = r".\annotations_"+timeOfExecution+ \
                                "\\negative"
                                # Create the folder if it does not exist already
                                if not os.path.exists(folder):
                                    os.makedirs(folder)            
                                # Save the image                
                                scipy.misc.imsave(folder + "\\" + \
                                directory[directory.rfind("\\")+1 :] + "_" + \
                                imName + "_" + \
                                str(uuid.uuid4().fields[-1]) + \
                                ".bmp", cropped)        
                        else:
                            continue
                        
            trainingExamples[directory]["positiveExamples"].append(positiveExamples)
            trainingExamples[directory]["negativeExamples"].append(negativeExamples)

            print "Positive examples gathered from this image: %i" % \
            len(positiveExamples)
            print "Negative examples gathered from this image: %i" % \
            len(negativeExamples) + "\n"
            
            totalNrOfPos += len(positiveExamples)
            totalNrOfNeg += len(negativeExamples)
    
    print "------------------------------------------------------------" 
    print "Mean pos annotation width: %.2f +- %.2f px"%(np.mean(annotationWidths),np.std(annotationWidths))
    print "Mean pos annotation height: %.2f +- %.2f px"%(np.mean(annotationHeights),np.std(annotationHeights))
    print "Mean pos aspect ratio: %.2f +- %.2f" % (np.mean(aspectRatios), np.std(aspectRatios))
    print "Total number of positive annotations: %i" % totalNrOfPos
    print "Total number of negative annotations: %i" % totalNrOfNeg
    print "Length of feature vector: ", len(feature)
    print "------------------------------------------------------------\n" 

    #print "\nMerging the data, assigning classes, and assigning labels..."
    trainData = []
    trainClasses = []
    labels = []
    label = 0
    for directory in trainingExamples.keys():
        
        for examples in trainingExamples[directory]["positiveExamples"]:
            trainData = trainData + examples 
            trainClasses = trainClasses + ([1] * len(examples))
            labels = labels + ([label] * len(examples))
            
        for examples in trainingExamples[directory]["negativeExamples"]:
            trainData = trainData + examples   
            trainClasses = trainClasses + ([0] * len(examples))
            labels = labels + ([label] * len(examples))
            
        label = label + 1  

    trainData    = np.asarray(trainData)
    trainClasses = np.asarray(trainClasses)
    labels       = np.asarray(labels)

    if type(thisManySamples) == int:
        # Take equal amount of positives and negatives
        np.random.seed(54321)
        # Shuffle negatives and maintain the same order in each array
        negExRowIndices = np.where(trainClasses==0)[0]
        np.random.shuffle(negExRowIndices)
        deleteThese = negExRowIndices[thisManySamples:]
        trainData = np.delete(trainData, deleteThese, 0)
        trainClasses = np.delete(trainClasses, deleteThese, 0)
        labels = np.delete(labels, deleteThese, 0)
        # Shuffle positives and maintain the same order in each array
        posExRowIndices = np.where(trainClasses==1)[0]
        np.random.shuffle(posExRowIndices)
        deleteThese = posExRowIndices[thisManySamples:]
        trainData = np.delete(trainData, deleteThese, 0)
        trainClasses = np.delete(trainClasses, deleteThese, 0)
        labels = np.delete(labels, deleteThese, 0)
        
    return (trainData, trainClasses, labels, groundTruth)

