# -*- coding: utf-8 -*-


# Libraries
import numpy as np
import cv2
import ConfigParser

# Files
import Filters


# ----------------------------------------------------------------

def VisualizeConfusionMatrix(img, detected, typeOfDetection):
    
    nrOf = 0
    thickness = 2
    
    for i,detection in enumerate(detected):
        
        if len(detection)==6 and detection[5] != typeOfDetection:
            continue
        
        if typeOfDetection == "truePositive":
            x, y, w, h = detection[4]
            colouring = (0, 255, 0)
            
        elif typeOfDetection == "falsePositive":
            x, y, w, h = detection[4]
            colouring = (255, 0, 0)
            
        elif typeOfDetection == "falseNegative":
            x, y, w, h = detection[2]
            colouring = (0, 0, 255)
            
        x, y, w, h = int(x), int(y), int(w), int(h)
        cv2.rectangle(img, (x, y), (w, h), color=colouring, thickness=thickness)
        
        # Draw running number inside each detection
        #cv2.putText(img, str(nrOf), (x+5,y+15), cv2.FONT_HERSHEY_SIMPLEX,
        #fontScale=0.42, color=colouring, thickness=thickness)
        
        nrOf += 1 
            
# ----------------------------------------------------------------

def AlreadyAdded(sectionName, results, whatKindOf):
    
    if not results: # List is empty
        return False

    if whatKindOf == "detection":
        for res in results:
            if res[3] == sectionName:
                return True
    elif whatKindOf == "annotation":
        for res in results:
            if res[1] == sectionName:
                return True
    
    return False
    
# ----------------------------------------------------------------

def Measure(pathToDetectionsIni, pathToTruthIni, imPath):
    
    # PAS metric threshold for considering detections as TP
    PASThresholdForTP = 0.3
    
    results = []    
    
    nrOfTP = 0
    nrOfFP = 0
    nrOfFN = 0
    
    configDetections = ConfigParser.RawConfigParser()
    configDetections.read(pathToDetectionsIni)    

    configTruth = ConfigParser.RawConfigParser()
    configTruth.read(pathToTruthIni)
    
    nrOfPosAnnotations = np.where(np.asarray(["positive" in configName for \
    configName in configTruth.sections()])==True)[0].shape[0]
    # Calculate overlap of each neighboring detection and annotation.
    # Store each comparison in a tuple and place it in a list.
    # List is chosen because it can be sorted in the next step.
    potentialMatches = []
     
    for section2 in configDetections.sections():       
        ulc2 = np.double(configDetections.get(section2, "ulc"))
        ulr2 = np.double(configDetections.get(section2, "ulr"))
        lrc2 = np.double(configDetections.get(section2, "lrc"))
        lrr2 = np.double(configDetections.get(section2, "lrr"))
        detectionCoords = (ulc2, ulr2, lrc2, lrr2)
            
        detCenter = ((ulc2+lrc2)/2,(ulr2+lrr2)/2)
        
        for i,section1 in enumerate(configTruth.sections()):
            
            if "positive" not in section1:
                continue
            
            ulc1 = np.double(configTruth.get(section1, "ulc"))
            ulr1 = np.double(configTruth.get(section1, "ulr"))
            lrc1 = np.double(configTruth.get(section1, "lrc"))
            lrr1 = np.double(configTruth.get(section1, "lrr"))
            refCoords = (ulc1, ulr1, lrc1, lrr1)
            
            refCenter = ((ulc1+lrc1)/2,(ulr1+lrr1)/2)
            
            # If detected rectangle center too far away from reference
            # rectangle center, do not calculate overlap and mark the
            # detection as false positive.
            # This check makes this nested loop much faster to compute.
            if np.abs(refCenter[0]-detCenter[0]) > 200 or \
               np.abs(refCenter[1]-detCenter[1]) > 200:         
                continue
            
            oL = Filters.Overlap(refCoords, detectionCoords, \
            hog=None, structureOfData="corners")

            # Tuple
            comparison = (oL,               # overlap
                          section1,         # truthName
                          refCoords,        # truthCoord
                          section2,         # detectionName
                          detectionCoords)  # detectionCoords
            
            potentialMatches.append(comparison)
        
        if not AlreadyAdded(section2, potentialMatches, "detection"):
            falsePos = (0.0,               # overlap
                        "",                # truthName
                        [],                # truthCoord
                        section2,          # detectionName
                        detectionCoords,   # detectionCoords
                        "falsePositive")   # hitType
            nrOfFP += 1
            results.append(falsePos)

    # Sort list of comparisons according to the amount of overlap
    # First element of the list will have the most overlap
    potentialMatches = sorted(potentialMatches,key=lambda k: k[0],reverse=True)
    
    # Preserve only the best unique detections
    for comparison in potentialMatches:
    
        if AlreadyAdded(comparison[3], results, "detection"):
            continue
        
        # Only one match for each annotation
        if comparison[0] >= PASThresholdForTP and \
            not AlreadyAdded(comparison[1], results, "annotation"): 
            comparison += ("truePositive",) # Append tuple
            results.append(comparison)
            nrOfTP += 1
            
        else:
            fp = (comparison[0],
                  "",
                  [],
                  comparison[3],
                  comparison[4],
                  "falsePositive") # Append tuple
            results.append(fp)
            nrOfFP += 1
            
    # Add false negatives: those annotations which are not already
    # added in the results
    for section in configTruth.sections():
        
        if "positive" not in section:
            continue
        
        if not AlreadyAdded(section, results, "annotation"):
            ulc1 = np.double(configTruth.get(section, "ulc"))
            ulr1 = np.double(configTruth.get(section, "ulr"))
            lrc1 = np.double(configTruth.get(section, "lrc"))
            lrr1 = np.double(configTruth.get(section, "lrr"))
            refCoords = (ulc1, ulr1, lrc1, lrr1)
            
            falseNeg = (0.0,               # overlap
                        section,           # truthName
                        refCoords,         # truthCoord
                        "",                # detectionName
                        [],                # detectionCoords
                        "falseNegative")   # hitType
            nrOfFN += 1
            results.append(falseNeg)
    
    # Open current image
    img = cv2.imread(imPath)
    height, width, depth = img.shape
    
    VisualizeConfusionMatrix(img, results, "truePositive")
    VisualizeConfusionMatrix(img, results, "falsePositive")
    VisualizeConfusionMatrix(img, results, "falseNegative")
        
    print "True positives  =", nrOfTP
    print "false positives =", nrOfFP
    print "false negatives =", nrOfFN

    # True Positive Rate
    if nrOfTP == 0 and nrOfFN == 0:
        TPR = 1.0
    else:
        TPR = np.double(nrOfTP) / np.double(nrOfTP+nrOfFN)
        
    # False Positive Rate   
    FPR = np.double(nrOfFP) / np.double(nrOfPosAnnotations)
    
    # Precision
    if np.double(nrOfTP+nrOfFP) > 0.0:
        precision = np.double(nrOfTP) / np.double(nrOfTP+nrOfFP)
    else:
        precision = 0.0
    
    # F1 score
    # Do not divide by zero
    if np.double(nrOfTP+nrOfFP+nrOfFN) > 0.0:
        F1 = 2.0*nrOfTP/(2.0*nrOfTP+nrOfFP+nrOfFN)
        print "F1 score = %.2f" % F1
    else:
        F1 = 0.0
        print "F1 score = 0"
        
    # F05 score
    if np.double(precision+TPR) > 0.0:
        F05 = np.double((0.5**2.0 + 1.0) * precision * TPR) / np.double(precision+TPR)
        print "F05 score = %.2f" % F05
    else: 
        F05 = 0
        print "F05 score = 0"
        
    # F09 score
    if np.double(precision+TPR) > 0.0:
        F09 = np.double((0.9**2.0 + 1.0) * precision * TPR) / np.double(precision+TPR)
        print "F09 score = %.2f" % F09
    else: 
        F09 = 0
        print "F09 score = 0"

    if nrOfTP + nrOfFN != nrOfPosAnnotations:
        print "\n!!!"
        print "!!! Error: different amount of annotations and matched detections!"
        print "!!! TP: %i, FN: %i, TP + FN: %i" % (nrOfTP, nrOfFN, nrOfTP+nrOfFN)
        print "!!! Nr of annotations:", nrOfPosAnnotations
        print "!!!\n"
    if nrOfTP + nrOfFP != len(configDetections.sections()):
        print "!!!"
        print "!!! Error: different amount of initial detections and matched detections!"
        print "!!! TP: %i, FP: %i, TP + FP: %i" % (nrOfTP, nrOfFP, nrOfTP+nrOfFP)
        print "!!! Nr of initial detection:", len(configDetections.sections())
        print "!!!"
    
    return (nrOfTP, nrOfFP, nrOfFN, TPR, FPR, F1, F05, F09, nrOfPosAnnotations, img)
