# -*- coding: utf-8 -*-

# Libraries
import cv2
import numpy as np
import pylab as pl
import datetime
import time

# Files
import DetectionProcess
import ImportData
import SVM

if __name__ == '__main__':
    
    parameterToBeVaried = "nbins"    
    
    defaultHOG = cv2.HOGDescriptor()
        
    if parameterToBeVaried == "blockSize":
        #valueRange = np.arange(4, 33, 4)
        valueRange =  2**(np.arange(2,6))
        # Default custom HOG parameters
        myParams = dict(
                _winSize           = (32,32),
                _cellSize          = (4,4),
                _nbins             = 9,
                _derivAperture     = defaultHOG.derivAperture,
                _winSigma          = defaultHOG.winSigma,
                _histogramNormType = defaultHOG.histogramNormType,
                _L2HysThreshold    = defaultHOG.L2HysThreshold,
                _gammaCorrection   = defaultHOG.gammaCorrection,
                _nlevels           = defaultHOG.nlevels
            )
    elif parameterToBeVaried == "cellSize":
        valueRange =  2**(np.arange(1,6))
        # Default custom HOG parameters
        myParams = dict(
                _winSize           = (32,32),
                _blockSize         = (32,32),
                _blockStride       = (16,16),
                _nbins             = 9,
                _derivAperture     = defaultHOG.derivAperture,
                _winSigma          = defaultHOG.winSigma,
                _histogramNormType = defaultHOG.histogramNormType,
                _L2HysThreshold    = defaultHOG.L2HysThreshold,
                _gammaCorrection   = defaultHOG.gammaCorrection,
                _nlevels           = defaultHOG.nlevels
            )
    elif parameterToBeVaried == "nbins":
        valueRange = 2**(np.arange(1,6))
        myParams = dict(
                _winSize           = (32,32),
                _blockSize         = (16,16),
                _blockStride       = (8,8),
                _cellSize          = (4,4),
                _derivAperture     = defaultHOG.derivAperture,
                _winSigma          = defaultHOG.winSigma,
                _histogramNormType = defaultHOG.histogramNormType,
                _L2HysThreshold    = defaultHOG.L2HysThreshold,
                _gammaCorrection   = defaultHOG.gammaCorrection,
                _nlevels           = 1
            )
            
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build classifiers with different values and try finding objects with them
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            
    results = []
    
    for value in valueRange:
        if parameterToBeVaried == "winSize":
            myParams["_winSize"]     = (value, value)
            myParams["_blockSize"]   = (value/4, value/4)
            myParams["_blockStride"] = (value/8, value/8)
            myParams["_cellSize"]    = (value/8, value/8)
        elif parameterToBeVaried == "blockSize":
            myParams["_blockSize"]   = (value, value)
            myParams["_blockStride"] = (value/2, value/2)
        elif parameterToBeVaried == "cellSize":
            myParams["_cellSize"] = (value, value)
        elif parameterToBeVaried == "nbins":
            myParams["_nbins"] = value
        
        hog = cv2.HOGDescriptor(**myParams)
                     
        # Import data and extract HOG features
        trainData, trainClasses, labels, groundTruth = ImportData. \
        ImportDataAndExtractHOGFeatures(hog=hog,
                                        days=["day1","day2","day3"],
                                        saveAnnotations=False,
                                        thisManySamples=100)
                                        
        # Build classifier with cross-validating cost              
        #cost = 10.0**(np.arange(-2,3,1))
        cost = 0.01
        model = SVM.Train(trainData, trainClasses, cost=cost)
        
        
        
        
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
        evaluate = ["singleImage", r".\testWithThese\day3\Tile002496.bmp"]            
                        
        outputFolder = r".\\" + searchMethod + "_" + datetime.datetime. \
                       fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S') 
                       
        parametersToStudy = []
                               
        searchResults = DetectionProcess.OneOrMultipleImageSearch(hog, model, 
                        searchMethod, params, 
                        [], evaluate, outputFolder, saveIm=True)
                          
        results.append(searchResults)
        
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Analyze reults
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    if evaluate[0] == "singleImage":
        pl.close("all")
        fig = pl.figure(figsize=(6, 6), facecolor='white')
        ax1 = fig.add_subplot(1,1,1)
        ax1.set_ylim(0,1)
        ax1.set_xlabel("Varied HOG param")
        ax1.set_ylabel("F1-score")
        for ix,res in enumerate(results):
            ax1.scatter(ix,res["F1"])
            pl.draw()
    


        
