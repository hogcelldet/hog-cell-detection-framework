# -*- coding: utf-8 -*-

# Libraries
import cv2
import numpy as np
import pylab as pl

# Files
import ImportData
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
    days=["day1","day2","day3"]
    saveAnnotations=False
    thisManySamples="all" # 4858
    trainData, trainClasses, labels, groundTruth = ImportData. \
    ImportDataAndExtractHOGFeatures(hog, days, saveAnnotations, thisManySamples)
    
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
    
    hE, hELabels, ROCResults = HardExamples.Search(hog, trainData, \
    trainClasses, labels, groundTruth, amountToInitialTraining=1.0, \
    saveImagesWithDetections=True, saveHardExampleImages=True,
    maxIters=1000, maxHardExamples=200000, calculateROC=True,
    ROCforThisManyFirstIters=6)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Draw ROC curves
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    pl.close("all")
    font = {'family' : 'sans-serif',
            'weight' : 'normal',
            'size'   : '12'}
    
    pl.rc('font', **font) 
    fig = pl.figure(figsize=(14,7), facecolor='none')
    ax1 = fig.add_subplot(1,2,1)
    ax1.plot([0, 1], [0, 1], 'k--')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.0])
    ax1.set_xlabel('False positive rate (FPR)')
    ax1.set_ylabel('True positive rate (TPR)')   
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Draw area to be zoomed in and lines to zoomed area.
    # This is done before ROC so that ROC curves will be on top of these.
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    startCoords = (0.0,0.9)
    endX = 0.1
    endY = 1.0
    # Small rectangle
    ax1.add_patch(pl.Rectangle(startCoords,endX,endY,facecolor='white',edgecolor='black'))
    
    # Lines going to big rectangle
    xRange = [startCoords[0], 0.3]
    yRange = [startCoords[1], 0.4]
    ax1.plot(xRange, yRange, '#cccccc')
    # Lines going to big rectangle
    xRange = [endX, 0.6]
    yRange = [endY, 0.7]
    ax1.plot(xRange, yRange, '#cccccc')
    
    rect = [0.3,0.4,0.3,0.3]
    ax2 = add_subplot_axes(ax1,rect)
    
    # Hide ticks
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot ROC curve on the left-hand side
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    width = 3
    colors = ["b","g","r","c","m","y","k"]
    font = "serif 14"
    
    for i in range(len(ROCResults["FPR"])):
        
        if i == 0:
            ax1.plot(ROCResults["FPR"][i], ROCResults["TPR"][i],
                    colors[i], linewidth=width,
                    label = "Iteration " + str(ROCResults["iter"][i]) +  \
                    "                                           " + \
                    "AUC %0.3f" % ROCResults["AUC"][i])
        else:
            ax1.plot(ROCResults["FPR"][i], ROCResults["TPR"][i],
                    colors[i], linewidth=width,
                    label = "Iteration " + str(ROCResults["iter"][i]) +  \
                    "  " * (4-len(str(ROCResults["iter"][i]))) +  \
                    "Hard examples " +  str(ROCResults["nrOfIterHE"][i]) + \
                    "  " * (6-len(str(ROCResults["nrOfIterHE"][i]))) + \
                    "AUC %0.3f" % ROCResults["AUC"][i])
                    
                    
        ax1.legend(loc="lower right", fontsize=12)
        ax1.grid()
        
        
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Draw zoomed in area
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    for i in range(len(ROCResults["AUC"])):
        #startIx = np.where(ROCResults["FPR"][i]==ROCResults["FPR"][i].flat[np.abs(ROCResults["FPR"][i] - startCoords[0]).argmin()])[0][-1]
        #endIx = np.where(ROCResults["FPR"][i]==ROCResults["FPR"][i].flat[np.abs(ROCResults["FPR"][i] - endX).argmin()])[0][-1]
        ax2.plot(ROCResults["FPR"][i],
                  ROCResults["TPR"][i],
                  colors[i], linewidth=width)
    ax2.axis([0.0,0.09,0.92,1.0])
    pl.show()
    pl.draw()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot F1-score on the right-hand side
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ax3 = fig.add_subplot(1,2,2)
    # -1 because on last iter there was no detection process
    detectionIters = ROCResults["iter"][-1]-1
    ax3.set_ylim([0.0, 1.0])
    ax3.set_xlim([2, detectionIters+1])
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('F1-score')
    meanF1 = np.zeros(detectionIters)
    index = 0
    testedImages = 10 # 6 from day 1, 2 from day 2, 2 from day 3
    for i in np.arange(1,len(ROCResults["F1"]),1):
        if i % testedImages == 0:
            index += 1
        meanF1[index] += ROCResults["F1"][i-1] # -1 because i starts from 1

    for i in range(len(meanF1)):
        meanF1[i] /= testedImages
    for res in meanF1:
        # Plot with the same color as 
        ax3.plot(np.arange(2,detectionIters+2),meanF1,
                 color="black", lw=3)
    
    ax3.grid()
    pl.draw()
    
