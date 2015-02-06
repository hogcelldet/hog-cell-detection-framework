# -*- coding: utf-8 -*-


# Libraries
import cv2
import numpy as np
import pylab as pl

# Files
import ImportData
import SVM
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
        
    # Initialize dictionary of lists where resutls will be saved
    ROCResults = {}
    ROCResults["FPR"]         = [] # False Positive Rate
    ROCResults["TPR"]         = [] # True Positive Rate
    ROCResults["AUC"]         = [] # Area Under ROC
    ROCResults["fvl"]         = [] # feature vector length
    ROCResults["forTraining"] = []
    ROCResults["nrOfNe"]      = []
    ROCResults["nrOfHe"]      = []
    
    defaultHOG = cv2.HOGDescriptor()
    
    myParams = dict(
            _winSize           = (32,32),
            _blockSize         = (8,8),
            _blockStride       = (4,4),
            _cellSize          = (4,4),
            _nbins             = 2,
            _derivAperture     = defaultHOG.derivAperture,
            _winSigma          = defaultHOG.winSigma,
            _histogramNormType = defaultHOG.histogramNormType,
            _L2HysThreshold    = defaultHOG.L2HysThreshold,
            _gammaCorrection   = defaultHOG.gammaCorrection,
            _nlevels           = 1
        )
        
    hog = cv2.HOGDescriptor(**myParams)
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Import twice as many samples as you are going to use
    # for training at maximum
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    # Import data and extract HOG features
    trainData, trainClasses, labels, groundTruth = ImportData. \
    ImportDataAndExtractHOGFeatures(hog=hog,
                                    days=["day1","day2","day3"],
                                    saveAnnotations=False,
                                    thisManySamples=4000)
                                    
    # Shuffle and maintain the same order in every array
    np.random.seed(123)
    shuffledOrder = range(trainData.shape[0])
    np.random.shuffle(shuffledOrder)
    trainData = np.asarray( [trainData[i,:] for i in shuffledOrder] )
    trainClasses = np.asarray( [trainClasses[i] for i in shuffledOrder] )
    labels = np.asarray( [labels[i] for i in shuffledOrder] )

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Find hard examples
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    hardExamples, hardExampleLabels, ROCResults2 = HardExamples.Search(hog, trainData, \
    trainClasses, labels, groundTruth, amountToInitialTraining=0.1, \
    saveImagesWithDetections=True, saveHardExampleImages=True)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Divide data into train data and test data
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    #random.seed(768)
    #random.shuffle(hardExamples)

    forTesting = 2000
    trainD, trainC, testD, testC = [],[],[],[]
    
    posExRowIndices = np.where(trainClasses==1)[0]
    negExRowIndices = np.where(trainClasses==0)[0]
    testD = np.concatenate((trainData[posExRowIndices[-forTesting:]],
                            trainData[negExRowIndices[-forTesting:]]))
    testC = np.concatenate((trainClasses[posExRowIndices[-forTesting:]],
                            trainClasses[negExRowIndices[-forTesting:]]))
                            
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build classifiers with different amount of training data
    # and calculate their ROC
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    hardExamplesPortionOfNegatives = 0.25
    for i,forTraining in enumerate([10,100,500,1000, 2000]): 
        
        nrOfHe = np.int(  hardExamplesPortionOfNegatives      * forTraining )
        nrOfNe = np.int( (1.0-hardExamplesPortionOfNegatives) * forTraining )
        
        trainD = np.concatenate((trainData[posExRowIndices[0:forTraining]],
                                 trainData[negExRowIndices[0:nrOfNe]],
                                 hardExamples[0:nrOfHe]))
                                 
        trainC = np.concatenate((trainClasses[posExRowIndices[0:forTraining]],
                                 trainClasses[negExRowIndices[0:nrOfNe]],
                                 ([0] * nrOfHe) ))
                                 
        trainL = np.concatenate((labels[posExRowIndices[0:forTraining]],
                                 labels[negExRowIndices[0:nrOfNe]],
                                 hardExampleLabels[0:nrOfHe] ))
        
        # Build classifier with cross-validating cost              
        cost = 10.0**(np.arange(-4,5,1))
        model = SVM.Train(trainD, trainC, cost=cost, CVtype="lolo", labels=trainL)
        
        # Calculate ROC
        fpr, tpr, roc_auc = SVM.ROC(model, trainD, trainC, testD, testC)
        
        ROCResults["FPR"].append(fpr)
        ROCResults["TPR"].append(tpr)
        ROCResults["AUC"].append(roc_auc)
        ROCResults["fvl"].append(trainData.shape[1])
        ROCResults["forTraining"].append(str(forTraining))
        ROCResults["nrOfNe"].append(str(nrOfNe))
        ROCResults["nrOfHe"].append(str(nrOfHe))
        
        
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Visualize
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    pl.close("all")
    fig = pl.figure(figsize=(13,6), facecolor='none')
    ax1 = fig.add_subplot(1,2,1, adjustable='box', aspect=1.0) # ROC
    ax2 = fig.add_subplot(1,2,2, adjustable='box', aspect=1.0) # HOG parameters table
    ax2.axis('off')
    
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
    ax11 = add_subplot_axes(ax1,rect)
    
    # Hide ticks
    ax11.get_xaxis().set_visible(False)
    ax11.get_yaxis().set_visible(False)
        
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot ROC curve
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
    width = 3
    #lineStyles = ["b-","g--","r-.","c:","m"]
    lineStyles = ["b","g","r","c","m","y"]
    for i in range(len(ROCResults["AUC"])):
        ax1.plot(ROCResults["FPR"][i], ROCResults["TPR"][i], lineStyles[i], linewidth=width, label= \
        ROCResults["forTraining"][i] + "+  "  + "  " * (4-len(ROCResults["forTraining"][i])) + \
        ROCResults["nrOfNe"][i]      + "-  "  + "  " * (4-len(ROCResults["nrOfNe"][i]))      + \
        ROCResults["nrOfHe"][i]      + "-- "  + "  " * (4-len(ROCResults["nrOfHe"][i]))      + \
        "AUC %0.3f" % ROCResults["AUC"][i])
        ax1.legend(loc="lower right", fontsize=12)
        
    ax1.plot([0, 1], [0, 1], 'k--')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.0])
    ax1.set_xlabel('False positive rate (FPR)')
    ax1.set_ylabel('True positive rate (TPR)')
    #pl.tight_layout()
    pl.show()
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Draw zoomed in area
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    for i in range(len(ROCResults["AUC"])):
        #startIx = np.where(ROCResults["FPR"][i]==ROCResults["FPR"][i].flat[np.abs(ROCResults["FPR"][i] - startCoords[0]).argmin()])[0][-1]
        #endIx = np.where(ROCResults["FPR"][i]==ROCResults["FPR"][i].flat[np.abs(ROCResults["FPR"][i] - endX).argmin()])[0][-1]
        ax11.plot(ROCResults["FPR"][i],
                  ROCResults["TPR"][i],
                  lineStyles[i], linewidth=width)
        ax11.axis([0.0,0.09,0.92,1.0])
    pl.show()
    pl.draw()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Draw HOG parameters table
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    row_labels = ['winSize',
                  'blockSize',
                  'blockStride',
                  'cellSize',
                  'nbins']
            
    table_vals = [[ myParams["_winSize"]     ],
                  [ myParams["_blockSize"]   ],
                  [ myParams["_blockStride"] ],
                  [ myParams["_cellSize"]    ],
                  [ myParams["_nbins"]       ]]

    the_table = ax2.table(cellText=table_vals,
                         colWidths = [0.1]*3,
                         rowLabels=row_labels,
                         loc='lower left')
    the_table.set_fontsize(12)
    the_table.scale(2, 2)
    #ax2.text(-0.194,0.37,'HOG parameters',size=14)
    #ax2.text(-0.122,0.37,'HOG parameters',size=14)
    
    pl.subplots_adjust(wspace = 0.22)
    pl.show()
    pl.draw()
    pl.savefig(r".\compareEffectsOfAmountOfTrainingData7_wSi"+str(myParams["_winSize"])+\
                                                  "_bSi"+str(myParams["_blockSize"])+\
                                                  "_bSt"+str(myParams["_blockStride"])+\
                                                  "_cS"+str(myParams["_cellSize"])+\
                                                  "_nbins"+str(myParams["_nbins"])+\
                                                  ".pdf")