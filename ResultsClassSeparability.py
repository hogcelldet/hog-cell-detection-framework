# -*- coding: utf-8 -*-


# Libraries
import cv2
import numpy as np
import pylab as pl

# Files
import ImportData
import SVM
    

if __name__ == '__main__':
    
    parameterToBeVaried = "Cost"    
    
    # Initialize dictionary of lists where resutls will be saved
    ROCResults = {}
    ROCResults["FPR"]               = [] # False Positive Rate
    ROCResults["TPR"]               = [] # True Positive Rate
    ROCResults["AUC"]               = [] # Area Under ROC
    ROCResults["fvl"]               = [] # feature vector length
    ROCResults["models"]            = []
    ROCResults[parameterToBeVaried] = []
    
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
            _nlevels           = defaultHOG.nlevels
            )
            
    hog = cv2.HOGDescriptor(**myParams)
    
    # Import data and extract HOG features
    trainData, trainClasses, labels, groundTruth = ImportData. \
    ImportDataAndExtractHOGFeatures(hog=hog,
                                    days=["day1","day2","day3"],
                                    saveAnnotations=False,
                                    thisManySamples=4000)
                                    
    # Shuffle and maintain the same order in every array
    np.random.seed(222)
    shuffledOrder = range(trainData.shape[0])
    np.random.shuffle(shuffledOrder)
    trainData = np.asarray( [trainData[i,:] for i in shuffledOrder] )
    trainClasses = np.asarray( [trainClasses[i] for i in shuffledOrder] )
    labels = np.asarray( [labels[i] for i in shuffledOrder] )

                                 
    # Divide into training and testing data
    forTesting = 2000
    forTraining = 2000
    trainD, trainC, testD, testC = [],[],[],[]
    
    posExRowIndices = np.where(trainClasses==1)[0]
    negExRowIndices = np.where(trainClasses==0)[0]
    testD = np.concatenate((trainData[posExRowIndices[-forTesting:]],
                            trainData[negExRowIndices[-forTesting:]]))
    trainD = np.concatenate((trainData[posExRowIndices[0:forTraining]],
                             trainData[negExRowIndices[0:forTraining]]))
    trainL = np.concatenate((labels[posExRowIndices[0:forTraining]],
                             labels[negExRowIndices[0:forTraining]]))
    testC = np.concatenate((trainClasses[posExRowIndices[-forTesting:]],
                            trainClasses[negExRowIndices[-forTesting:]]))
    trainC = np.concatenate((trainClasses[posExRowIndices[0:forTraining]],
                             trainClasses[negExRowIndices[0:forTraining]]))
                                    

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # C-V C
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    cost = 10.0**(np.arange(-4,5,1))
    model = SVM.Train(trainD, trainC, cost=cost, CVtype="lolo", labels=trainL)
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Visualize
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    pl.close("all")
    fig = pl.figure(figsize=(13,6), facecolor='none')
    ax1 = fig.add_subplot(1,2,1)#, adjustable='box', aspect=1.0) # ROC
    ax2 = fig.add_subplot(1,2,2)#, adjustable='box', aspect=1.0) # HOG parameters table
    ax2.axis('off')
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Visalize class separability
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    
    dotProduct = np.dot( testD, model.coef_.transpose() )
    dotProduct = np.array([x[0] + model.intercept_[0] for x in dotProduct]) # add bias
        
    class0 = dotProduct[ np.where(testC==0)[0] ]
    class1 = dotProduct[ np.where(testC==1)[0] ]
    
    
    ax1.hist(class1, bins=50, normed= True, color="#003333", alpha=0.7,
         histtype='stepfilled', label=["Cancer cells"],
         edgecolor="none")

    ax1.hist(class0, bins=50, normed= True, color="#FF3333", alpha=0.7,
             histtype='stepfilled', label=["Everything else"],
             edgecolor="none")
             

             
    ax1.set_xlabel("Location", fontsize=12)
    ax1.set_ylabel("Frequency", fontsize=12)
    
    ax1.legend(loc="upper right", fontsize=12)
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
    
    #if parameterToBeVaried == "blockStride":
        #pl.subplots_adjust(wspace = 0.18)
    #else:
        #pl.subplots_adjust(wspace = 0.22)
        


    pl.savefig(r".\resultsBestSVMCost" + \
             "_wSi" + str(myParams["_winSize"])+\
             "_bSi"+str(myParams["_blockSize"])+\
             "_bSt"+str(myParams["_blockStride"])+\
             "_cS"+str(myParams["_cellSize"])+\
             "_nbins"+str(myParams["_nbins"])+\
             ".pdf")
             
                                                     


