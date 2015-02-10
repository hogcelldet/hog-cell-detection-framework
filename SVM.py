# -*- coding: utf-8 -*-






import numpy as np
import sklearn as sk
import pylab as pl
from sklearn import svm  
from sklearn import cross_validation
from sklearn.cross_validation import LeaveOneLabelOut





"""

This function builds linear SVM classifier.

Inputs and their descriptions:
-------------------------------------------------------------------------------
trainData: 2D array, where m rows are examples and n columns are features
trainClasses: 1D array of m class labels
cost: Can be container (array or list), scalar or empty.
      In case of container: Perform cross validation of cost with the values
                            inside container. Use the best cost value to train
                            classifier.
      In case of scalar: Train classifier with provided cost value.
      In case of empty or non-recognized value: Use default cost value (0.1)
                                                to train classifier
CVtype: Can be integer, string representing integer, "lolo", or empty
        In case of integer or string of integer: Perform integer-fold
                                                 cross-validation
        In case of "lolo": Perform leave one label out cross-validation
                           using labels parameter.
        In case of empty: use 5-fold cross-validation
labels: 1D array of m labels. This parameter will be used if CVtype="lolo"

Output:
-------------------------------------------------------------------------------
Linear SVM classifier

"""

def Train(trainData, trainClasses, cost=None, CVtype=5, labels=None):
    
    # If cost parameter is array or list, perform cross-validation
    # with the range provided in the container
    if isinstance(cost, np.ndarray) or isinstance(cost, list) and len(cost)>0:
    
        print "------------------------------------------------------------"
        print "Searching for the best cost parameter with cross-validation"
        print "------------------------------------------------------------\n"
    
        # Initialize empty array for storing mean scores
        meanScores = np.empty(np.shape(cost))  
        
        # Search best regularization parameter  
        for i, C in enumerate(cost):  
            
            print "Current cost: " + str(C) + ", CV type: " + str(CVtype)
           
            # Initialize classifier
            classifier = svm.SVC(C=C, kernel='linear') 
            
            # Determine CV type
            if CVtype == "lolo":
                cv = LeaveOneLabelOut(labels)
            else:
                cv = np.int(CVtype)
            
            # Perform cross-validation
            scores = cross_validation.cross_val_score \
            (classifier, trainData, trainClasses, cv=cv)
            
            print "Accuracy: %0.3f %% (+/- %0.5f) \n" % \
            (100*scores.mean(), scores.std() / 2)
           
            meanScores[i] = scores.mean()
        
        # Select the best cost
        indexOfBestC = meanScores.argmax()  
        C = cost[indexOfBestC]
        print "Cross validation done, best value of C was:",C
        
        # Build classifier with best C & all the data
        classifier = svm.SVC(C=C, kernel="linear")
        model = classifier.fit(trainData, trainClasses)
        
        print "Score with param cost %.4f for all data: %.2f %% \n" % \
        (C, 100*model.score(trainData,trainClasses))



    # If cost is scalar, train classifier using it
    elif isinstance(cost, np.int) or \
         isinstance(cost, np.float) or \
         isinstance(cost, np.double):

        print "------------------------------------------------------------"
        print "Training linear SVM classifier using provided cost value"
        print "------------------------------------------------------------\n"
        
        # Initialize classifier
        classifier = svm.SVC(C=cost, kernel='linear') 
        
        # Train the classifier
        model = classifier.fit(trainData, trainClasses)
        
        # Calculate score by classifying the same data that 
        # the classifier was trained with
        score = model.score(trainData, trainClasses)
        
        print "Accuracy with cost %.4f for all data: %.3f %%" % \
        (cost, 100*score)



    # If cost is not specified or recognized as container or scalar,
    # default cost value is used to train classifier
    else:
        print "------------------------------------------------------------"
        print "Cost not provided or recognized as container or scalar"
        print "Training linear SVM classifier using default cost (1.0)"
        print "------------------------------------------------------------\n"     
        
        # Initialize classifier
        classifier = svm.SVC(kernel='linear') 
        
        # Train the classifier
        model = classifier.fit(trainData, trainClasses)
        
        # Calculate score by classifying the same data that 
        # the classifier was trained with
        score = model.score(trainData, trainClasses)
        
        print "Accuracy with cost %.4f for all data: %.3f %%" % \
        (model.C, 100*score)
    
    
    
    return model





"""

This function calculates ROC curve and its AUC for SVM.

Linear SVM model is built for given trainData and trainClasses.
The model is tested with given testData and testClasses.

Function returns:
FPR = False Positive Rate
TPR = True Positive Rate
AUC = Area Under Curve

"""

def ROC(model, trainData, trainClasses, testData, testClasses):
        
    # Initialize classifier with best C, all the data & probabilities=True
    classifier = svm.SVC(C=model.C, kernel='linear', probability=True)
    
    # Train the classifier
    model = classifier.fit(trainData, trainClasses)
    
    # Test the classifier
    predictedClasses = model.decision_function(testData)

    # Calculate ROC
    fpr, tpr, thresholds = sk.metrics.roc_curve\
    (testClasses, predictedClasses, pos_label=1)

    # Calculate ROC AUC (Area Under Curve)
    roc_auc = sk.metrics.auc(fpr, tpr)
    
    return (fpr, tpr, roc_auc)





"""

This function visualizes class separability by projecting trainData onto
normal vector w and showing histogram of projected samples.

"""

def VisualizeClassSeparability(model, trainData, trainClasses):
    
    dotProduct = np.dot( trainData, model.coef_.transpose() )
    
    # Add bias so that classes start separating from zero on x-axis 
    dotProduct = np.array([x[0] + model.intercept_[0] for x in dotProduct])
	
    class0 = dotProduct[ np.where(trainClasses==0)[0] ]
    class1 = dotProduct[ np.where(trainClasses==1)[0] ]
    
    pl.figure(figsize=(11,7), facecolor='none') 
    
    pl.hist(class1, bins=50, normed= True, color="#003333", alpha=0.7,
             histtype='stepfilled', label=["Cancer cells"],
             edgecolor="none")
	
    pl.hist(class0, bins=50, normed= True, color="#FF3333", alpha=0.7,
             histtype='stepfilled', label=["Everything else"],
             edgecolor="none")
             
    pl.title("")
    pl.xlabel("Location", fontsize=12)
    pl.ylabel("Frequency", fontsize=12)
    pl.legend(loc="upper right", fontsize=12)
    pl.tight_layout()
    pl.show()
	
    #pl.savefig('classDistribution.png')

    