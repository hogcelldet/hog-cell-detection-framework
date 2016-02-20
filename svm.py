# -*- coding: utf-8 -*-


import numpy as np
import pylab as pl
from sklearn import svm
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import LeaveOneLabelOut


"""

This function builds linear SVM classifier.

Inputs and their descriptions:
-------------------------------------------------------------------------------
train_data: 2D array, where m rows are examples and n columns are features
train_classes: 1D array of m class labels
cost: Can be container (array or list), scalar or empty.
      In case of container: Perform cross-validation of cost with the values
                            inside container. Use the best cost value to train
                            classifier.
      In case of scalar: Train classifier with provided cost value.
      In case of empty or non-recognized value: Use default cost value (0.1)
                                                to train classifier
cv_type: Can be integer, string representing integer, "lolo", or empty
         In case of integer or string of integer: Perform integer-fold
                                                  cross-validation
         In case of "lolo": Perform leave one label out cross-validation
                            using labels parameter.
         In case of empty: use 5-fold cross-validation
labels: 1D array of m labels. This parameter will be used if cv_type="lolo"

Output:
-------------------------------------------------------------------------------
Linear SVM classifier

"""


def train(train_data, train_classes, cost=None, cv_type=5, labels=None):

    # If cost parameter is array or list, perform cross-validation
    # with the range provided in the container
    if (isinstance(cost, np.ndarray) or
            isinstance(cost, list) and len(cost) > 0):
    
        print "------------------------------------------------------------"
        print "Searching for the best cost parameter with cross-validation"
        print "------------------------------------------------------------\n"

        # Determine CV type
        if cv_type == "lolo":
            cv = LeaveOneLabelOut(labels)
        else:
            cv = np.int(cv_type)

        grid = GridSearchCV(
            estimator=svm.SVC(
                kernel='linear',
                class_weight='auto',
                probability=False,
                random_state=123,
            ),
            param_grid={"C": cost}, cv=cv,
            scoring='roc_auc',
            n_jobs=-1, pre_dispatch='2*n_jobs',
            refit=True, verbose=1
        )
        grid.fit(train_data, train_classes)
        print("      # ROC AUC of the best estimator on the validation data:" +
              " %0.3f %%" % (grid.best_score_ * 100))
        print "      # " + \
              "Cost which gave the best results on the validation data:"
        for param, val in grid.best_params_.items():
            print "         - %s : %s" % (param, val)
        model = grid.best_estimator_

    # If cost is scalar, train classifier using it
    elif (isinstance(cost, np.int) or
          isinstance(cost, np.float) or
          isinstance(cost, np.double)):

        print "------------------------------------------------------------"
        print "Training linear SVM with cost = %s" % cost
        print "------------------------------------------------------------\n"
        
        # Initialize classifier
        classifier = svm.SVC(C=cost, kernel='linear', probability=False,
                             class_weight='auto', random_state=123)

        # Train the classifier
        model = classifier.fit(train_data, train_classes)

        # Calculate ROC AUC by classifying the same data that
        # the classifier was trained with
        roc_auc = roc_auc_score(classifier.predict(train_data), train_classes)
        
        print "ROC AUC with cost %.4s for all data: %.3f %%" % \
              (cost, 100*roc_auc)

    # If cost is not specified or recognized as container or scalar,
    # default cost value is used to train classifier
    else:
        print "------------------------------------------------------------"
        print "Cost not provided or recognized as container or scalar"
        print "Training linear SVM classifier using default cost (1.0)"
        print "------------------------------------------------------------\n"
        # Initialize classifier
        classifier = svm.SVC(C=1.0, kernel='linear', probability=False,
                             class_weight='auto', random_state=123)
        
        # Train the classifier
        model = classifier.fit(train_data, train_classes)
        
        # Calculate ROC AUC by classifying the same data that
        # the classifier was trained with
        roc_auc = roc_auc_score(classifier.predict(train_data), train_classes)
        
        print "ROC AUC with cost %.4s for all data: %.3f %%" % \
              (cost, 100*roc_auc)
    
    return model


"""

This function calculates ROC curve and its AUC for SVM.

Linear SVM model is built for given train_data and train_classes.
The model is tested with given test_data and test_classes.

Function returns:
FPR = False Positive Rate
TPR = True Positive Rate
AUC = Area Under Curve

"""


def roc(model, train_data, train_classes, test_data, test_classes):
        
    # Initialize classifier with best C, all the data & probabilities=True
    classifier = svm.SVC(C=model.C, kernel='linear', probability=True,
                         class_weight='auto', random_state=123)
    
    # Train the classifier
    classifier.fit(train_data, train_classes)
    
    # Test the classifier
    predicted_classes = classifier.decision_function(test_data)

    # Calculate ROC
    fpr, tpr, thresholds = roc_curve(test_classes, predicted_classes,
                                     pos_label=1)

    # Calculate ROC AUC (Area Under Curve)
    roc_auc = auc(fpr, tpr)
    
    return fpr, tpr, roc_auc


"""

This function visualizes class separability by projecting train_data onto
normal vector w and showing histogram of projected samples.

"""


def visualize_class_separability(model, train_data, train_classes):
    
    dot_product = np.dot(train_data, model.coef_.transpose())
    
    # Add bias so that classes start separating from zero on x-axis 
    dot_product = np.array([x[0] + model.intercept_[0] for x in dot_product])

    class0 = dot_product[np.where(train_classes == 0)[0]]
    class1 = dot_product[np.where(train_classes == 1)[0]]
    
    pl.figure(figsize=(11, 7), facecolor='none')
    
    pl.hist(class1, bins=50, normed=True, color="#003333", alpha=0.7,
            histtype='stepfilled', label=["Cancer cells"], edgecolor="none")

    pl.hist(class0, bins=50, normed=True, color="#FF3333", alpha=0.7,
            histtype='stepfilled', label=["Everything else"], edgecolor="none")
             
    pl.title("")
    pl.xlabel("Location", fontsize=12)
    pl.ylabel("Frequency", fontsize=12)
    pl.legend(loc="upper right", fontsize=12)
    pl.tight_layout()
    pl.show()
    # pl.savefig('class_distribution.png')
