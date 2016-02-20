# -*- coding: utf-8 -*-


import os
import time
import glob
import uuid
import datetime

import scipy
import cv2
import numpy as np

import svm
import utils
import filters
import DetectionProcess
import MeasurePerformance


def search(hog, train_data, train_classes, labels,
           ground_truth, amount_to_initial_training=1.0,
           save_images_with_detections=False, save_hard_example_images=True,
           max_iters=1000, calculate_roc=False,
           roc_for_this_many_first_iters=5):

    # Initialize dictionary of lists where results will be saved
    roc_results = {"FPR": [], "TPR": [], "AUC": [], "iter": [],
                   "nrOfIterHE": [], "F1": []}

    total_hard_examples = []
    total_hard_example_labels = []

    # Find hard examples using smaller amount of data.
    # Else, use all the data.
    if amount_to_initial_training != 1.0:
        neg_ex_row_indices = np.where(train_classes == 0)[0]
        pos_ex_row_indices = np.where(train_classes == 1)[0]

        neg_ind = neg_ex_row_indices[
                   :int(amount_to_initial_training * len(neg_ex_row_indices))]
        pos_ind = pos_ex_row_indices[
                   :int(amount_to_initial_training * len(pos_ex_row_indices))]

        train_data = np.concatenate(
            (train_data[neg_ind], train_data[pos_ind]))
        train_classes = np.concatenate(
            (train_classes[neg_ind], train_classes[pos_ind]))
        labels = np.concatenate((labels[neg_ind], labels[pos_ind]))

    if save_hard_example_images or save_images_with_detections:
        # Output folder name
        parent_folder = ("hardExamples_" +
                         datetime.datetime.fromtimestamp(
                             time.time()).strftime('%Y-%m-%d_%H-%M-%S'))
        # Create parent output folder if does not exist yet
        if not os.path.exists(parent_folder):
            os.makedirs(parent_folder)

    if calculate_roc:
        roc_results, cost = roc(train_data, train_classes, labels, roc_results)
        roc_results["iter"].append(1)  # First iteration
        roc_results["nrOfIterHE"].append(0)  # Zero hard examples

    for i in np.arange(2, max_iters, 1):

        iter_hard_examples = []
        iter_hard_example_labels = []

        # Search and build SVM model.
        # If ROC was calculated last on last iteration, we already have
        # cross-validated cost.
        if calculate_roc and i <= roc_for_this_many_first_iters:
            model = svm.train(train_data, train_classes, cost=cost)
        else:  # Else, cross-validate new cost value
            cost = 10.0 ** (np.arange(-2, 3, 1))
            model = svm.train(train_data, train_classes,
                              cost=cost, cv_type="lolo", labels=labels)

        # Use the model to detect cells from already seen images
        # and compare them to ground truth in order to find false positives
        w = model.coef_[0]
        hog.setSVMDetector(w)

        sliding_window_method = "detectMultiScale"

        params = dict(
            hitThreshold=-model.intercept_[0],
            winStride=(2, 2),
            # IMPORTANT! if same as blockStride, no detections will be produced
            padding=(0, 0),
            # IMPORTANT! if bigger than (0,0),
            # detections can have negative values which cropping does not like
            scale=1.05,
            finalThreshold=2,
            useMeanshiftGrouping=False
        )

        # Input folder location and possible image types
        day_dirs = utils.list_dirs(r".\trainWithThese")
        file_types = ["bmp", 'jpg', 'png']

        # Loop through input folders (days)
        for ii, directory in enumerate(day_dirs):

            # Search hard examples only from images from these days,
            # because subsequent day images are not annotated 100% correctly
            # and thus "false positives" might be actual positives
            if ("day1" not in directory and
                    "day2" not in directory and
                    "day3" not in directory):
                continue

            # Get list of images in the folder
            images = []
            for file_type in file_types:
                images = images + glob.glob(directory + "\*." + file_type)

            # Loop through images
            for j, im_file_name in enumerate(images):

                # Image name without file type extension
                im_name = im_file_name[im_file_name.rfind("\\") + 1:
                                       im_file_name.rfind(".")]

                print ("\nProcessing " +
                       directory[directory.rfind("\\") + 1:] +
                       " image " + str(j + 1) + "/" + str(len(images)) + "...")

                # Open current image
                im = cv2.imread(im_file_name, cv2.CV_LOAD_IMAGE_GRAYSCALE)

                # Second copy that remains untouched for cropping
                im_orig = cv2.imread(im_file_name, cv2.CV_LOAD_IMAGE_GRAYSCALE)

                # Detect
                found, time_taken, w = DetectionProcess.sliding_window(
                    hog, im, sliding_window_method, params,
                    filter_detections=False)

                if save_hard_example_images:
                    # Create the folder if it does not exist already
                    child_folder = (parent_folder + "\\" +
                                    "hardExamples_iter" + str(i))
                    if not os.path.exists(child_folder):
                        os.makedirs(child_folder)

                # Find hard examples
                for ri, r in enumerate(found):
                    for qi, q in enumerate(ground_truth[directory][im_name][
                                               "positiveExamples"]):

                        if filters.get_overlap(r, q) > 0.0000001:
                            break

                        # This example is false positive if it overlaps
                        # less than 0.0000001 % with any of the true positives
                        elif (qi == len(ground_truth[directory][im_name][
                                            "positiveExamples"]) - 1):

                            # You can set minimum weight/confidence threshold
                            # for hard examples here.
                            if w[ri] <= 0.0:
                                continue

                            cropped = im_orig[r[1]:r[1] + r[3],
                                              r[0]:r[0] + r[2]]

                            # Crop & resize
                            cropped = cv2.resize(cropped, hog.winSize)

                            # Generate feature
                            feature = hog.compute(cropped)[:, 0]

                            iter_hard_examples.append(feature)
                            iter_hard_example_labels.append(ii)

                            # Save the image
                            if save_hard_example_images:
                                cv2.imwrite(
                                    child_folder + "\\" + im_name + "_" +
                                    str(uuid.uuid4().fields[-1])[:5] +
                                    ".png", cropped)

                # Save the results in .INI.
                # Create the folder where at least detections.INI files will 
                # be saved. Images with detections will be saved over there
                # as well later, if input argument of this function says so.
                child_folder = (parent_folder + "\\" +
                                directory[directory.rfind(
                                    "\\") + 1:] + "_imagesWithDetections")
                # Create the folder if it does not exist already
                if not os.path.exists(child_folder):
                    os.makedirs(child_folder)

                path_to_detections_ini = (child_folder + "\\" + im_name +
                                          "_iter" + str(i) + ".ini")

                DetectionProcess.save_ini(found, path_to_detections_ini,
                                          sliding_window_method, im_file_name,
                                          hog)

                # Analyze the results, build confusion matrix
                path_to_truth_ini = (im_file_name[:im_file_name.rfind(".")] +
                                     "_annotations.ini")

                tp, fp, fn, tpr, fpr, f1, f05, f09, n_cells_truth, \
                    im_with_detections = MeasurePerformance.Measure(
                        path_to_detections_ini, path_to_truth_ini,
                        im_file_name)

                roc_results["F1"].append(f1)

                if save_images_with_detections:
                    # Save the image with detections
                    scipy.misc.imsave(
                        child_folder + "\\" +
                        im_file_name[im_file_name.rfind("\\") + 1:
                                     im_file_name.rfind(".")] + "_iter" +
                        str(i) + ".png", im_with_detections)

        # If no hard examples were found, draw ROC for the last time and exit
        if len(iter_hard_examples) == 0:
            roc_results, cost = roc(train_data, train_classes, labels,
                                    roc_results)
            roc_results["iter"].append(i)
            roc_results["nrOfIterHE"].append(0)
            break

        # Concatenate
        total_hard_examples = total_hard_examples + iter_hard_examples
        total_hard_example_labels = (total_hard_example_labels +
                                     iter_hard_example_labels)

        # List to array
        iter_hard_example_labels = np.asarray(iter_hard_example_labels)
        iter_hard_examples = np.asarray(iter_hard_examples)

        # Append
        train_data = np.concatenate((train_data, iter_hard_examples))
        train_classes = np.concatenate(
            (train_classes, ([0] * iter_hard_examples.shape[0])))
        labels = np.concatenate((labels, iter_hard_example_labels))

        # Save the number of hard examples on first iteration
        if i == 2:
            n_he_first_iter = iter_hard_examples.shape[0]

        # If the search is not complete, print number of HE and calculate 
        # ROC if needed
        if iter_hard_examples.shape[0] >= (0.05 * n_he_first_iter):
            print "\nHard examples found: " + str(
                iter_hard_examples.shape[0]) + "\n"
            if calculate_roc and i < roc_for_this_many_first_iters:
                roc_results, cost = roc(train_data, train_classes, labels,
                                        roc_results)
                roc_results["iter"].append(i)
                roc_results["nrOfIterHE"].append(iter_hard_examples.shape[0])
        # Search is complete, calculate ROC for the last time if needed and
        # exit the search
        else:
            print "\n|--------------------------------------------------"
            print "| < 5 % hard examples found from the initial amount!"
            print "| Exiting the search..."
            print "|--------------------------------------------------"
            if calculate_roc:
                roc_results, cost = roc(train_data, train_classes, labels,
                                        roc_results)
                roc_results["iter"].append(i)
                roc_results["nrOfIterHE"].append(iter_hard_examples.shape[0])
            break

    total_hard_examples = np.asarray(total_hard_examples)
    total_hard_example_labels = np.asarray(total_hard_example_labels)

    return total_hard_examples, total_hard_example_labels, roc_results


def roc(train_data, train_classes, labels, roc_results):
    # Shuffle and maintain the same order in every array
    np.random.seed(222)
    shuffled_order = range(train_data.shape[0])
    np.random.shuffle(shuffled_order)
    train_data = np.asarray([train_data[zz, :] for zz in shuffled_order])
    train_classes = np.asarray([train_classes[zz] for zz in shuffled_order])
    labels = np.asarray([labels[zz] for zz in shuffled_order])

    # Find pos & neg indices
    pos_ex_row_indices = np.where(train_classes == 1)[0]
    neg_ex_row_indices = np.where(train_classes == 0)[0]

    # Take 75 % for training and 25 % for testing
    for_training_pos = np.int(0.75 * len(pos_ex_row_indices))
    for_training_neg = np.int(0.75 * len(neg_ex_row_indices))
    for_testing_pos = len(pos_ex_row_indices) - for_training_pos
    for_testing_neg = len(neg_ex_row_indices) - for_training_neg

    # Partition the data
    testing_data = np.concatenate((
        train_data[pos_ex_row_indices[-for_testing_pos:]],
        train_data[neg_ex_row_indices[-for_testing_neg:]]))
    training_data = np.concatenate((
        train_data[pos_ex_row_indices[0:for_training_pos]],
        train_data[neg_ex_row_indices[0:for_training_neg]]))
    training_labels = np.concatenate((
        labels[pos_ex_row_indices[0:for_training_pos]],
        labels[neg_ex_row_indices[0:for_training_neg]]))
    testing_classes = np.concatenate((
        train_classes[pos_ex_row_indices[-for_testing_pos:]],
        train_classes[neg_ex_row_indices[-for_testing_neg:]]))
    training_classes = np.concatenate((
        train_classes[pos_ex_row_indices[0:for_training_pos]],
        train_classes[neg_ex_row_indices[0:for_training_neg]]))

    # Determine best C
    cost = 10.0 ** (np.arange(-2, 3, 1))
    model = svm.train(training_data, training_classes,
                      cost=cost, cv_type="lolo", labels=training_labels)

    # Calculate ROC with the best C
    fpr, tpr, roc_auc = svm.roc(model, training_data, training_classes,
                                testing_data, testing_classes)

    # Save results
    roc_results["FPR"].append(fpr)
    roc_results["TPR"].append(tpr)
    roc_results["AUC"].append(roc_auc)

    return roc_results, model.C
