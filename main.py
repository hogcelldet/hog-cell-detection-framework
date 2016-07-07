# -*- coding: utf-8 -*-

import os
import glob
import time
import pickle
import datetime

import cv2
import numpy as np
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D

import svm
import utils
import detect
import evaluation
import importdata
import hardexamples


def estimate_growth_curve():

    # -------------------------------------------------------------------------
    # Settings
    # -------------------------------------------------------------------------

    default_hog = cv2.HOGDescriptor()
    my_params = dict(
        _winSize=(32, 32),
        _blockSize=(8, 8),
        _blockStride=(4, 4),
        _cellSize=(4, 4),
        _nbins=9,
        _derivAperture=default_hog.derivAperture,
        _winSigma=default_hog.winSigma,
        _histogramNormType=default_hog.histogramNormType,
        _L2HysThreshold=default_hog.L2HysThreshold,
        _gammaCorrection=default_hog.gammaCorrection,
        _nlevels=1  # Max number of HOG window scales
    )
    hog = cv2.HOGDescriptor(**my_params)

    search_hard_examples = True
    load_hard_examples_from_file = False

    # -------------------------------------------------------------------------
    # Import data
    # -------------------------------------------------------------------------

    train_data, train_classes, labels, ground_truth = \
        importdata.import_images_extract_hog_features(
            hog=hog, days=["day1", "day2", "day3"],
            save_annotations=False)

    # -------------------------------------------------------------------------
    # Find hard examples
    # -------------------------------------------------------------------------

    he, he_labels = None, None

    if search_hard_examples:
        he, he_labels, _ = hardexamples.search(
            hog, train_data, train_classes, labels, ground_truth,
            amount_to_initial_training=1.0, save_images_with_detections=True,
            save_hard_example_images=True, max_iters=10)

    if load_hard_examples_from_file:
        with open('savedVariables/hardExamples.pickle') as f:
            he, he_labels = pickle.load(f)

    if he is not None and he_labels is not None:
        train_data = np.concatenate((train_data, he))
        train_classes = np.concatenate((train_classes, [0] * he.shape[0]))
        labels = np.concatenate((labels, he_labels))

    # -------------------------------------------------------------------------
    # Learn the data
    # -------------------------------------------------------------------------

    cost = 10.0**(np.arange(-4, 5, 1))
    model = svm.train(train_data=train_data, train_classes=train_classes,
                      cost=cost, cv_type="lolo", labels=labels)

    # -------------------------------------------------------------------------
    # Detect
    # -------------------------------------------------------------------------

    sliding_window_method = "detectMultiScale"
    detection_settings = {
        "hog": hog,
        "svm": model,
        "sliding_window_method": sliding_window_method,
        "sliding_window_params":
        {
            "hitThreshold": -model.intercept_[0],
            "winStride": (4, 4),
            "padding": (8, 8),
            "scale": 1.05,
            "finalThreshold": 2,
            "useMeanshiftGrouping": False
        },
        "params_and_their_ranges_to_be_varied": [],
        "images": os.path.join("testWithThese"),
        "output_folder": os.path.join(
            sliding_window_method + "_" +
            datetime.datetime.fromtimestamp(time.time()).strftime(
                '%Y-%m-%d_%H-%M-%S')),
        "save_im": True,
        "check_for_break_condition1": False,
        "check_for_break_condition2": False,
        "save_detection_binary_images": True,
    }

    print ""
    print "Estimating growth curve..."
    print "------------------------------------------------------------"

    search_results = detect.image_sweep(**detection_settings)

    # -------------------------------------------------------------------------
    # Plot the growth curve
    # -------------------------------------------------------------------------

    # Average results from each day
    avg_detected_cells = []
    avg_truth_cells = []
    avg_tp = []
    avg_fp = []
    avg_fn = []
    for ix, day in enumerate(search_results):
        images_per_day = 0
        avg_detected_cells.append(0)
        avg_truth_cells.append(0)
        avg_tp.append(0)
        avg_fp.append(0)
        avg_fn.append(0)
        # Sum results from this day
        for im in day:
            avg_detected_cells[ix] += im["TP"][0] + im["FP"][0]
            avg_truth_cells[ix] += im["nr_of_cells_truth"][0]
            images_per_day += 1
            avg_tp[ix] += im["TP"][0]
            avg_fp[ix] += im["FP"][0]
            avg_fn[ix] += im["FN"][0]
        avg_detected_cells[-1] = avg_detected_cells[-1] / images_per_day
        avg_truth_cells[-1] = avg_truth_cells[-1] / images_per_day
        avg_tp[-1] = avg_tp[-1] / images_per_day
        avg_fp[-1] = avg_fp[-1] / images_per_day
        avg_fn[-1] = avg_fn[-1] / images_per_day

    # Normalize
    avg_norm_detected_cells = np.array(avg_detected_cells) / np.array(
        avg_truth_cells[0] * np.ones(len(avg_detected_cells)))
    avg_norm_truth_cells = np.array(avg_truth_cells) / np.array(
        avg_truth_cells[0] * np.ones(len(avg_truth_cells)))
    avg_norm_tp = np.array(avg_tp) / np.array(
        avg_truth_cells[0] * np.ones(len(avg_tp)))
    avg_norm_fp = np.array(avg_fp) / np.array(
        avg_truth_cells[0] * np.ones(len(avg_fp)))
    avg_norm_fn = np.array(avg_fn) / np.array(
        avg_truth_cells[0] * np.ones(len(avg_fn)))

    # Visualize
    # pl.close("all")
    fig = pl.figure(figsize=(14, 7), facecolor='white')
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.grid(color='black', linestyle='-.', linewidth=1, alpha=0.2)
    ax1.plot(avg_norm_truth_cells, color="c", linestyle="-", marker="o",
             label="Manual", linewidth=3)
    ax1.plot(avg_norm_detected_cells, color="m", linestyle="-", marker='o',
             label="HOG TP+FP", linewidth=3)
    ax1.plot(avg_norm_tp, "g--", label="HOG TP", linewidth=2)
    ax1.plot(avg_norm_fp, "r--", label="HOG FP", linewidth=2)
    ax1.plot(avg_norm_fn, "b--", label="HOG FN", linewidth=2)
    ax1.set_xticklabels(["1", "2", "3", "4", "5", "6"])
    ax1.set_xlabel('Experiment time (days)')
    ax1.set_ylabel('Relative number of cells ')
    ax1.legend(loc='upper left')
    pl.draw()

    relative_error = \
        (np.abs(np.array(avg_truth_cells) -
                np.array(avg_detected_cells))) / np.array(avg_truth_cells)
    relative_error = relative_error[1:]
    np.max(relative_error)
    np.mean(relative_error)

    # -------------------------------------------------------------------------
    # Calculate and visualize BSI
    # -------------------------------------------------------------------------

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.grid(color='black', linestyle='-.', linewidth=1, alpha=0.2)

    colors = ["b", "g", "r", "c", "m", "y", "k"]
    utils.create_ground_truth_ims(detection_settings["images"])
    truth_dir = r".\groundTruth"
    estim_dir = detection_settings["output_folder"] + "\\detectionBinaries"
    file_types = ["bmp", 'jpg', 'png']

    truth_folders = utils.list_dirs(truth_dir)
    esim_folders = utils.list_dirs(estim_dir)

    total_bsi_res = {"TET": [], "TEE": [], "dSeg": []}

    # Loop through input folders
    for i in range(len(truth_folders)):

        # Get list of images in the folder
        truth_images = []
        esim_images = []
        for fileType in file_types:
            truth_images = (truth_images + glob.glob(truth_folders[i] +
                            "\*." + fileType))
            esim_images = (esim_images + glob.glob(esim_folders[i] +
                           "\*." + fileType))

        # Loop through the images
        for j in range(len(truth_images)):

            if len(truth_images) > j and len(esim_images) > j:
                truth_image_path = truth_images[j]
                estim_image_path = esim_images[j]

                tet, tee, d_seg = evaluation.bivariate_similarity_index(
                    truth_image_path, estim_image_path)

                total_bsi_res["TET"].append(tet)
                total_bsi_res["TEE"].append(tee)
                total_bsi_res["dSeg"].append(d_seg)

                # Draw label only once
                if j == 0:
                    ax2.scatter(tet, tee, s=120.0, c=colors[i],
                                label="Day " + str(i + 1))
                else:
                    ax2.scatter(tet, tee, s=120.0, c=colors[i])

        ax2.legend(loc="upper left", fontsize=12, scatterpoints=1)

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("TET")
    ax2.set_ylabel("TEE")
    pl.draw()

    print np.mean(total_bsi_res["dSeg"])


def class_separability():

    default_hog = cv2.HOGDescriptor()

    my_params = dict(
        _winSize=(32, 32),
        _blockSize=(8, 8),
        _blockStride=(4, 4),
        _cellSize=(4, 4),
        _nbins=9,
        _derivAperture=default_hog.derivAperture,
        _winSigma=default_hog.winSigma,
        _histogramNormType=default_hog.histogramNormType,
        _L2HysThreshold=default_hog.L2HysThreshold,
        _gammaCorrection=default_hog.gammaCorrection,
        _nlevels=default_hog.nlevels
    )

    hog = cv2.HOGDescriptor(**my_params)

    # Import data and extract HOG features
    train_data, train_classes, labels, ground_truth = importdata. \
        import_images_extract_hog_features(hog=hog,
                                           days=["day1", "day2", "day3"],
                                           save_annotations=False,
                                           n_samples=4000)

    # Divide into training and testing data
    for_testing = 2000
    for_training = 2000

    pos_ex_row_indices = np.where(train_classes == 1)[0]
    neg_ex_row_indices = np.where(train_classes == 0)[0]
    test_d = np.concatenate((
        train_data[pos_ex_row_indices[-for_testing:]],
        train_data[neg_ex_row_indices[-for_testing:]]))
    train_d = np.concatenate((
        train_data[pos_ex_row_indices[0:for_training]],
        train_data[neg_ex_row_indices[0:for_training]]))
    train_l = np.concatenate((
        labels[pos_ex_row_indices[0:for_training]],
        labels[neg_ex_row_indices[0:for_training]]))
    test_c = np.concatenate((
        train_classes[pos_ex_row_indices[-for_testing:]],
        train_classes[neg_ex_row_indices[-for_testing:]]))
    train_c = np.concatenate((
        train_classes[pos_ex_row_indices[0:for_training]],
        train_classes[neg_ex_row_indices[0:for_training]]))

    # -------------------------------------------------------------------------
    # C-V C
    # -------------------------------------------------------------------------

    cost = 10.0 ** (np.arange(-4, 5, 1))
    model = svm.train(train_d, train_c, cost=cost, cv_type="lolo",
                      labels=train_l)

    # -------------------------------------------------------------------------
    # Visualize
    # -------------------------------------------------------------------------

    pl.close("all")
    fig = pl.figure(figsize=(13, 6), facecolor='none')
    # ROC
    ax1 = fig.add_subplot(1, 2, 1)  # , adjustable='box', aspect=1.0)
    # HOG parameters table
    ax2 = fig.add_subplot(1, 2, 2)  # , adjustable='box', aspect=1.0)
    ax2.axis('off')

    # -------------------------------------------------------------------------
    # Visualize class separability
    # -------------------------------------------------------------------------

    dot_product = np.dot(test_d, model.coef_.transpose())
    dot_product = np.array(
        [x[0] + model.intercept_[0] for x in dot_product])  # add bias

    class0 = dot_product[np.where(test_c == 0)[0]]
    class1 = dot_product[np.where(test_c == 1)[0]]

    ax1.hist(class1, bins=50, normed=True, color="#003333", alpha=0.7,
             histtype='stepfilled', label=["Cancer cells"],
             edgecolor="none")

    ax1.hist(class0, bins=50, normed=True, color="#FF3333", alpha=0.7,
             histtype='stepfilled', label=["Everything else"],
             edgecolor="none")

    ax1.set_xlabel("Location", fontsize=12)
    ax1.set_ylabel("Frequency", fontsize=12)

    ax1.legend(loc="upper right", fontsize=12)
    pl.show()
    pl.draw()

    # -------------------------------------------------------------------------
    # Draw HOG parameters table
    # -------------------------------------------------------------------------

    utils.draw_hog_params_table(hog_params=my_params, axis=ax2)
    utils.save_fig(file_name_prefix='ClassSeparability',
                   hog_params=my_params)


def effect_of_train_data_amount():

    default_hog = cv2.HOGDescriptor()

    my_params = dict(
        _winSize=(32, 32),
        _blockSize=(8, 8),
        _blockStride=(4, 4),
        _cellSize=(4, 4),
        _nbins=2,
        _derivAperture=default_hog.derivAperture,
        _winSigma=default_hog.winSigma,
        _histogramNormType=default_hog.histogramNormType,
        _L2HysThreshold=default_hog.L2HysThreshold,
        _gammaCorrection=default_hog.gammaCorrection,
        _nlevels=1
    )

    hog = cv2.HOGDescriptor(**my_params)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Import twice as many samples as you are going to use
    # for training at maximum
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Import data and extract HOG features
    train_data, train_classes, labels, ground_truth = importdata. \
        import_images_extract_hog_features(hog=hog,
                                           days=["day1", "day2", "day3"],
                                           save_annotations=False,
                                           n_samples=4000)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Find hard examples
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    hard_examples, hard_example_labels, _ = hardexamples.search(
        hog, train_data, train_classes, labels, ground_truth,
        amount_to_initial_training=0.1,
        save_images_with_detections=True, save_hard_example_images=True)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Divide data into train data and test data
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    for_testing = 2000
    pos_ex_row_indices = np.where(train_classes == 1)[0]
    neg_ex_row_indices = np.where(train_classes == 0)[0]
    test_d = np.concatenate((train_data[pos_ex_row_indices[-for_testing:]],
                            train_data[neg_ex_row_indices[-for_testing:]]))
    test_c = np.concatenate((train_classes[pos_ex_row_indices[-for_testing:]],
                            train_classes[neg_ex_row_indices[-for_testing:]]))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build classifiers with different amount of training data
    # and calculate their ROC
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Initialize dictionary of lists where results will be saved
    results = {"FPR": [], "TPR": [], "AUC": [], "fvl": [],
               "forTraining": [], "nrOfNe": [], "nrOfHe": []}

    hard_examples_portion_of_negatives = 0.25
    for i, forTraining in enumerate([10, 100, 500, 1000, 2000]):
        nr_he = np.int(hard_examples_portion_of_negatives * forTraining)
        nr_ne = np.int((1.0-hard_examples_portion_of_negatives) * forTraining)

        train_d = np.concatenate((
            train_data[pos_ex_row_indices[0:forTraining]],
            train_data[neg_ex_row_indices[0:nr_ne]], hard_examples[0:nr_he]))

        train_c = np.concatenate((
            train_classes[pos_ex_row_indices[0:forTraining]],
            train_classes[neg_ex_row_indices[0:nr_ne]], ([0] * nr_he)))

        train_l = np.concatenate((labels[pos_ex_row_indices[0:forTraining]],
                                 labels[neg_ex_row_indices[0:nr_ne]],
                                 hard_example_labels[0:nr_he]))

        # Build classifier with cross-validated cost
        cost = 10.0 ** (np.arange(-4, 5, 1))
        model = svm.train(train_d, train_c, cost=cost, cv_type="lolo",
                          labels=train_l, scoring='accuracy')

        # Calculate ROC
        fpr, tpr, roc_auc = svm.roc(model, train_d, train_c, test_d, test_c)

        results["FPR"].append(fpr)
        results["TPR"].append(tpr)
        results["AUC"].append(roc_auc)
        results["fvl"].append(train_data.shape[1])
        results["forTraining"].append(str(forTraining))
        results["nrOfNe"].append(str(nr_ne))
        results["nrOfHe"].append(str(nr_he))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Visualize
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    pl.close("all")
    fig = pl.figure(figsize=(13, 6), facecolor='none')
    ax1 = fig.add_subplot(1, 2, 1, adjustable='box', aspect=1.0)  # ROC
    ax2 = fig.add_subplot(1, 2, 2, adjustable='box',
                          aspect=1.0)  # HOG parameters table
    ax2.axis('off')

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Draw area to be zoomed in and lines to zoomed area.
    # This is done before ROC so that ROC curves will be on top of these.
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    start_coords = (0.0, 0.9)
    end_x = 0.1
    end_y = 1.0
    # Small rectangle
    ax1.add_patch(pl.Rectangle(start_coords, end_x, end_y, facecolor='white',
                               edgecolor='black'))
    # Lines going to big rectangle
    x_range = [start_coords[0], 0.3]
    y_range = [start_coords[1], 0.4]
    ax1.plot(x_range, y_range, '#cccccc')
    # Lines going to big rectangle
    x_range = [end_x, 0.6]
    y_range = [end_y, 0.7]
    ax1.plot(x_range, y_range, '#cccccc')

    rect = [0.3, 0.4, 0.3, 0.3]
    ax11 = utils.add_subplot_axes(ax1, rect)

    # Hide ticks
    ax11.get_xaxis().set_visible(False)
    ax11.get_yaxis().set_visible(False)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot ROC curve
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    width = 3
    line_styles = ["b", "g", "r", "c", "m", "y"]
    for i in range(len(results["AUC"])):
        ax1.plot(
            results["FPR"][i], results["TPR"][i], line_styles[i],
            linewidth=width,
            label=(results["forTraining"][i] + "+  " + "  " *
                   (4 - len(results["forTraining"][i])) +
                   results["nrOfNe"][i] + "-  " + "  " *
                   (4 - len(results["nrOfNe"][i])) +
                   results["nrOfHe"][i] + "-- " + "  " *
                   (4 - len(results["nrOfHe"][i])) +
                   "AUC %0.3f" % results["AUC"][i]))
        ax1.legend(loc="lower right", fontsize=12)

    ax1.plot([0, 1], [0, 1], 'k--')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.0])
    ax1.set_xlabel('False positive rate (FPR)')
    ax1.set_ylabel('True positive rate (TPR)')
    pl.show()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Draw zoomed in area
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    for i in range(len(results["AUC"])):
        ax11.plot(results["FPR"][i],
                  results["TPR"][i],
                  line_styles[i], linewidth=width)
        ax11.axis([0.0, 0.09, 0.92, 1.0])
    pl.show()
    pl.draw()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Draw HOG parameters table
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    utils.draw_hog_params_table(hog_params=my_params, axis=ax2)
    utils.save_fig(file_name_prefix='CompareEffectOfAmountOfTrainingData',
                   hog_params=my_params)


def effect_of_hard_examples_amount():

    default_hog = cv2.HOGDescriptor()
    my_params = dict(
        _winSize=(32, 32),
        _blockSize=(8, 8),
        _blockStride=(4, 4),
        _cellSize=(4, 4),
        _nbins=9,
        _derivAperture=default_hog.derivAperture,
        _winSigma=default_hog.winSigma,
        _histogramNormType=default_hog.histogramNormType,
        _L2HysThreshold=default_hog.L2HysThreshold,
        _gammaCorrection=default_hog.gammaCorrection,
        _nlevels=1)
    hog = cv2.HOGDescriptor(**my_params)

    # Import data and extract HOG features
    train_data, train_classes, labels, ground_truth = importdata. \
        import_images_extract_hog_features(
            hog, days=["day1", "day2", "day3"], save_annotations=False)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Find hard examples
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    he, he_labels, he_results = hardexamples.search(
        hog, train_data, train_classes, labels, ground_truth,
        amount_to_initial_training=.01, save_images_with_detections=True,
        save_hard_example_images=True, max_iters=1000,
        calculate_roc=True, roc_for_this_many_first_iters=6)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Draw ROC curves
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    pl.close("all")
    font = {'family': 'sans-serif',
            'weight': 'normal',
            'size': '12'}

    pl.rc('font', **font)
    fig = pl.figure(figsize=(14, 7), facecolor='none')
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot([0, 1], [0, 1], 'k--')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.0])
    ax1.set_xlabel('False positive rate (FPR)')
    ax1.set_ylabel('True positive rate (TPR)')

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Draw area to be zoomed in and lines to zoomed area.
    # This is done before ROC so that ROC curves will be on top of these.
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    start_coords = (0.0, 0.9)
    end_x = 0.1
    end_y = 1.0
    # Small rectangle
    ax1.add_patch(pl.Rectangle(start_coords, end_x, end_y, facecolor='white',
                               edgecolor='black'))

    # Lines going to big rectangle
    x_range = [start_coords[0], 0.3]
    y_range = [start_coords[1], 0.4]
    ax1.plot(x_range, y_range, '#cccccc')
    # Lines going to big rectangle
    x_range = [end_x, 0.6]
    y_range = [end_y, 0.7]
    ax1.plot(x_range, y_range, '#cccccc')

    rect = [0.3, 0.4, 0.3, 0.3]
    ax2 = utils.add_subplot_axes(ax1, rect)

    # Hide ticks
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot ROC curve on the left-hand side
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    width = 3
    colors = ["b", "g", "r", "c", "m", "y", "k"]

    for i in range(len(he_results["FPR"])):

        if i == 0:
            ax1.plot(he_results["FPR"][i], he_results["TPR"][i],
                     colors[i], linewidth=width,
                     label="Iteration " + str(he_results["iter"][i]) +
                           "                                           " +
                           "AUC %0.3f" % he_results["AUC"][i])
        else:
            ax1.plot(he_results["FPR"][i], he_results["TPR"][i],
                     colors[i], linewidth=width,
                     label=(
                         "Iteration " + str(he_results["iter"][i]) +
                         "  " * (4 - len(str(he_results["iter"][i]))) +
                         "Hard examples " + str(he_results["nrOfIterHE"][i]) +
                         "  " * (6 - len(str(he_results["nrOfIterHE"][i]))) +
                         "AUC %0.3f" % he_results["AUC"][i]))

        ax1.legend(loc="lower right", fontsize=12)
        ax1.grid()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Draw zoomed in area
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    for i in range(len(he_results["AUC"])):
        ax2.plot(he_results["FPR"][i],
                 he_results["TPR"][i],
                 colors[i], linewidth=width)
    ax2.axis([0.0, 0.09, 0.92, 1.0])
    pl.show()
    pl.draw()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot F1-score on the right-hand side
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ax3 = fig.add_subplot(1, 2, 2)
    # -1 because on last iter there was no detection process
    detection_iters = he_results["iter"][-1] - 1
    ax3.set_ylim([0.0, 1.0])
    ax3.set_xlim([2, detection_iters + 1])
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('F1-score')
    mean_f1 = np.zeros(detection_iters)
    index = 0
    tested_images = 10  # 6 from day 1, 2 from day 2, 2 from day 3
    for i in np.arange(1, len(he_results["F1"]), 1):
        if i % tested_images == 0:
            index += 1
        mean_f1[index] += he_results["F1"][i - 1]  # -1 because i starts from 1

    for i in range(len(mean_f1)):
        mean_f1[i] /= tested_images
    ax3.plot(np.arange(2, detection_iters + 2), mean_f1,
             color="black", lw=3)
    ax3.grid()
    pl.draw()


def hog_params_in_classification():

    param_to_vary = "cellSize"
    default_hog = cv2.HOGDescriptor()

    if param_to_vary == "winSize":
        value_range = 2 ** (np.arange(4, 7))
        my_params = dict(
            _cellSize=(4, 4),
            _nbins=9,
            _derivAperture=default_hog.derivAperture,
            _winSigma=default_hog.winSigma,
            _histogramNormType=default_hog.histogramNormType,
            _L2HysThreshold=default_hog.L2HysThreshold,
            _gammaCorrection=default_hog.gammaCorrection,
            _nlevels=default_hog.nlevels)
    elif param_to_vary == "blockSize":
        value_range = 2 ** (np.arange(2, 6))
        my_params = dict(
            _winSize=(32, 32),
            _blockStride=(4, 4),
            _cellSize=(4, 4),
            _nbins=9,
            _derivAperture=default_hog.derivAperture,
            _winSigma=default_hog.winSigma,
            _histogramNormType=default_hog.histogramNormType,
            _L2HysThreshold=default_hog.L2HysThreshold,
            _gammaCorrection=default_hog.gammaCorrection,
            _nlevels=default_hog.nlevels)
    elif param_to_vary == "blockStride":
        value_range = 2 ** (np.arange(1, 4))
        my_params = dict(
            _winSize=(32, 32),
            _blockSize=(8, 8),
            _cellSize=(4, 4),
            _nbins=9,
            _derivAperture=default_hog.derivAperture,
            _winSigma=default_hog.winSigma,
            _histogramNormType=default_hog.histogramNormType,
            _L2HysThreshold=default_hog.L2HysThreshold,
            _gammaCorrection=default_hog.gammaCorrection,
            _nlevels=default_hog.nlevels)
    elif param_to_vary == "cellSize":
        value_range = 2 ** (np.arange(1, 4))
        my_params = dict(
            _winSize=(32, 32),
            _blockSize=(8, 8),
            _blockStride=(8, 8),
            _nbins=9,
            _derivAperture=default_hog.derivAperture,
            _winSigma=default_hog.winSigma,
            _histogramNormType=default_hog.histogramNormType,
            _L2HysThreshold=default_hog.L2HysThreshold,
            _gammaCorrection=default_hog.gammaCorrection,
            _nlevels=default_hog.nlevels)
    elif param_to_vary == "nbins":
        value_range = [1, 2, 3, 6, 9, 18, 36]
        my_params = dict(
            _winSize=(32, 32),
            _blockSize=(16, 16),
            _blockStride=(8, 8),
            _cellSize=(8, 8),
            _derivAperture=default_hog.derivAperture,
            _winSigma=default_hog.winSigma,
            _histogramNormType=default_hog.histogramNormType,
            _L2HysThreshold=default_hog.L2HysThreshold,
            _gammaCorrection=default_hog.gammaCorrection,
            _nlevels=default_hog.nlevels)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build classifiers with different values and calculate their ROC
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Initialize dictionary of lists where results will be saved
    roc_results = {"FPR": [], "TPR": [], "AUC": [], "fvl": [],
                   param_to_vary: []}
    for value in value_range:
        if param_to_vary == "winSize":
            my_params["_winSize"] = (value, value)
            my_params["_blockSize"] = (value / 2, value / 2)
            my_params["_blockStride"] = (value / 4, value / 4)
            my_params["_cellSize"] = (value / 4, value / 4)
        elif param_to_vary == "blockSize":
            my_params["_blockSize"] = (value, value)
        elif param_to_vary == "blockStride":
            my_params["_blockStride"] = (value, value)
        elif param_to_vary == "cellSize":
            my_params["_cellSize"] = (value, value)
        elif param_to_vary == "nbins":
            my_params["_nbins"] = value

        hog = cv2.HOGDescriptor(**my_params)

        # Import data and extract HOG features
        train_data, train_classes, labels, ground_truth = importdata. \
            import_images_extract_hog_features(hog=hog,
                                               days=["day1", "day2", "day3"],
                                               save_annotations=False,
                                               n_samples=4000)

        # Divide into training and testing data
        for_testing = 2000
        for_training = 2000

        pos_ex_row_ix = np.where(train_classes == 1)[0]
        neg_ex_row_ix = np.where(train_classes == 0)[0]
        test_d = np.concatenate((train_data[pos_ex_row_ix[-for_testing:]],
                                train_data[neg_ex_row_ix[-for_testing:]]))
        train_d = np.concatenate((train_data[pos_ex_row_ix[0:for_training]],
                                 train_data[neg_ex_row_ix[0:for_training]]))
        train_l = np.concatenate((labels[pos_ex_row_ix[0:for_training]],
                                 labels[neg_ex_row_ix[0:for_training]]))
        test_c = np.concatenate((train_classes[pos_ex_row_ix[-for_testing:]],
                                train_classes[neg_ex_row_ix[-for_testing:]]))
        train_c = np.concatenate((train_classes[pos_ex_row_ix[0:for_training]],
                                 train_classes[neg_ex_row_ix[0:for_training]]))

        print "\n------------------------------------------------------------"
        print param_to_vary + ":", value
        print "trainData.shape:", train_data.shape
        print "------------------------------------------------------------"

        # Build classifier with cross-validating cost
        cost = 10.0 ** (np.arange(-4, 5, 1))
        model = svm.train(train_d, train_c, cost=cost, cv_type="lolo",
                          labels=train_l)

        # Calculate ROC
        fpr, tpr, roc_auc = svm.roc(model, train_d, train_c, test_d, test_c)

        svm.visualize_class_separability(model, train_d, train_c)

        roc_results["FPR"].append(fpr)
        roc_results["TPR"].append(tpr)
        roc_results["AUC"].append(roc_auc)
        roc_results["fvl"].append(train_data.shape[1])
        roc_results[param_to_vary].append((value, value))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Visualize
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    pl.close("all")
    fig = pl.figure(figsize=(13, 6), facecolor='none')
    ax1 = fig.add_subplot(1, 2, 1, adjustable='box', aspect=1.0)  # ROC
    ax2 = fig.add_subplot(1, 2, 2, adjustable='box',
                          aspect=1.0)  # HOG parameters table
    ax2.axis('off')

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Draw area to be zoomed in and lines to zoomed area.
    # This is done before ROC so that ROC curves will be on top of these.
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    start_coords = (0.0, 0.9)
    end_x = 0.1
    end_y = 1.0
    # Small rectangle
    ax1.add_patch(pl.Rectangle(start_coords, end_x, end_y, facecolor='white',
                               edgecolor='black'))

    # Lines going to big rectangle
    x_range = [start_coords[0], 0.3]
    y_range = [start_coords[1], 0.4]
    ax1.plot(x_range, y_range, '#cccccc')
    # Lines going to big rectangle
    x_range = [end_x, 0.6]
    y_range = [end_y, 0.7]
    ax1.plot(x_range, y_range, '#cccccc')

    rect = [0.3, 0.4, 0.3, 0.3]
    ax11 = utils.add_subplot_axes(ax1, rect)

    # Hide ticks
    ax11.get_xaxis().set_visible(False)
    ax11.get_yaxis().set_visible(False)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot ROC curve
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    width = 3
    line_styles = ["b", "g", "r", "c", "m", "y", "k"]
    for i in range(len(roc_results["AUC"])):

        if param_to_vary == "winSize":
            label_text = "Window size "
        elif param_to_vary == "blockSize":
            label_text = "Block size "
        elif param_to_vary == "blockStride":
            label_text = "Block stride "
        elif param_to_vary == "cellSize":
            label_text = "Cell size "
        elif param_to_vary == "nbins":
            label_text = "Nr of bins "

        ax1.plot(roc_results["FPR"][i], roc_results["TPR"][i], line_styles[i],
                 linewidth=width, label=(
                label_text + str(roc_results[param_to_vary][i]) +
                "  " * (10 - len(str(roc_results[param_to_vary][i]))) +
                "fvl " + str(roc_results["fvl"][i]) +
                "  " * (6 - len(str(roc_results["fvl"][i]))) +
                "AUC %0.3f" % roc_results["AUC"][i]))

        ax1.legend(loc="lower right", fontsize=12)

    ax1.plot([0, 1], [0, 1], 'k--')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.0])
    ax1.set_xlabel('False positive rate (FPR)')
    ax1.set_ylabel('True positive rate (TPR)')
    pl.show()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Draw zoomed in area
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    for i in range(len(roc_results["AUC"])):
        ax11.plot(roc_results["FPR"][i],
                  roc_results["TPR"][i],
                  line_styles[i], linewidth=width)
    ax11.axis([0.0, 0.09, 0.92, 1.0])
    pl.show()
    pl.draw()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Draw HOG parameters table
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    if param_to_vary == "blockSize":
        row_labels = ['winSize',
                      'blockStride',
                      'cellSize',
                      'nbins']
        table_vals = [[my_params["_winSize"]],
                      [my_params["_blockStride"]],
                      [my_params["_cellSize"]],
                      [my_params["_nbins"]]]

    if param_to_vary == "blockStride":
        row_labels = ['winSize',
                      'blockSize',
                      'cellSize',
                      'nbins']
        table_vals = [[my_params["_winSize"]],
                      [my_params["_blockSize"]],
                      [my_params["_cellSize"]],
                      [my_params["_nbins"]]]

    if param_to_vary == "cellSize":
        row_labels = ['winSize',
                      'blockSize',
                      'blockStride',
                      'nbins']
        table_vals = [[my_params["_winSize"]],
                      [my_params["_blockSize"]],
                      [my_params["_blockStride"]],
                      [my_params["_nbins"]]]

    if param_to_vary == "nbins":
        row_labels = ['winSize',
                      'blockSize',
                      'blockStride',
                      'cellSize']
        table_vals = [[my_params["_winSize"]],
                      [my_params["_blockSize"]],
                      [my_params["_blockStride"]],
                      [my_params["_cellSize"]]]

    the_table = ax2.table(cellText=table_vals,
                          colWidths=[0.1] * 3,
                          rowLabels=row_labels,
                          loc='lower left')
    the_table.set_fontsize(12)
    the_table.scale(2, 2)
    pl.show()
    pl.draw()

    utils.save_fig(
        file_name_prefix="Compare_" + param_to_vary + "_InClassification",
        hog_params=my_params)


def hog_params_in_sliding_window_detection():

    param_to_vary = "nbins"

    default_hog = cv2.HOGDescriptor()

    if param_to_vary == "blockSize":
        value_range = 2 ** (np.arange(2, 6))
        my_params = dict(
            _winSize=(32, 32),
            _cellSize=(4, 4),
            _nbins=9,
            _derivAperture=default_hog.derivAperture,
            _winSigma=default_hog.winSigma,
            _histogramNormType=default_hog.histogramNormType,
            _L2HysThreshold=default_hog.L2HysThreshold,
            _gammaCorrection=default_hog.gammaCorrection,
            _nlevels=default_hog.nlevels)
    elif param_to_vary == "cellSize":
        value_range = 2 ** (np.arange(1, 6))
        my_params = dict(
            _winSize=(32, 32),
            _blockSize=(32, 32),
            _blockStride=(16, 16),
            _nbins=9,
            _derivAperture=default_hog.derivAperture,
            _winSigma=default_hog.winSigma,
            _histogramNormType=default_hog.histogramNormType,
            _L2HysThreshold=default_hog.L2HysThreshold,
            _gammaCorrection=default_hog.gammaCorrection,
            _nlevels=default_hog.nlevels)
    elif param_to_vary == "nbins":
        value_range = 2 ** (np.arange(1, 6))
        my_params = dict(
            _winSize=(32, 32),
            _blockSize=(16, 16),
            _blockStride=(8, 8),
            _cellSize=(4, 4),
            _derivAperture=default_hog.derivAperture,
            _winSigma=default_hog.winSigma,
            _histogramNormType=default_hog.histogramNormType,
            _L2HysThreshold=default_hog.L2HysThreshold,
            _gammaCorrection=default_hog.gammaCorrection,
            _nlevels=1)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build classifiers with different values and try finding objects with them
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    results = []
    images = r".\testWithThese\day3\Tile002496.bmp"
    search_method = "detectMultiScale"

    for value in value_range:
        if param_to_vary == "winSize":
            my_params["_winSize"] = (value, value)
            my_params["_blockSize"] = (value / 4, value / 4)
            my_params["_blockStride"] = (value / 8, value / 8)
            my_params["_cellSize"] = (value / 8, value / 8)
        elif param_to_vary == "blockSize":
            my_params["_blockSize"] = (value, value)
            my_params["_blockStride"] = (value / 2, value / 2)
        elif param_to_vary == "cellSize":
            my_params["_cellSize"] = (value, value)
        elif param_to_vary == "nbins":
            my_params["_nbins"] = value

        hog = cv2.HOGDescriptor(**my_params)

        # Import data and extract HOG features
        train_data, train_classes, labels, ground_truth = importdata. \
            import_images_extract_hog_features(hog=hog,
                                               days=["day1", "day2", "day3"],
                                               save_annotations=False,
                                               n_samples=100)

        # Build classifier with cross-validated cost
        model = svm.train(train_data, train_classes, cost=0.01)

        hog.setSVMDetector(model.coef_[0])

        search_results = detect.image_sweep(
            hog=hog, svm=model, sliding_window_method=search_method,
            sliding_window_params=dict(
                hitThreshold=-model.intercept_[0],
                winStride=(4, 4),
                padding=(8, 8),
                scale=1.05,
                finalThreshold=2,
                useMeanshiftGrouping=False),
            params_and_their_ranges_to_be_varied=[],
            images=images,
            output_folder=(r".\\" + search_method + "_" +
                           datetime.datetime.fromtimestamp(
                               time.time()).strftime('%Y-%m-%d_%H-%M-%S')),
            save_im=True)

        results.append(search_results)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Visualize
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    if os.path.isfile(images):
        # One image was used
        pl.close("all")
        fig = pl.figure(figsize=(6, 6), facecolor='white')
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.set_xticks(range(len(value_range)))
        ax1.set_xticklabels([str(i) for i in value_range])
        ax1.set_ylim(0, 1)
        ax1.set_xlabel(param_to_vary)
        ax1.set_ylabel("F1-score")
        for ix, res in enumerate(results):
            ax1.scatter(ix, res[0]["F1"])
            pl.draw()


def compare_sliding_window_params():
    default_hog = cv2.HOGDescriptor()

    my_params = dict(
        _winSize=(32, 32),  # Window size
        _blockSize=(16, 16),  # Block size
        _blockStride=(8, 8),  # Block step size
        _cellSize=(4, 4),  # Cell size
        _nbins=9,  # Number of orientation bins
        _derivAperture=default_hog.derivAperture,
        _winSigma=default_hog.winSigma,
        _histogramNormType=default_hog.histogramNormType,
        _L2HysThreshold=default_hog.L2HysThreshold,
        _gammaCorrection=default_hog.gammaCorrection,
        _nlevels=1  # Max number of HOG window increases
    )
    hog = cv2.HOGDescriptor(**my_params)

    # Import data and extract HOG features
    train_data, train_classes, labels, ground_truth = importdata. \
        import_images_extract_hog_features(
            hog=hog, days=["day1", "day2", "day3"], save_annotations=False,
            n_samples=100)

    # -------------------------------------------------------------------------
    # Find hard examples
    # -------------------------------------------------------------------------

    search_hard_examples = True
    load_hard_examples_from_file = False

    if search_hard_examples:
        he, he_labels, _ = hardexamples.search(
            hog, train_data, train_classes, labels, ground_truth,
            amount_to_initial_training=1.0, save_images_with_detections=True,
            save_hard_example_images=True, max_iters=1000,
            calculate_roc=False, roc_for_this_many_first_iters=5)

        train_data = np.concatenate((train_data, he))
        train_classes = np.concatenate((train_classes, [0] * he.shape[0]))
        labels = np.concatenate((labels, he_labels))

    # Load from file
    if load_hard_examples_from_file:
        with open('savedVariables/hardExamples.pickle') as f:
            he, he_labels = pickle.load(f)

        train_data = np.concatenate((train_data, he))
        train_classes = np.concatenate((train_classes, [0] * he.shape[0]))
        labels = np.concatenate((labels, he_labels))

    # -------------------------------------------------------------------------
    # Learn the data
    # -------------------------------------------------------------------------

    learn_new_model = True
    save_model_to_hard_drive = False

    if learn_new_model:
        cost = 10.0 ** (np.arange(-2, 3, 1))
        model = svm.train(train_data, train_classes,
                          cost=cost, cv_type="lolo", labels=labels)
    else:
        with open('savedVariables/'
                  'trainData,trainClasses,labels,model.pickle') as f:
            train_data, train_classes, labels, model = pickle.load(f)

    if save_model_to_hard_drive:
        with open('savedVariables/trainData,trainClasses,model_30_1', 'w')as f:
            pickle.dump([train_data, train_classes, model], f)

    # -------------------------------------------------------------------------
    # Vary detection parameters
    # -------------------------------------------------------------------------

    search_method = "detectMultiScale"
    # search_method = "detect"

    images = r".\testWithThese\day2\Tile002496.bmp"
    # images = r".\testWithThese"
    # images = r".\trainWithThese"

    params_to_study = ["scale", "winStride"]
    # params_to_study = ["winStride"]
    # params_to_study = ["nlevels"]

    all_param_names_and_ranges = dict(
        hitThreshold=[0.1, 0.0, -0.1],  # np.arange(5.0,-5.0,-0.1),
        winStride=[(s, s) for s in range(5, 10, 2)],
        # [(s,s) for s in range(1,9,1)],
        padding=[(s, s) for s in range(1, 33, 1)],
        scale=[1.0, 1.05, 1.1],  # np.arange(1.00,1.09,0.01),
        finalThreshold=np.arange(1.2, 2.8, 0.2),
        nlevels=[1, 10, 20, 64]  # np.arange(1,11,1)
    )

    # Set default search parameters
    if search_method == "detectMultiScale":
        sliding_window_params = dict(
            hitThreshold=-model.intercept_[0],
            winStride=(4, 4),
            padding=(8, 8),
            scale=1.05,
            finalThreshold=2,
            useMeanshiftGrouping=False)
    elif search_method == "detect":
        sliding_window_params = dict(
            hitThreshold=-model.intercept_[0],
            winStride=(4, 4),
            padding=(8, 8))
    else:
        raise (Exception("Unknown search_method: %s" % search_method))

    # Pick up ranges for selected parameters
    if len(params_to_study) == 1:
        params_and_their_ranges_to_be_varied = (
            params_to_study[0],
            all_param_names_and_ranges[params_to_study[0]])
    elif len(params_to_study) == 2:
        params_and_their_ranges_to_be_varied = (
            params_to_study[0],
            all_param_names_and_ranges[params_to_study[0]],
            params_to_study[1],
            all_param_names_and_ranges[params_to_study[1]])
    else:
        raise (Exception("params_to_study length must equal to 1 or 2"))

    check_for_break_condition1 = False
    check_for_break_condition2 = False
    if "hitThreshold" in params_to_study:
        check_for_break_condition1 = True

    search_results = detect.image_sweep(
        hog=hog, svm=model, sliding_window_method=search_method,
        sliding_window_params=sliding_window_params,
        params_and_their_ranges_to_be_varied=(
            params_and_their_ranges_to_be_varied),
        images=images,
        output_folder=(r".\\" + search_method + "_" +
                       datetime.datetime.fromtimestamp(
                           time.time()).strftime('%Y-%m-%d_%H-%M-%S')),
        save_im=True,
        check_for_break_condition1=check_for_break_condition1,
        check_for_break_condition2=check_for_break_condition2)

    # -------------------------------------------------------------------------
    # Visualize results
    # -------------------------------------------------------------------------

    if len(params_to_study) == 1:
        draw_f1 = True
        draw_roc = False

        if os.path.isfile(images):
            # Search targeted one image

            if draw_roc:
                pl.figure(figsize=(7, 7), facecolor='white')
                pl.plot(search_results[0]["FPR"],
                        search_results[0]["TPR"], lw=2)
                pl.xlim(0, 1)
                pl.ylim(0, 1)

            if draw_f1:
                fig = pl.figure(figsize=(7, 7), facecolor='white')
                ax1 = fig.add_subplot(1, 1, 1)
                ax1.plot(range(len(search_results[0]["F1"])),
                         search_results[0]["F1"], lw=2)
                ax1.set_xlabel(params_to_study[0])
                ax1.set_ylabel("F1-score")
                ax1.set_xticks(
                    range(len(params_and_their_ranges_to_be_varied[1])))
                ax1.set_xticklabels(
                    [str(x) for x in params_and_their_ranges_to_be_varied[1]])
                pl.ylim(0, 1)
                pl.tight_layout()
                # pl.draw()
                pl.show()

        else:
            # Search targeted folders of images

            if draw_roc:
                pl.figure(figsize=(7, 7), facecolor='white')
                for folder in search_results:
                    for image in folder:
                        pl.plot(image["FPR"], image["TPR"], lw=2)
                        pl.xlim(0, 1)
                        pl.ylim(0, 1)

            # Plot F1-score as a function of varied parameter
            elif draw_f1:
                pl.close("all")
                fig = pl.figure(figsize=(6, 6), facecolor='white')
                ax1 = fig.add_subplot(1, 1, 1)
                ax1.set_ylim(0, 1)
                ax1.set_xlabel(params_to_study[0])
                ax1.set_ylabel("F1-score")

                if params_to_study[0] == "hitThreshold":
                    # Determine which img had shortest range of F1-values
                    # searched, so that equally short results can be drawn
                    # between images from each day, according to the shortest
                    # range of F1-values searched. Different images have
                    # different search lengths because of
                    # "check_for_break_condition1" early stopping.
                    # We did not always search the whole range of hitThreshold
                    # values to save some time. We stopped the search short
                    # after reaching peak F1 scores.
                    f1_lengths = []
                    for i, folder in enumerate(search_results):
                        for j, image in enumerate(folder):
                            f1_lengths.append(len(image["F1"]))
                    shortest_f1 = np.min(f1_lengths)
                    x_range = all_param_names_and_ranges[
                                  params_to_study[0]][0:shortest_f1]
                    ax1.set_xlim(np.min(x_range), np.max(x_range))
                else:
                    # Other than hitThreshold parameters: use the full range
                    # of parameter values.
                    x_range = range(
                        len(params_and_their_ranges_to_be_varied[1]))
                    shortest_f1 = len(x_range)
                    ax1.set_xticks(x_range)
                    ax1.set_xticklabels(
                        [str(x) for x in
                         params_and_their_ranges_to_be_varied[1]])

                # Average for each varied parameter value (on x-axis)
                # for each day
                mean_f1_res = []
                for i, folder in enumerate(search_results):
                    # Initialize list for the mean scores
                    mean_f1_res.append(np.zeros(shortest_f1))
                    # Add results together
                    for j, image in enumerate(folder):
                        for k in range(shortest_f1):
                            mean_f1_res[i][k] += image["F1"][k]

                # Average and plot
                colors = ["b", "g", "r", "c", "m", "y", "k"]
                best_mean_f1 = []
                best_mean_xval = []
                for i in range(len(mean_f1_res)):
                    # Average results by dividing by the number of images
                    # taken each day
                    for j in range(len(mean_f1_res[i])):
                        mean_f1_res[i][j] /= len(search_results[i])
                    # Plot
                    ax1.plot(x_range, mean_f1_res[i], lw=3, color=colors[i],
                             label="Day " + str(i + 1))
                    ax1.legend(loc="best", fontsize=12)
                    # Stem best F1 score
                    best_mean_f1.append(np.max(mean_f1_res[i]))

                    if params_to_study[0] == "hitThreshold":
                        best_mean_xval.append(x_range[np.argmax(mean_f1_res[i])])
                    else:
                        best_mean_xval.append(
                            all_param_names_and_ranges[params_to_study[0]]
                            [x_range[np.argmax(mean_f1_res[i])]])

                    markerline = pl.stem([x_range[np.argmax(mean_f1_res[i])]],
                                         [np.max(mean_f1_res[i])],
                                         colors[i] + '-.')
                    pl.setp(markerline, 'markerfacecolor', colors[i])

                if params_to_study[0] == "hitThreshold":
                    ax1.annotate(
                        "Mean\n" + str(round(np.mean(best_mean_xval), 1)),
                        xy=(np.mean(best_mean_xval), 0), xycoords='data',
                        xytext=(0.4, 0.15), textcoords='axes fraction',
                        arrowprops=dict(facecolor='black', width=2,
                                        headwidth=10, shrink=0.05),
                        horizontalalignment='right',
                        verticalalignment='top')
                else:
                    ax1.annotate(
                        "Mean\n" + str(round(np.mean(best_mean_xval), 1)),
                        xy=(np.where(
                            all_param_names_and_ranges[params_to_study[0]] ==
                            np.mean(best_mean_xval))[0], 0), xycoords='data',
                        xytext=(0.4, 0.15), textcoords='axes fraction',
                        arrowprops=dict(facecolor='black', width=2,
                                        headwidth=10, shrink=0.05),
                        horizontalalignment='right',
                        verticalalignment='top')

                # pl.draw()
                pl.show()

                # Calculate mean of best mean F1 scores
                best_f1_res = []  # np.zeros(len(searchResults))
                for i, folder in enumerate(search_results):
                    for j, image in enumerate(folder):
                        best_f1_res.append(np.max(image["F1"]))
                print np.mean(best_f1_res)

    # Draw 3D barplot
    elif len(params_to_study) == 2:

        # Rows: parametersToStudy[0]
        # Cols: parametersToStudy[1]
        mean_res = np.zeros((
            len(params_and_their_ranges_to_be_varied[1]),
            len(params_and_their_ranges_to_be_varied[3])))
        mean_time_taken = np.zeros((
            len(params_and_their_ranges_to_be_varied[1]),
            len(params_and_their_ranges_to_be_varied[3])))

        if os.path.isfile(images):

            for i in range(mean_res.shape[0]):
                for j in range(mean_res.shape[1]):
                    mean_res[i, j] = search_results[0]["F1"][i + j]
                    mean_time_taken[i, j] = search_results[
                        0]["time_taken"][i + j]

        else:
            # Search targeted folders of images.

            # Sum the results of all images
            nr_of_images = 0
            for folder in search_results:
                for image in folder:
                    nr_of_images += 1
                    for i in range(mean_res.shape[0]):
                        for j in range(mean_res.shape[1]):
                            mean_res[i, j] += image["F1"][i + j]
                            mean_time_taken[i, j] += image["time_taken"][i + j]
            # Take average: divide the values by the number of images
            for i in range(mean_res.shape[0]):
                for j in range(mean_res.shape[1]):
                    mean_res[i, j] /= nr_of_images
                    mean_time_taken[i, j] /= nr_of_images

        # Determine the locations at which the bars start
        xpos = np.repeat(
            range(len(all_param_names_and_ranges[params_to_study[0]])),
            len(all_param_names_and_ranges[params_to_study[1]]))
        ypos = np.tile(
            range(len(all_param_names_and_ranges[params_to_study[1]])),
            len(all_param_names_and_ranges[params_to_study[0]]))
        zpos = np.zeros(len(xpos))  # height

        # Determine step sizes: width, depth, height
        dx = np.ones(len(xpos))
        dy = np.ones(len(ypos))
        dz = mean_res.flatten()
        dz_time = mean_time_taken.flatten()

        # Normalize dz to [0,1] for colormap
        dz_normalized = dz
        max_f1, min_f1 = np.max(dz_normalized), np.min(dz_normalized)
        dz_normalized = (dz_normalized - min_f1) / (max_f1 - min_f1)
        colors = pl.cm.cool(dz_normalized)

        # Normalize dzTime to [0,1] for colormap
        dz_time_normalized = dz_time
        time_max = np.max(dz_time_normalized)
        time_min = np.min(dz_time_normalized)
        dz_time_normalized = (dz_time_normalized - time_min) / (
        time_max - time_min)
        colors_time = pl.cm.cool(dz_time_normalized)

        # Plot scores
        fig = pl.figure(figsize=(13, 6), facecolor='white')
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')  # Plot time taken
        ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, alpha=0.6)
        ax2.bar3d(xpos, ypos, zpos, dx, dy, dz_time, color=colors_time,
                  alpha=0.6)
        ax1.set_xlabel(
            params_to_study[0], fontsize=14, fontweight='bold', color='k')
        ax2.set_xlabel(
            params_to_study[0], fontsize=14, fontweight='bold', color='k')
        ax1.set_ylabel(
            params_to_study[1], fontsize=14, fontweight='bold', color='k')
        ax2.set_ylabel(
            params_to_study[1], fontsize=14, fontweight='bold', color='k')
        ax1.set_zlabel('F1-score', fontsize=14, fontweight='bold', color='k')
        ax2.set_zlabel('Computation time (seconds)', fontsize=14,
                       fontweight='bold', color='k')
        ax1.set_zlim(0, 1)
        ax1.set_xticks(np.unique(xpos))
        ax1.set_yticks(np.unique(ypos))
        ax2.set_xticks(np.unique(xpos))
        ax2.set_yticks(np.unique(ypos))
        ax1.set_xticklabels(
            [str(x) for x in all_param_names_and_ranges[params_to_study[0]]])
        ax1.set_yticklabels(
            [str(x) for x in all_param_names_and_ranges[params_to_study[1]]])
        ax2.set_xticklabels(
            [str(x) for x in all_param_names_and_ranges[params_to_study[0]]])
        ax2.set_yticklabels(
            [str(x) for x in all_param_names_and_ranges[params_to_study[1]]])
        # Set viewing angles, 1st argument: z-height, 2nd: angle
        ax1.view_init(26, 60)
        ax2.view_init(26, 60)
        pl.tight_layout()
        # pl.draw()
        pl.show()


def svm_cost_in_classification():

    param_to_vary = "Cost"
    default_hog = cv2.HOGDescriptor()
    my_params = dict(
        _winSize=(32, 32),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9,
        _derivAperture=default_hog.derivAperture,
        _winSigma=default_hog.winSigma,
        _histogramNormType=default_hog.histogramNormType,
        _L2HysThreshold=default_hog.L2HysThreshold,
        _gammaCorrection=default_hog.gammaCorrection,
        _nlevels=default_hog.nlevels)
    hog = cv2.HOGDescriptor(**my_params)

    # Import data and extract HOG features
    train_data, train_classes, labels, ground_truth = importdata. \
        import_images_extract_hog_features(hog=hog,
                                           days=["day1", "day2", "day3"],
                                           save_annotations=False,
                                           n_samples=4000)

    # Divide into training and testing data
    for_testing = 2000
    for_training = 2000
    pos_ex_row_ix = np.where(train_classes == 1)[0]
    neg_ex_row_ix = np.where(train_classes == 0)[0]
    test_d = np.concatenate((train_data[pos_ex_row_ix[-for_testing:]],
                             train_data[neg_ex_row_ix[-for_testing:]]))
    train_d = np.concatenate((train_data[pos_ex_row_ix[0:for_training]],
                              train_data[neg_ex_row_ix[0:for_training]]))
    test_c = np.concatenate((train_classes[pos_ex_row_ix[-for_testing:]],
                             train_classes[neg_ex_row_ix[-for_testing:]]))
    train_c = np.concatenate((train_classes[pos_ex_row_ix[0:for_training]],
                              train_classes[neg_ex_row_ix[0:for_training]]))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build classifiers with different values and calculate their ROC
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Initialize dictionary of lists where resutls will be saved
    results = {"FPR": [], "TPR": [], "AUC": [], "fvl": [], "models": [],
               param_to_vary: []}
    cost = 10.0 ** (np.arange(-4, 5, 1))
    for c in cost:
        print "Current cost:", c

        # cost = 10.0**(np.arange(-2,3,1))
        model = svm.train(train_d, train_c, cost=c)

        # Calculate ROC
        fpr, tpr, roc_auc = svm.roc(model, train_d, train_c, test_d, test_c)

        results["FPR"].append(fpr)
        results["TPR"].append(tpr)
        results["AUC"].append(roc_auc)
        results["fvl"].append(train_data.shape[1])
        results["models"].append(model)
        results[param_to_vary].append(c)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Visualize
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    pl.close("all")
    fig = pl.figure(figsize=(13, 6), facecolor='none')
    ax1 = fig.add_subplot(1, 2, 1, adjustable='box', aspect=1.0)  # ROC
    ax2 = fig.add_subplot(1, 2, 2, adjustable='box',
                          aspect=1.0)  # HOG parameters table
    ax2.axis('off')

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Draw area to be zoomed in and lines to zoomed area.
    # This is done before ROC so that ROC curves will be on top of these.
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    start_coords = (0.0, 0.9)
    end_x = 0.1
    end_y = 1.0
    # Small rectangle
    ax1.add_patch(pl.Rectangle(start_coords, end_x, end_y, facecolor='white',
                               edgecolor='black'))

    # Lines going to big rectangle
    x_range = [start_coords[0], 0.3]
    y_range = [start_coords[1], 0.4]
    ax1.plot(x_range, y_range, '#cccccc')
    # Lines going to big rectangle
    x_range = [end_x, 0.6]
    y_range = [end_y, 0.7]
    ax1.plot(x_range, y_range, '#cccccc')

    rect = [0.3, 0.4, 0.3, 0.3]
    ax11 = utils.add_subplot_axes(ax1, rect)

    # Hide ticks
    ax11.get_xaxis().set_visible(False)
    ax11.get_yaxis().set_visible(False)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot ROC curve
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    width = 3
    line_styles = ["b", "g", "r", "c", "m", "y", "k", "#FF00FF", "#00FFFF",
                   "#FFFF00"]
    for i in range(len(results["AUC"])):

        if param_to_vary == "Cost":
            label_text = "Cost "

        ax1.plot(results["FPR"][i], results["TPR"][i], line_styles[i],
                 linewidth=width, label=(
                label_text + str(results[param_to_vary][i]) +
                "  " * (10 - len(str(results[param_to_vary][i]))) +
                "AUC %0.3f" % results["AUC"][i]))

        ax1.legend(loc="lower right", fontsize=12)

    ax1.plot([0, 1], [0, 1], 'k--')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.0])
    ax1.set_xlabel('False positive rate (FPR)')
    ax1.set_ylabel('True positive rate (TPR)')
    # pl.tight_layout()
    pl.show()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Draw zoomed in area
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    for i in range(len(results["AUC"])):
        ax11.plot(results["FPR"][i],
                  results["TPR"][i],
                  line_styles[i], linewidth=width)
    ax11.axis([0.0, 0.09, 0.92, 1.0])
    pl.show()
    pl.draw()
