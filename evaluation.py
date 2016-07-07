# -*- coding: utf-8 -*-

import sys
import ConfigParser

import cv2
import numpy as np
from PIL import Image as PILImage

import filters


def visualize_confusion_matrix(img, detected, type_of_detection):
    detection_count = 0
    thickness = 2

    for i, detection in enumerate(detected):

        if len(detection) == 6 and detection[5] != type_of_detection:
            continue

        if type_of_detection == "truePositive":
            x, y, w, h = detection[4]
            colouring = (0, 255, 0)

        elif type_of_detection == "falsePositive":
            x, y, w, h = detection[4]
            colouring = (255, 0, 0)

        elif type_of_detection == "falseNegative":
            x, y, w, h = detection[2]
            colouring = (0, 0, 255)

        else:
            print "!!!"
            print "!!! Error: detection type not recognized!"
            print "!!!"
            sys.exit(2)

        x, y, w, h = int(x), int(y), int(w), int(h)

        cv2.rectangle(img, (x, y), (w, h), color=colouring,
                      thickness=thickness)

        # Draw running number inside each detection
        # cv2.putText(img, str(nrOf), (x+5,y+15), cv2.FONT_HERSHEY_SIMPLEX,
        # fontScale=0.42, color=colouring, thickness=thickness)

        detection_count += 1


def is_already_added(section_name, results, rectangle_type):

    if not results:  # List is empty
        return False

    if rectangle_type == "detection":
        for res in results:
            if res[3] == section_name:
                return True

    elif rectangle_type == "annotation":
        for res in results:
            if res[1] == section_name:
                return True

    return False


def assess_detections(path_to_detections_ini, path_to_truth_ini, im_path):

    # PAS metric threshold for considering detections as TP
    pas_threshold_for_tp = 0.3

    results = []

    n_tp = 0
    n_fp = 0
    n_fn = 0

    config_detections = ConfigParser.RawConfigParser()
    config_detections.read(path_to_detections_ini)

    config_truth = ConfigParser.RawConfigParser()
    config_truth.read(path_to_truth_ini)

    is_positive = np.array(
        ["positive" in config_name for config_name in config_truth.sections()])
    n_pos_annotations = np.where(is_positive == True)[0].shape[0]

    # Calculate overlap of each neighboring detection and annotation.
    # Store each comparison in a tuple and place it in a list.
    # List is chosen because it can be sorted in the next step.
    potential_matches = []

    for section2 in config_detections.sections():
        ulc2 = np.double(config_detections.get(section2, "ulc"))
        ulr2 = np.double(config_detections.get(section2, "ulr"))
        lrc2 = np.double(config_detections.get(section2, "lrc"))
        lrr2 = np.double(config_detections.get(section2, "lrr"))
        detection_coords = (ulc2, ulr2, lrc2, lrr2)

        detCenter = ((ulc2+lrc2) / 2, (ulr2+lrr2) / 2)

        for i, section1 in enumerate(config_truth.sections()):

            if "positive" not in section1:
                continue

            ulc1 = np.double(config_truth.get(section1, "ulc"))
            ulr1 = np.double(config_truth.get(section1, "ulr"))
            lrc1 = np.double(config_truth.get(section1, "lrc"))
            lrr1 = np.double(config_truth.get(section1, "lrr"))
            ref_coords = (ulc1, ulr1, lrc1, lrr1)

            ref_center = ((ulc1+lrc1) / 2, (ulr1+lrr1) / 2)

            # If detected rectangle center too far away from reference
            # rectangle center, do not calculate overlap and mark the
            # detection as false positive.
            # This check makes this nested loop much faster to compute.
            if (np.abs(ref_center[0] - detCenter[0]) > 200 or
                    np.abs(ref_center[1] - detCenter[1]) > 200):
                continue

            overlap = filters.get_overlap(
                ref_coords, detection_coords, hog=None,
                structure_of_data="corners")

            comparison = (overlap,  # overlap
                          section1,  # truthName
                          ref_coords,  # truthCoord
                          section2,  # detectionName
                          detection_coords)  # detection_coords

            potential_matches.append(comparison)

        if not is_already_added(section2, potential_matches, "detection"):
            false_pos = (0.0,  # overlap
                         "",  # truthName
                         [],  # truthCoord
                         section2,  # detectionName
                         detection_coords,  # detection_coords
                         "falsePositive")  # hitType
            n_fp += 1
            results.append(false_pos)

    # Sort list of comparisons according to the amount of overlap
    # First element of the list will have the most overlap
    potential_matches = sorted(potential_matches, key=lambda k: k[0],
                               reverse=True)

    # Preserve only the best unique detections
    for comparison in potential_matches:

        if is_already_added(comparison[3], results, "detection"):
            continue

        # Only one match for each annotation
        if (comparison[0] >= pas_threshold_for_tp and
                not is_already_added(comparison[1], results, "annotation")):
            comparison += ("truePositive",)  # Append tuple
            results.append(comparison)
            n_tp += 1

        else:
            fp = (comparison[0],
                  "",
                  [],
                  comparison[3],
                  comparison[4],
                  "falsePositive")  # Append tuple
            results.append(fp)
            n_fp += 1

    # Add false negatives: those annotations which are not already
    # added in the results
    for section in config_truth.sections():

        if "positive" not in section:
            continue

        if not is_already_added(section, results, "annotation"):
            ulc1 = np.double(config_truth.get(section, "ulc"))
            ulr1 = np.double(config_truth.get(section, "ulr"))
            lrc1 = np.double(config_truth.get(section, "lrc"))
            lrr1 = np.double(config_truth.get(section, "lrr"))
            ref_coords = (ulc1, ulr1, lrc1, lrr1)

            false_neg = (0.0,  # overlap
                         section,  # truthName
                         ref_coords,  # truthCoord
                         "",  # detectionName
                         [],  # detection_coords
                         "falseNegative")  # hitType
            n_fn += 1
            results.append(false_neg)

    # Open current image
    img = cv2.imread(im_path)

    visualize_confusion_matrix(img, results, "truePositive")
    visualize_confusion_matrix(img, results, "falsePositive")
    visualize_confusion_matrix(img, results, "falseNegative")

    print "True positives  =", n_tp
    print "false positives =", n_fp
    print "false negatives =", n_fn

    # True Positive Rate
    if n_tp == 0 and n_fn == 0:
        TPR = 1.0
    else:
        TPR = np.double(n_tp) / np.double(n_tp + n_fn)

    # False Positive Rate
    FPR = np.double(n_fp) / np.double(n_pos_annotations)

    # Precision
    if np.double(n_tp + n_fp) > 0.0:
        precision = np.double(n_tp) / np.double(n_tp + n_fp)
    else:
        precision = 0.0

    # F1 score
    # Do not divide by zero
    if np.double(n_tp + n_fp + n_fn) > 0.0:
        F1 = 2.0 * n_tp / (2.0 * n_tp + n_fp + n_fn)
        print "F1 score = %.2f" % F1
    else:
        F1 = 0.0
        print "F1 score = 0"

    # F05 score
    if np.double(precision + TPR) > 0.0:
        F05 = np.double((0.5 ** 2.0 + 1.0) * precision * TPR) / np.double(
            precision + TPR)
        print "F05 score = %.2f" % F05
    else:
        F05 = 0
        print "F05 score = 0"

    # F09 score
    if np.double(precision + TPR) > 0.0:
        F09 = np.double((0.9 ** 2.0 + 1.0) * precision * TPR) / np.double(
            precision + TPR)
        print "F09 score = %.2f" % F09
    else:
        F09 = 0
        print "F09 score = 0"

    if n_tp + n_fn != n_pos_annotations:
        print "\n!!!"
        print ("!!! Error: different amount of annotations and "
               "matched detections!")
        print "!!! TP: %i, FN: %i, TP + FN: %i" % (n_tp, n_fn, n_tp + n_fn)
        print "!!! Nr of annotations:", n_pos_annotations
        print "!!!\n"
    if n_tp + n_fp != len(config_detections.sections()):
        print "!!!"
        print ("!!! Error: different amount of initial detections and "
               "matched detections!")
        print "!!! TP: %i, FP: %i, TP + FP: %i" % (
            n_tp, n_fp, n_tp + n_fp)
        print "!!! Nr of initial detection:", len(config_detections.sections())
        print "!!!"

    return n_tp, n_fp, n_fn, TPR, FPR, F1, F05, F09, n_pos_annotations, img


"""

This function calculates Bivariate Similarity Index.

Inputs:
path_to_truth_im = string, path to binary ground truth image
path_to_estim_im = string, path to binary estimated image

Outputs:
tet = float between [0,1]
tee = float between [0,1]
d_seq = Segmentation distance (float)

"""


def bivariate_similarity_index(path_to_truth_im, path_to_estim_im):

    # t = Truth
    # e = Estimate

    # a = The number of pixels where the value of t and e is (1,1), i.e., TP
    # b = The number of pixels where the value of t and e is (0,1), i.e., FP
    # c = The number of pixels where the value of t and e is (1,0), i.e., FN
    # d = The number of pixels where the value of t and e is (0,0), i.e., TN

    # Open and identify ground truth & estimated image file
    im_truth = PILImage.open(path_to_truth_im)
    im_estim = PILImage.open(path_to_estim_im)

    # Save dimensions
    width, height = im_truth.size

    # Read from the file
    pix_truth = im_truth.load()
    pix_estim = im_estim.load()

    # Initialize all to zero
    a, b, c, d, t, e = [0.0] * 6

    # Loop through every pixel
    for y in range(height):
        for x in range(width):

            # Calculate a,b,c,d.
            # For similarity and difference metrics, see
            # "A Survey of Binary Similarity and Distance Measures":
            # http://www.iiisci.org/journal/CV$/sci/pdfs/GS315JG.pdf

            if pix_truth[x, y] == 255 and pix_estim[x, y] == 255:
                a += 1
            elif pix_truth[x, y] == 0 and pix_estim[x, y] == 255:
                b += 1
            elif pix_truth[x, y] == 255 and pix_estim[x, y] == 0:
                c += 1
            elif pix_truth[x, y] == 0 and pix_estim[x, y] == 0:
                d += 1

            # Calculate t and e for Bivariate Similarity Index.
            # For more information, see
            # "Comparison of segmentation algorithms for
            #  fluorescence microscopy images of cells":
            # http://onlinelibrary.wiley.com/doi/10.1002/cyto.a.21079/abstract

            if pix_truth[x, y] == 255:
                t += 1
            if pix_estim[x, y] == 255:
                e += 1

    # Bivariate Similarity Index
    tet = a / t
    tee = a / e

    # Segmentation distance d_seg
    d_seg = np.sqrt((1-tet)**2 + (1-tee)**2)

    return tet, tee, d_seg
