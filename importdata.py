# -*- coding: utf-8 -*-

import os
import time
import glob
import uuid
import ConfigParser

import cv2
import numpy as np
from scipy import misc

import utils


"""
Imports training data and generates features

Arguments:
hog = OpenCV HOG class instance.
days = List of strings of folder names indicating that negative annotations
       can be automatically collected from the images inside these folders.
save_annotations = Boolean value determining if user wants to save all the
                   annotations as images to the hard drive after resizing.
n_samples = Optional integer.
            If provided: This many positive examples and
                         this many negative examples will be imported.
            Else: All examples will be imported.
"""


def import_images_extract_hog_features(hog, days, save_annotations,
                                       n_samples=None):

    # Initialize
    win_size_aspect_ratio = float(hog.winSize[0]) / hog.winSize[1]
    time_of_execution = str(time.time()).split('.')[0]
    utils.cls()  # Clear screen
    total_n_pos = 0
    total_n_neg = 0
    annotation_widths = []
    annotation_heights = []
    aspect_ratios = []
    training_examples = {}  # Dictionary for features
    # Structure:
    # Key: Dir (string)
    #  Key: positiveExamples
    #  Key: negativeExamples
    ground_truth = {}  # Dictionary for annotation coordinates
    # Structure:
    # Key: Dir
    #  Key: Image
    #   Key: positiveExamples
    #   Key: negativeExamples

    print "Importing annotations:"
    print "------------------------------------------------------------"

    # Input folder location and possible image types
    day_folders = utils.list_dirs(r".\trainWithThese")
    file_types = ["bmp", "jpg", "png"]

    # Loop through input folders (days)
    for k, folder in enumerate(day_folders):

        # Initialize new dictionaries
        training_examples[folder] = {}
        ground_truth[folder] = {}
        # Initialize new lists
        training_examples[folder]["positiveExamples"] = []
        training_examples[folder]["negativeExamples"] = []

        # Get list of images in the folder
        images = []
        for fileType in file_types:
            images = images + glob.glob(folder + "\*." + fileType)

        # Loop through images
        for i, im_file_name in enumerate(images):

            # Image name without file type extension
            im_name = (im_file_name[im_file_name.rfind("\\")+1:
                                    im_file_name.rfind(".")])

            # Initialize new dictionaries
            ground_truth[folder][im_name] = {}
            # Initialize new lists
            ground_truth[folder][im_name]["positiveExamples"] = []
            ground_truth[folder][im_name]["negativeExamples"] = []

            print "\n" + im_file_name + ":"

            # Open current image
            img = cv2.imread(im_file_name, 0)
            height, width = img.shape

            # Path to image's annotation file
            path_to_annotation_file = (os.path.splitext(im_file_name)[0] +
                                       "_annotations.ini")
            # If file does not exist --> go to next image
            if not os.path.isfile(path_to_annotation_file):
                continue
            # If file does exist --> open it
            config = ConfigParser.RawConfigParser()
            config.read(path_to_annotation_file)

            # Initialize
            positive_examples = []
            negative_examples = []

            # Create new black image where annotations are marked in white
            # so that negative examples can be gathered from the rest
            # of the locations
            marked_annotations = np.zeros(img.shape)

            # -----------------------------------------------
            # Gather annotated positive and negative examples
            # -----------------------------------------------

            # Go through every annotation and extract it from original image
            for section in config.sections():

                # Get coordinates
                ulc = int(float(config.get(section, "ulc")))
                ulr = int(float(config.get(section, "ulr")))
                lrc = int(float(config.get(section, "lrc")))
                lrr = int(float(config.get(section, "lrr")))

                # Make sure the section options is not not NaN or INF
                if np.isnan(ulc) or np.isnan(ulr) or \
                   np.isnan(lrc) or np.isnan(lrr):
                    break

                ann_height = lrr-ulr
                ann_width = lrc-ulc

                # ----------------------------------------------------
                # Crop and resize annotation
                # ----------------------------------------------------

                ann_aspect_ratio = (ann_height/ann_width)

                # If aspect ratio is correct, simply crop image as it is
                if ann_aspect_ratio == win_size_aspect_ratio:
                    annotation = img[ulr:lrr, ulc:lrc]
                    annotation = cv2.resize(annotation, hog.winSize)

                # If more width than height --> select height accordingly.
                # Aspect ratio is preserved
                elif ann_aspect_ratio < win_size_aspect_ratio:
                    pixels_extend = (win_size_aspect_ratio * ann_width) / 2
                    # If no room to extend height, resize and lose aspect ratio
                    if (ulr < np.floor(pixels_extend) or
                            lrr > height-np.ceil(pixels_extend)):
                        annotation = img[ulr:lrr, ulc:lrc]
                        annotation = cv2.resize(annotation, hog.winSize)
                    # If there is room to extend, then extend height
                    else:
                        annotation = img[ulr-np.int(np.floor(pixels_extend)):
                                         lrr+np.int(np.ceil(pixels_extend)),
                                         ulc:lrc]
                        annotation = cv2.resize(annotation, hog.winSize)

                # If more height than width --> select width accordingly.
                # Aspect ratio is preserved
                elif ann_height > ann_width:
                    pixels_extend = (win_size_aspect_ratio * ann_height) / 2
                    # If no room to extend width, resize and lose aspect ratio
                    if ulc < np.floor(pixels_extend) or \
                       lrc > width-np.ceil(pixels_extend):
                        annotation = img[ulr:lrr, ulc:lrc]
                        annotation = cv2.resize(annotation, hog.winSize)
                    # If there is room to extend, then extend width
                    else:
                        annotation = img[ulr:lrr,
                                         ulc-np.int(np.floor(pixels_extend)):
                                         lrc+np.int(np.ceil(pixels_extend))]
                        annotation = cv2.resize(annotation, hog.winSize)

                # ----------------------------------------------------
                # Calculate HOG features
                # ----------------------------------------------------

                feature = hog.compute(annotation)

                # ----------------------------------------------------
                # Save the feature
                # ----------------------------------------------------

                if "positiveExample" in section:

                    # Save size & aspect ratio
                    annotation_heights.append(ann_height)
                    annotation_widths.append(ann_width)
                    aspect_ratios.append(ann_aspect_ratio)

                    positive_examples.append(feature)

                    # Save coordinates to dictionary
                    ground_truth[folder][im_name]["positiveExamples"] \
                        .append([ulc, ulr, lrc-ulc, lrr-ulr])

                    if save_annotations:

                        folder = (r".\annotations_" + time_of_execution +
                                  "\\positive")
                        # Create the folder if it does not exist already
                        if not os.path.exists(folder):
                            os.makedirs(folder)

                        # Save the image
                        cv2.imwrite(
                            folder + "\\" +
                            folder[folder.rfind("\\")+1:] + "_" +
                            im_name + "_" +
                            str(uuid.uuid4().fields[-1]) + ".bmp", annotation)

                elif "negativeExample" in section:

                    negative_examples.append(feature)

                    # Save coordinates to dictionary
                    ground_truth[folder][im_name]["negativeExamples"] \
                        .append([ulc, ulr, lrc-ulc, lrr-ulr])

                    if save_annotations:

                        folder = (r".\annotations_" + time_of_execution +
                                  "\\negative")
                        # Create the folder if it does not exist already
                        if not os.path.exists(folder):
                            os.makedirs(folder)

                        # Save the image
                        cv2.imwrite(
                            folder + "\\" +
                            folder[folder.rfind("\\")+1:] + "_" +
                            im_name + "_" +
                            str(uuid.uuid4().fields[-1]) + ".bmp", annotation)

                else:
                    print ("WARNING: "
                           "Section without positive/negative identifier")

                # Mark handled annotation with white
                marked_annotations[ulr:lrr, ulc:lrc] = 1

            # ---------------------------------------------------------------
            # Gather more negative examples automatically from the background
            # ---------------------------------------------------------------

            # Gather only from images from these days, because on subsequent
            # day images do not have enough background space anyway and they
            # are not fully annotated, which could cause some of positive
            # examples ending up being marked as negative example.
            if any(day in folder for day in days):

                # Loop through image with step size = winSize
                for y in xrange(hog.winSize[0],
                                width-hog.winSize[0],
                                hog.winSize[0]+10):
                    for x in xrange(hog.winSize[1],
                                    height-hog.winSize[1],
                                    hog.winSize[1]+10):

                        # Crop area from binary image
                        current_area = marked_annotations[x:x+hog.winSize[1],
                                                          y:y+hog.winSize[0]]

                        # Check if currentArea is completely black,
                        # i.e., it is background
                        if np.sum(current_area) != 0:
                            continue

                        # Crop the same area from original image
                        cropped = img[x:x+hog.winSize[1],
                                      y:y+hog.winSize[0]]

                        # Extract feature
                        feature = hog.compute(cropped)
                        # Save the feature
                        negative_examples.append(feature)

                        # Save coordinates to dictionary
                        ground_truth[folder][im_name]["negativeExamples"] \
                            .append([x, y, hog.winSize[1], hog.winSize[0]])

                        if save_annotations:
                            folder = (r".\annotations_" + time_of_execution +
                                      "\\negative")
                            # Create the folder if it does not exist
                            if not os.path.exists(folder):
                                os.makedirs(folder)
                            # Save the image
                            misc.imsave(
                                folder + "\\" +
                                folder[folder.rfind("\\")+1:] + "_" +
                                im_name + "_" + str(uuid.uuid4().fields[-1]) +
                                ".bmp", cropped)

            training_examples[folder]["positiveExamples"].append(
                positive_examples)
            training_examples[folder]["negativeExamples"].append(
                negative_examples)

            print "Positive examples gathered from this image: %i" % len(
                positive_examples)
            print "Negative examples gathered from this image: %i" % len(
                negative_examples) + "\n"

            total_n_pos += len(positive_examples)
            total_n_neg += len(negative_examples)

    print "------------------------------------------------------------"
    print "Mean pos annotation width: %.2f +- %.2f px" % (
        np.mean(annotation_widths), np.std(annotation_widths))
    print "Mean pos annotation height: %.2f +- %.2f px" % (
        np.mean(annotation_heights), np.std(annotation_heights))
    print "Mean pos aspect ratio: %.2f +- %.2f" % (
        np.mean(aspect_ratios), np.std(aspect_ratios))
    print "Total number of positive annotations: %i" % total_n_pos
    print "Total number of negative annotations: %i" % total_n_neg
    print "Length of feature vector: ", len(feature)
    print "------------------------------------------------------------\n"

    # print "\nMerging the data, assigning classes, and assigning labels..."
    train_data = []
    train_classes = []
    labels = []
    label = 0
    for folder in training_examples.keys():

        for examples in training_examples[folder]["positiveExamples"]:
            train_data = train_data + examples
            train_classes = train_classes + ([1] * len(examples))
            labels = labels + ([label] * len(examples))

        for examples in training_examples[folder]["negativeExamples"]:
            train_data = train_data + examples
            train_classes = train_classes + ([0] * len(examples))
            # Let's assign random labels to negative examples so that each
            # day (label) will have also negative examples.
            # Images from latest days (labels) consist of only
            # positive examples and that is why no negative examples are not
            # collected from those days.
            labels = labels + np.random.randint(
                low=0, high=len(training_examples.keys()),
                size=len(examples)).tolist()
        label += 1

    # Squeeze removes unnecessary extra dimension
    train_data = np.squeeze(np.array(train_data))
    train_classes = np.array(train_classes)
    labels = np.array(labels)

    np.random.seed(54321)

    # Take equal amount of positives and negatives
    if isinstance(n_samples, int):
        # Shuffle negatives and maintain the same order in each array
        neg_ex_row_indices = np.where(train_classes == 0)[0]
        np.random.shuffle(neg_ex_row_indices)
        delete_these = neg_ex_row_indices[n_samples:]
        train_data = np.delete(train_data, delete_these, 0)
        train_classes = np.delete(train_classes, delete_these, 0)
        labels = np.delete(labels, delete_these, 0)
        # Shuffle positives and maintain the same order in each array
        pos_ex_row_indices = np.where(train_classes == 1)[0]
        np.random.shuffle(pos_ex_row_indices)
        delete_these = pos_ex_row_indices[n_samples:]
        train_data = np.delete(train_data, delete_these, 0)
        train_classes = np.delete(train_classes, delete_these, 0)
        labels = np.delete(labels, delete_these, 0)

    # Shuffle and maintain the same order in every array
    shuffled_order = range(train_data.shape[0])
    np.random.shuffle(shuffled_order)
    train_data = np.asarray([train_data[i, :] for i in shuffled_order])
    train_classes = np.asarray([train_classes[i] for i in shuffled_order])
    labels = np.asarray([labels[i] for i in shuffled_order])

    return train_data, train_classes, labels, ground_truth
