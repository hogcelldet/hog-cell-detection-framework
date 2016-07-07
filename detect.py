# -*- coding: utf-8 -*-


import os
import glob
import time
import ConfigParser

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.misc import imsave

import utils
import filters
import evaluation


"""

This function takes care of looping through image(s).

"""


def image_sweep(hog, svm, sliding_window_method, sliding_window_params,
                params_and_their_ranges_to_be_varied, images,
                output_folder, save_im, check_for_break_condition1=False,
                check_for_break_condition2=False,
                save_detection_binary_images=False):

    # Initialize output list
    results = []

    # Set SVM weights as detector.
    # Bias will be added to hitThreshold parameter.
    w = svm.coef_[0]
    hog.setSVMDetector(w)
    
    # Create output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    if os.path.isfile(images):
        # Single image search

        # Open image
        im_file_name = images
        im = cv2.imread(im_file_name, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        
        results.append(parameter_sweep(
                    hog, svm, sliding_window_method, sliding_window_params,
                    params_and_their_ranges_to_be_varied, output_folder,
                    save_im, check_for_break_condition1,
                    check_for_break_condition2, save_detection_binary_images,
                    im, im_file_name))

    else:
        # Search is targeting folders of images.
        # Expecting that each folder contains images for single day.
        
        # Input folder location and possible image types
        list_of_day_dirs = utils.list_dirs(images)
        file_types = ["bmp", 'jpg', 'png']
        
        # Loop through folders (days)
        for day, directory in enumerate(list_of_day_dirs):
            
            # Get list of images in the folder
            image_list = []
            for file_type in file_types:
                image_list = image_list + glob.glob(directory + "\*." +
                                                    file_type)
            
            folder_results = []
            
            # Loop through images
            for i, im_file_name in enumerate(image_list):
                
                print "\nProcessing " + directory[directory.rfind("\\")+1:] + \
                    " image " + str(i+1) + "/" + str(len(image_list)) + "..."
                print "----------------------------"
                
                # Open image
                im = cv2.imread(im_file_name, cv2.CV_LOAD_IMAGE_GRAYSCALE)

                folder_results.append(parameter_sweep(
                    hog, svm, sliding_window_method, sliding_window_params,
                    params_and_their_ranges_to_be_varied, output_folder,
                    save_im, check_for_break_condition1,
                    check_for_break_condition2, save_detection_binary_images,
                    im, im_file_name))

            results.append(folder_results)
            
    return results


"""

This function takes care of looping through values of given parameter(s).

"""


def parameter_sweep(hog, svm, sliding_window_method, sliding_window_params,
                    params_and_their_ranges_to_be_varied, output_folder,
                    save_im, check_for_break_condition1,
                    check_for_break_condition2, save_detection_binary_images,
                    im, im_file_name):
                                
    # Initialize dictionary of lists where results will be saved
    results = dict()
    results["im_name"] = []  # Name of image
    results["nr_of_cells_truth"] = []  # True number of cells
    results["TP"] = []  # Number of True Positives
    results["FP"] = []  # Number of False Positives
    results["FN"] = []  # Number of False Negatives
    results["FPR"] = []  # False Positive Rate
    results["TPR"] = []  # True Positive Rate
    results["F1"] = []  # F1 score
    results["F05"] = []  # F05 score
    results["F09"] = []  # F09 score
    results["time_taken"] = []  # Time taken to detect cells
    results["varied_param_1_and_val"] = []  # Parameter which is varied
    # Parameter which stays steady (grid-search)
    results["varied_param_2_and_val"] = []
    results["w"] = []  # Confidence for detections
    results["all_params"] = []  # All detection parameters

    if len(params_and_their_ranges_to_be_varied) == 0:
        # If there are no parameter values to be searched,
        # use predefined values

        varied_param_1_and_val = []
        varied_param_2_and_val = []

        results, stop_searching = detect_and_analyze(
            hog, svm, sliding_window_method, sliding_window_params,
            output_folder, save_im, check_for_break_condition1,
            check_for_break_condition2,
            save_detection_binary_images, im, im_file_name, results,
            varied_param_1_and_val, varied_param_2_and_val,
            iteration=0)

    elif len(params_and_their_ranges_to_be_varied) == 2:
        # One parameter values are searched

        for i in range(len(params_and_their_ranges_to_be_varied[1])):

            print "\nCurrent " + params_and_their_ranges_to_be_varied[0] + \
                ":", params_and_their_ranges_to_be_varied[1][i]

            varied_param_1_and_val = [
                params_and_their_ranges_to_be_varied[0],
                params_and_their_ranges_to_be_varied[1][i]]
            varied_param_2_and_val = []

            results, stop_searching = detect_and_analyze(
                hog, svm, sliding_window_method, sliding_window_params,
                output_folder, save_im, check_for_break_condition1,
                check_for_break_condition2,
                save_detection_binary_images, im, im_file_name, results,
                varied_param_1_and_val, varied_param_2_and_val,
                iteration=i)

            if stop_searching:
                break

    elif len(params_and_their_ranges_to_be_varied) == 4:
        # Two parameter values are searched (grid-search)
        
        for i in range(len(params_and_their_ranges_to_be_varied[3])):

            varied_param_2_and_val = [
                params_and_their_ranges_to_be_varied[2],
                params_and_their_ranges_to_be_varied[3][i]]

            print "\nSetting " + varied_param_2_and_val[0] + \
                " to " + str(varied_param_2_and_val[1]) +  \
                " and looping through " + \
                params_and_their_ranges_to_be_varied[0]
            
            # Vary detection parameter and save results
            for j in range(len(params_and_their_ranges_to_be_varied[1])):
                
                varied_param_1_and_val = [
                    params_and_their_ranges_to_be_varied[0],
                    params_and_their_ranges_to_be_varied[1][j]]

                print "\nCurrent " + varied_param_1_and_val[0] + \
                    ":", varied_param_1_and_val[1]
                
                results, stop_searching = detect_and_analyze(
                    hog, svm, sliding_window_method, sliding_window_params,
                    output_folder, save_im, check_for_break_condition1,
                    check_for_break_condition2,
                    save_detection_binary_images, im, im_file_name, results,
                    varied_param_1_and_val, varied_param_2_and_val,
                    iteration=i)
                                          
                if stop_searching:
                    break
                
    else:
        print "!!!"
        print ("!!! Warning: Invalid params_and_their_ranges_to_be_varied "
               "value. It must have length of 0, 2 or 4.")

    return results


"""

This function is responsible for single detection task with given parameters.

"""


def detect_and_analyze(hog, svm, sliding_window_method, sliding_window_params,
                       output_folder, save_im, check_for_break_condition1,
                       check_for_break_condition2,
                       save_detection_binary_images, im, im_file_name, results,
                       varied_param_1_and_val, varied_param_2_and_val,
                       iteration):

    if varied_param_1_and_val:
        # If parameter 1 is to be varied,
        # set its value in default parameter values
        hog, sliding_window_params = set_search_params(
            hog, svm, sliding_window_method, sliding_window_params,
            varied_param_1_and_val)

    if varied_param_2_and_val:
        # If parameter 2 is to be varied,
        # set its value in default parameter values
        hog, sliding_window_params = set_search_params(
            hog, svm, sliding_window_method, sliding_window_params,
            varied_param_2_and_val)
                                             
    # Generate settings string to be used in file name
    if sliding_window_method == "detectMultiScale":
        settings = (
            "_iter_" + str(iteration) +
            "_nlev_" + str(hog.nlevels) +
            "_hT_" + str(sliding_window_params["hitThreshold"]) +
            "_wS_" + str(sliding_window_params["winStride"]) +
            "_padding_" + str(sliding_window_params["padding"]) +
            "_scale_" + str(sliding_window_params["scale"]) +
            "_fT_" + str(sliding_window_params["finalThreshold"]) +
            "_mS_" + str(sliding_window_params["useMeanshiftGrouping"]))
    else:
        settings = (
            "_iter_" + str(iteration) +
            "_hitThreshold_" + str(sliding_window_params["hitThreshold"]) +
            "_winStride_" + str(sliding_window_params["winStride"]))

    # Run sliding window procedure
    found, time_taken, w = sliding_window(
        hog, im, sliding_window_method, sliding_window_params,
        filter_detections=False)
                          
    # Image name without path or file extension
    im_name = im_file_name[im_file_name.rfind("\\")+1:im_file_name.rfind(".")]
    
    # Detection binary images are saved when Bivariate Similarity Index (BSI)
    # is calculated.
    if save_detection_binary_images:
        # Create new black image where detections are marked
        binary_im = Image.new("L", [im.shape[1], im.shape[0]], "black")
        for x, y, w, h in found:
            # Mark rectangle detections with white
            draw = ImageDraw.Draw(binary_im)
            draw.rectangle([x, y, x+w, y+h],  fill=255)
        # Determine the image day
        day = im_file_name[
              im_file_name[:im_file_name.rfind("\\")].rfind(
                  "\\")+4:im_file_name.rfind("\\")
              ]
        # Save classification image
        estim_dir = output_folder + '\\detectionBinaries'
        child_of_estim_dir = estim_dir + "\\" + day + "\\"
        if not os.path.exists(child_of_estim_dir):
            os.makedirs(child_of_estim_dir)
        imsave(child_of_estim_dir + im_name + "_day" + day + ".png", binary_im)
    
    # Save the results in .INI
    path_to_detections_ini = output_folder+"\\"+im_name+settings+".ini"
    save_ini(found, path_to_detections_ini, sliding_window_method,
             im_file_name, hog)
    
    # Determine ground truth .INI file name
    path_to_truth_ini = \
        im_file_name[:im_file_name.rfind(".")]+"_annotations.ini"
    
    # Analyze the results, build confusion matrix
    TP, FP, FN, TPR, FPR, F1, F05, F09, nrOfCellsTruth, imWithDetections = \
        evaluation.assess_detections(
            path_to_detections_ini, path_to_truth_ini, im_file_name)
    
    # Save image with annotations
    if save_im:
        which_day = im_file_name[:im_file_name.rfind("\\")][-4:]
        imsave(output_folder + "\\" + im_name + settings + "_" + which_day +
               ".png", imWithDetections)

    # Save results
    results["im_name"].append(im_file_name)
    results["nr_of_cells_truth"].append(nrOfCellsTruth)
    results["TP"].append(TP)
    results["FP"].append(FP)
    results["FN"].append(FN)
    results["FPR"].append(FPR)
    results["TPR"].append(TPR)
    results["F1"].append(F1)
    results["F05"].append(F05)
    results["F09"].append(F09)
    results["time_taken"].append(time_taken)
    results["varied_param_1_and_val"].append(varied_param_1_and_val)
    results["varied_param_2_and_val"].append(varied_param_2_and_val)
    results["w"].append(w)
    results["all_params"].append(sliding_window_params)

    stop_searching = False
    if check_for_break_condition1:
        if TPR >= 1.0:
            print "Ending the search because TPR >= 1.0"
            stop_searching = True
        elif FPR >= 1.0:
            print "Ending the search because FPR >= 1.0"
            stop_searching = True
        elif len(results["F1"]) > 3:
            # Make sure that we have gone beyond max F1
            if np.max(results["F1"]) > 0.6:
                if (results["F1"][-1] < results["F1"][-3]) and\
                        (results["F1"][-1] <= 0.6):
                    print "\nEnding the search because 3 iterations " + \
                        "in a row number F1-score has been decreasing and " + \
                        "F1-score is now <= 0.6"
                    stop_searching = True
        elif len(results["F1"]) > 3:
            if results["F1"][-1] == 0.0 and \
               results["F1"][-2] == 0.0 and \
               results["F1"][-3] == 0.0:
                print "\nEnding the search because 3 iterations " + \
                    "in a row number F1-score has been 0.0"
                stop_searching = True

    if check_for_break_condition2:
        nr_of_iters = 4
        if len(results["FN"]) > nr_of_iters and \
           results["FN"][-(nr_of_iters+1)] < results["FN"][-nr_of_iters] < \
           results["FN"][-(nr_of_iters-1)] < results["FN"][-(nr_of_iters-2)]:
            print ("\nEnding the search because the number of false negatives "
                   "has been increasing" + str(nr_of_iters) +
                   " iterations in a row")
            # Delete the last results from every variable
            for key in results.keys():
                results[key] = results[key][:-nr_of_iters]
            stop_searching = True

    return results, stop_searching


"""

This function initializes parameters correctly for hog.detect and
hog.detectMultiScale function.

"""


def set_search_params(hog, svm, sliding_window_method, sliding_window_params,
                      param_to_set):

    param_name, param_val = param_to_set
    
    # In case of hitThreshold being set, add value to bias.
    # hitThreshold parameter is included in both sliding_window_method.
    # In scikit-learn's SVM, bias is -model.intercept_[0]
    if param_name == "hitThreshold":
        sliding_window_params[param_name] =\
            -svm.intercept_[0] + param_val
    
    # Set parameters for detectMultiScale -function
    elif sliding_window_method == "detectMultiScale":
        
        # Nlevels parameter has to be changed when initializing 
        # HOG class instance
        if param_name == "nlevels":
            my_params = dict(
                _winSize=hog.winSize,
                _blockSize=hog.blockSize,
                _blockStride=hog.blockStride,
                _cellSize=hog.cellSize,
                _nbins=hog.nbins,
                _derivAperture=hog.derivAperture,
                _winSigma=hog.winSigma,
                _histogramNormType=hog.histogramNormType,
                _L2HysThreshold=hog.L2HysThreshold,
                _gammaCorrection=hog.gammaCorrection,
                _nlevels=param_val
            )

            hog = cv2.HOGDescriptor(**my_params)
            
            w = svm.coef_[0]
            hog.setSVMDetector(w)

        # Other parameters can be changed without initializing new HOG
        # class instance (like it has to initialized with nlevels)
        else:
            sliding_window_params[param_name] = param_val
    
    # Set parameters for detect -function
    elif sliding_window_method == "detect":
        
        # Nlevels parameter has to be changed when initializing 
        # HOG class instance
        if param_name == "nlevels":
            my_params = dict(
                _winSize=hog.winSize,
                _blockSize=hog.blockSize,
                _blockStride=hog.blockStride,
                _cellSize=hog.cellSize,
                _nbins=hog.nbins,
                _derivAperture=hog.derivAperture,
                _winSigma=hog.winSigma,
                _histogramNormType=hog.histogramNormType,
                _L2HysThreshold=hog.L2HysThreshold,
                _gammaCorrection=hog.gammaCorrection,
                _nlevels=param_val
            )
            hog = cv2.HOGDescriptor(**my_params)
            w = svm.coef_[0]
            hog.setSVMDetector(w)
            
        else:
            sliding_window_params[param_name] = param_val
                   
    return hog, sliding_window_params


"""

This function saves detections to .ini file

Inputs:
found = List of detections returned by hog.detect or hog.detectMultiScale.
pathToDetectionsIni = String describing the path and filename of the ini file.
searchMethod = "detect" or "detectMultiScale", specifying the function type,
                which produced detections.
imFileName = String of image name from which objects were detected.
             Will be usedin .ini section names.
hog = OpenCV hogDescriptor.

"""


def save_ini(found, path_to_detections_ini, search_method, im_file_name, hog):
    
    # Create empty file & open it
    config = ConfigParser.RawConfigParser()
    if not os.path.isfile(path_to_detections_ini):
        # If you receive IOError: [Errno 2] here, it most probably
        # means you have too long file name
        open(path_to_detections_ini, "a").close()
        config.read(path_to_detections_ini)
    
    # Use index in secName instead of datetime because datetime is 
    # too inaccurate/slow to generate unique names for sections.
    index = 1
    
    for detection in found:
        # Both detectMultiScale & detect return
        # upper left corner coordinates of detections
        ulc = detection[0]
        ulr = detection[1]
        # detectMultiScale returns also width & height of detections
        if search_method == "detectMultiScale":
            w, h = detection[2:4]
        # detect does not return width & height -->
        # set them the same as winSize
        else:  # search_method == "detect"
            w, h = hog.winSize
        # Name of section
        sec_name = (os.path.basename(im_file_name) + "_positiveExample_" +
                    str(index))
        # Create section
        if not config.has_section(sec_name):
            config.add_section(sec_name)
        # Set section keys (properties)
        config.set(sec_name, "date", time.asctime())
        config.set(sec_name, "ulc", ulc)
        config.set(sec_name, "ulr", ulr)
        config.set(sec_name, "lrc", ulc + w)
        config.set(sec_name, "lrr", ulr + h)
        index += 1
    
    # Write result to file
    with open(path_to_detections_ini, "w") as outfile:
        config.write(outfile)


"""

This function runs the sliding window procedure and filters out detections.

Inputs:
hog = OpenCV HOGDescriptor class instance.
img = Source image. Preferably opened with OpenCV to ensure compatibility.
searchMethod = "detect" or "detectMultiScale", specifying function type.
params = Dictionary of function parameters.
filterDetections = Boolean, specifying whether to filter detections or not.

Outputs:
detections, computationTime, detectionWeights

"""


def sliding_window(hog, img, search_method, params, filter_detections):
    
    # Start the clock
    start_time = time.time()
    
    if search_method == "detectMultiScale":
        found, w = hog.detectMultiScale(img, **params)
    
    elif search_method == "detect":
        found, w = hog.detect(img, **params)
    
    else:
        print "!!!"
        print "!!! Error: searchMethod argument not recognized!"
        print "!!!"
        return
    
    # Stop the clock
    time_taken = time.time()-start_time  # In seconds
    print "Time taken: %i min %.1f seconds" % (
        np.floor(time_taken/60), time_taken-(60*np.floor(time_taken/60))
    )
    
    print "Nr of initial detections: %i" % len(found)
    
    if filter_detections:
        found_filtered_1 = []
        found_filtered_2 = []
        # print ("Filtering detections "
        #        "(inside each other & too much overlap)...")
        
        # Filter detections inside each other
        for ri, r in enumerate(found):
            for qi, q in enumerate(found):
                # Filter out detection which has detection inside
                if ri != qi and filters.is_inside(q, r, []):
                    # Delete also its weight
                    np.delete(w, ri)
                    break
                # Pass a detection through this filter if it is does not
                # have any detections inside 
                elif qi == len(found)-1:
                    found_filtered_1.append(r)
        # print "len(found_filtered_1): %i" % len(found_filtered_1)
        
        # Filter overlapping detections            
        for ri, r in enumerate(found_filtered_1):
            for qi, q in enumerate(found_filtered_1):
                # Filter out detection which overlap too much
                if ri != qi and filters.get_overlap(r, q, [], None) > 0.8:
                    # Delete also its weight
                    np.delete(w, ri)
                    break
                # Pass a detection through this filter if it is not
                # overlapping too much with any of the other detections 
                elif qi == len(found_filtered_1)-1:
                    found_filtered_2.append(r)   
        # print "len(found_filtered_2): %i" % len(found_filtered_2)
        
        print "Nr of detections after filtering: %i" % len(found_filtered_2)
        return found_filtered_2, time_taken, w
        
    else:
        return found, time_taken, w
