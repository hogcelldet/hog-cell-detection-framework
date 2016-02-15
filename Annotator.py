# -*- coding: utf-8 -*-

"""

Annotation creation/viewing app.
The coordinates of annotations are saved in INI format.

Left mouse button click + drag: Create new annotation.
Right mouse button click: Delete annotation.

Tab key: Change between annotating positive and negative examples.

z key: Activate/deactivate zooming mode.
       Left mouse button click + drag to select area to be zoomed in.

Space: Move to the next image.
Esc: Exit the program.

Requires the images to be stored in sub-folder(s) of a folder.
Folder structure, for example:

images
|
|
|\
| \
|  day1
|  |
|  |\
|  | image1.jpg
|  |\
|  | image2.jpg
|   \
|    image3.jpg
|
 \
  \
   day2
   |
   |\
   | image4.jpg
    \
     image5.jpg


"""


import os
import sys
import time
import glob
import copy
import Tkinter
import datetime
import tkFileDialog
import ConfigParser

import cv2
import numpy as np
import easygui as eg
from PIL import Image as PILImage

import utils


# -----------------------------------------------------------------------------


# Show smaller versions of images according to this factor
decimation_factor = 1.2
dragging = False


class Mode:
    def __init__(self):
        pass
    annotate_original = 1
    annotate_zoomed = 2
    delete_from_original = 3
    delete_from_zoomed = 4
    select_area_to_zoom_in = 5
    continue_to_next_im = 6


class AnnotationType:
    def __init__(self):
        pass
    positive_example = 1
    negative_example = 2


# -----------------------------------------------------------------------------


def draw_border(im, border_width, border_color):

    im_height, im_width = im.shape[:2]

    # Left border
    im[0:im_height, 0:border_width, 0] = border_color[0]
    im[0:im_height, 0:border_width, 1] = border_color[1]
    im[0:im_height, 0:border_width, 2] = border_color[2]

    # Right border
    im[0:im_height, im_width - border_width:im_width, 0] = border_color[0]
    im[0:im_height, im_width - border_width:im_width, 1] = border_color[1]
    im[0:im_height, im_width - border_width:im_width, 2] = border_color[2]

    # Top border
    im[0:border_width, 0:im_width, 0] = border_color[0]
    im[0:border_width, 0:im_width, 1] = border_color[1]
    im[0:border_width, 0:im_width, 2] = border_color[2]

    # Bottom border
    im[im_height - border_width:im_height, 0:im_width, 0] = border_color[0]
    im[im_height - border_width:im_height, 0:im_width, 1] = border_color[1]
    im[im_height - border_width:im_height, 0:im_width, 2] = border_color[2]

    return im


# -----------------------------------------------------------------------------


def mouse_handler(event, x_coord, y_coord, flags, param):
    global dragging
    global start_point
    global end_point
    global mode
    global original_with_annotations
    global zoomed_image
    global caption

    # Create new annotation by clicking left mouse button and dragging
    if event == cv2.cv.CV_EVENT_LBUTTONDOWN and not dragging:
        start_point = (x_coord, y_coord)
        dragging = True

    if event == cv2.cv.CV_EVENT_LBUTTONUP and dragging:
        end_point = (x_coord, y_coord)
        dragging = False

    if event == cv2.cv.CV_EVENT_MOUSEMOVE and dragging:

        if mode == Mode.annotate_original:
            img_copy = copy.copy(original_with_annotations)

            if annotation_type == AnnotationType.positive_example:
                cv2.rectangle(img_copy, start_point, (x_coord, y_coord),
                              (0, 255, 0))
            else:
                cv2.rectangle(img_copy, start_point, (x_coord, y_coord),
                              (0, 0, 255))

            cv2.imshow(caption, img_copy)

        if mode == Mode.select_area_to_zoom_in:
            img_copy = copy.copy(original_with_annotations)
            img_copy = draw_border(img_copy, 5, (0, 255, 255))
            cv2.rectangle(img_copy, start_point, (x_coord, y_coord),
                          (0, 255, 255))
            cv2.imshow(caption, img_copy)

        if mode == Mode.annotate_zoomed:
            zoomed_img_copy = copy.copy(zoomed_image)
            zoomed_img_copy = draw_annotations(zoomed_img_copy)

            if annotation_type == AnnotationType.positive_example:
                cv2.rectangle(zoomed_img_copy, start_point, (x_coord, y_coord),
                              (0, 255, 0))
            else:
                cv2.rectangle(zoomed_img_copy, start_point, (x_coord, y_coord),
                              (0, 0, 255))

            cv2.imshow(caption, zoomed_img_copy)

    # If right mouse button is clicked, delete annotations in that coordinate
    if event == cv2.cv.CV_EVENT_RBUTTONUP:
        end_point = (x_coord, y_coord)
        dragging = False

        if zoomed_image is None:
            print "Delete from original"
            mode = Mode.delete_from_original
        else:
            print "Delete from zoomed"
            mode = Mode.delete_from_zoomed


# -----------------------------------------------------------------------------


def draw_annotations(im):
    global zoomed_coords
    global zoom_rate

    # Go through every section and draw it
    for ini_section in config.sections():

        ulc_coord = int(
                float(config.get(ini_section, "ulc")) / decimation_factor)
        ulr_coord = int(
                float(config.get(ini_section, "ulr")) / decimation_factor)
        lrc_coord = int(
                float(config.get(ini_section, "lrc")) / decimation_factor)
        lrr_coord = int(
                float(config.get(ini_section, "lrr")) / decimation_factor)

        # Make sure the section options is not not NaN or INF
        if np.isnan(ulc_coord) or np.isnan(ulr_coord) or \
                np.isnan(lrc_coord) or np.isnan(lrr_coord):
            break

        # Draw rectangle to full-sized image (use coordinates as they come)
        if (mode == Mode.annotate_original or
                mode == Mode.select_area_to_zoom_in):

            if "positiveExample" in ini_section:
                cv2.rectangle(im, (ulc_coord, ulr_coord),
                                  (lrc_coord, lrr_coord), (0, 255, 0))
            else:
                cv2.rectangle(im, (ulc_coord, ulr_coord),
                                  (lrc_coord, lrr_coord), (0, 0, 255))

        # Draw rectangle to zoomed-in image (use modified coordinates)
        elif mode == Mode.annotate_zoomed:

            # Check that section is inside zoomed area
            if (ulc_coord <= zoomed_coords[2] and
                    ulr_coord <= zoomed_coords[3] and
                    lrc_coord >= zoomed_coords[0] and
                    lrr_coord >= zoomed_coords[1]):

                # Correct coordinates relative to zoomed area
                lrc_coord = (lrc_coord - ulc_coord) * zoom_rate + \
                            (ulc_coord - zoomed_coords[0]) * zoom_rate
                lrr_coord = (lrr_coord - ulr_coord) * zoom_rate + \
                            (ulr_coord - zoomed_coords[1]) * zoom_rate
                ulc_coord = (ulc_coord - zoomed_coords[0]) * zoom_rate
                ulr_coord = (ulr_coord - zoomed_coords[1]) * zoom_rate

                if "positiveExample" in ini_section:
                    cv2.rectangle(im, (ulc_coord, ulr_coord),
                                      (lrc_coord, lrr_coord), (0, 255, 0))
                else:
                    cv2.rectangle(im, (ulc_coord, ulr_coord),
                                      (lrc_coord, lrr_coord), (0, 0, 255))

    return im


# -----------------------------------------------------------------------------


if __name__ == "__main__":

    # Exchanges information between mouse event handler and main function
    global annotation_type
    annotation_type = AnnotationType.positive_example
    caption = ""

    zoomed_coords = None
    zoomed_image = None
    zoom_rate = 4  # How much do you want to zoom?

    # Select input folder dialog
    root = Tkinter.Tk()
    input_dir = tkFileDialog.askdirectory(
        parent=root,
        initialdir=os.path.dirname(os.path.abspath(__file__)),
        title=("Select folder containing folders of "
               "images and their annotations"))
    if input_dir != "":
        list_of_day_dirs = utils.list_dirs(input_dir)
    else:
        eg.msgbox("Invalid folder! Press OK to exit.", title="Invalid folder")
        sys.exit()

    # Selected folder should have subfolders holding microscope images
    # and one annotation file for each image
    image_file_types = ["bmp", "jpg", "png"]
    annotation_file_type = "ini"

    # Loop through folders
    for index, directory in enumerate(list_of_day_dirs):

        # Get list of images in the folder
        image_file_list = []
        for file_type in image_file_types:
            image_file_list = image_file_list + \
                              glob.glob(directory + "\*." + file_type)

        # Get list of annotation files in the folder
        annotation_file_list = glob.glob(
            directory + "\*." + annotation_file_type)

        # Loop through images
        for i, path_to_image in enumerate(image_file_list):

            # Image name without file type extension
            im_name = path_to_image[path_to_image.rfind("\\") + 1:
                                    path_to_image.rfind(".")]

            # If new image, open new window
            if caption != path_to_image:
                cv2.destroyAllWindows()
            caption = path_to_image

            img = PILImage.open(path_to_image)
            image = np.array(img.convert('RGB'))
            image = cv2.resize(
                image,
                (int(image.shape[1] / decimation_factor),
                 int(image.shape[0] / decimation_factor)))

            # New config parser
            config = ConfigParser.RawConfigParser()
            # Check if annotation file exists
            path_to_annotation_file = ""
            for file_path in annotation_file_list:
                if im_name in file_path:
                    path_to_annotation_file = file_path
            # If file does not exist --> create an empty one
            if path_to_annotation_file == "":
                print "Warning: No annotations file found for image " + im_name
                print "         Creating empty annotation file."
                path_to_annotation_file = (directory + "\\" + im_name +
                                           "_annotations.ini")
                open(path_to_annotation_file, "a").close()
                config.read(path_to_annotation_file)
            # If file exists --> open
            else:
                config.read(path_to_annotation_file)

            mode = Mode.annotate_original
            operate = True

            while operate:

                n_pos = 0
                n_neg = 0
                for section in config.sections():
                    if "positiveExample" in section:
                        n_pos += 1
                    else:
                        n_neg += 1

                print "Positive annotations in this image: %i" % n_pos
                print "Negative annotations in this image: %i \n" % n_neg

                if mode == Mode.annotate_original:
                    original_with_annotations = draw_annotations(
                            copy.copy(image))
                    cv2.imshow(caption, original_with_annotations)
                elif mode == Mode.annotate_zoomed:
                    zoomed_with_annotations = draw_annotations(
                            copy.copy(zoomed_image))
                    cv2.imshow(caption, zoomed_with_annotations)
                elif mode == Mode.select_area_to_zoom_in:
                    original_with_annotations = draw_annotations(
                            copy.copy(image))
                    original_with_annotations_and_border = draw_border(
                        original_with_annotations, 5, (0, 255, 255))
                    cv2.imshow(caption, original_with_annotations_and_border)

                cv2.setMouseCallback(caption, mouse_handler)

                start_point = None
                end_point = None

                while True:
                    key = cv2.waitKey(10)
                    # Esc key to stop
                    if key == 27:
                        print "Esc pressed, exiting...\n"
                        sys.exit()
                    # Space key to skip to next image
                    elif key == 32:
                        print "Space pressed, moving on to the next image...\n"
                        mode = Mode.continue_to_next_im
                        break
                    elif end_point is not None:
                        break
                    # Tabulator key to change between pos/neg
                    elif key == 9:
                        if annotation_type == AnnotationType.positive_example:
                            annotation_type = AnnotationType.negative_example
                            print "-------------------------------------------"
                            print "You are now annotating negative examples..."
                            print "-------------------------------------------"
                            print ""
                        else:
                            annotation_type = AnnotationType.positive_example
                            print "-------------------------------------------"
                            print "You are now annotating positive examples..."
                            print "-------------------------------------------"
                            print ""
                    # "z" key to zoom in to current mouse coords
                    elif key == 122:
                        if (mode != Mode.select_area_to_zoom_in and
                                mode != Mode.annotate_zoomed):
                            original_with_annotations_and_border = draw_border(
                                copy.copy(original_with_annotations),
                                5, (0, 255, 255))
                            cv2.imshow(caption,
                                       original_with_annotations_and_border)
                            cv2.setMouseCallback(caption, mouse_handler)
                            mode = Mode.select_area_to_zoom_in
                            print "Zooming mode activated\n"
                        else:
                            mode = Mode.annotate_original
                            original_with_annotations = draw_annotations(
                                    copy.copy(image))
                            cv2.imshow(caption, original_with_annotations)
                            cv2.setMouseCallback(caption, mouse_handler)
                            zoomed_image = None
                            print "Zooming mode deactivated\n"

                if (mode == Mode.annotate_original or
                        mode == Mode.annotate_zoomed):

                    if len(start_point) != 2 or len(end_point) != 2:
                        continue

                    ulc = np.min([start_point[0], end_point[0]])
                    lrc = np.max([start_point[0], end_point[0]])
                    ulr = np.min([start_point[1], end_point[1]])
                    lrr = np.max([start_point[1], end_point[1]])

                    if mode == Mode.annotate_zoomed:
                        ulc = ulc / zoom_rate + zoomed_coords[0]
                        lrc = lrc / zoom_rate + zoomed_coords[0]
                        ulr = ulr / zoom_rate + zoomed_coords[1]
                        lrr = lrr / zoom_rate + zoomed_coords[1]

                    # Make sure that actual rectangle was drawn and not
                    # just a dot, line, or NaN
                    width, height = img.size
                    if ulc >= lrc or ulr >= lrr or \
                       ulc > width or ulc < 0 or \
                       lrc > width or lrc < 0 or \
                       ulr > height or ulr < 0 or \
                       lrr > height or lrr < 0 or \
                       np.isnan(ulc) or np.isnan(ulr) or \
                       np.isnan(lrc) or np.isnan(lrr):
                        continue

                    if annotation_type == AnnotationType.positive_example:
                        sec_name = (im_name + "_" + "positiveExample" + "_" +
                                    str(datetime.datetime.now()))
                    else:
                        sec_name = (im_name + "_" + "negativeExample" + "_" +
                                    str(datetime.datetime.now()))

                    if not config.has_section(sec_name):
                        config.add_section(sec_name)

                    config.set(sec_name, "date", time.asctime())
                    config.set(sec_name, "ulc",
                               np.ceil(ulc * decimation_factor))
                    config.set(sec_name, "ulr",
                               np.ceil(ulr * decimation_factor))
                    config.set(sec_name, "lrc",
                               np.ceil(lrc * decimation_factor))
                    config.set(sec_name, "lrr",
                               np.ceil(lrr * decimation_factor))

                if mode == Mode.continue_to_next_im:
                    mode = Mode.annotate_original
                    break

                # Go through every annotation/section in the config/.ini and 
                # delete each that have the clicked coordinate inside of them
                if (mode == Mode.delete_from_original or
                        mode == Mode.delete_from_zoomed):

                    for section in config.sections():

                        ulc = float(config.get(section, "ulc"))
                        ulr = float(config.get(section, "ulr"))
                        lrc = float(config.get(section, "lrc"))
                        lrr = float(config.get(section, "lrr"))

                        if mode == Mode.delete_from_original:
                            x = end_point[0] * decimation_factor
                            y = end_point[1] * decimation_factor
                        else:
                            x = (end_point[0] / zoom_rate +
                                 zoomed_coords[0]) * decimation_factor
                            y = (end_point[1] / zoom_rate +
                                 zoomed_coords[1]) * decimation_factor

                        if not np.isnan(x) and not np.isnan(y):
                            if ulc <= x <= lrc and ulr <= y <= lrr:
                                # print "You clicked inside annotation!"
                                config.remove_section(section)

                    if mode == Mode.delete_from_original:
                        mode = Mode.annotate_original
                    else:
                        mode = Mode.annotate_zoomed

                    continue

                if mode == Mode.select_area_to_zoom_in:
                    zoomed_image = copy.copy(image)

                    ulc = np.min([start_point[0], end_point[0]])
                    lrc = np.max([start_point[0], end_point[0]])
                    ulr = np.min([start_point[1], end_point[1]])
                    lrr = np.max([start_point[1], end_point[1]])

                    # Make sure that actual rectangle was drawn and not
                    # just a dot, line, or NaN
                    width, height = img.size
                    if ulc >= lrc or ulr >= lrr or \
                       ulc > width or ulc < 0 or \
                       lrc > width or lrc < 0 or \
                       ulr > height or ulr < 0 or \
                       lrr > height or lrr < 0 or \
                       np.isnan(ulc) or np.isnan(ulr) or \
                       np.isnan(lrc) or np.isnan(lrr):
                        continue

                    zoomed_image = zoomed_image[ulr:lrr, ulc:lrc]
                    zoomed_image = cv2.resize(
                            zoomed_image,
                            (zoomed_image.shape[1] * zoom_rate,
                             zoomed_image.shape[0] * zoom_rate))
                    zoomed_coords = [ulc, ulr, lrc, lrr]

                    mode = Mode.annotate_zoomed
                    zoomed_image_with_annotations = draw_annotations(
                            copy.copy(zoomed_image))
                    cv2.imshow(caption, zoomed_image_with_annotations)
                    cv2.setMouseCallback(caption, mouse_handler)

                # Write result to file
                with open(path_to_annotation_file, "w") as outfile:
                    config.write(outfile)

    # Finally
    eg.msgbox("All images annotated", title="Annotation completed!")
    cv2.destroyAllWindows()
    sys.exit()
