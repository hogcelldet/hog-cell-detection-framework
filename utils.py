import os
import glob
import ConfigParser

import pylab as pl
import numpy as np
from scipy import misc
from PIL import Image, ImageDraw


# Returns a list of paths to sub folders
def list_dirs(folder):
    return [d for d in (os.path.join(folder, d1) for d1 in os.listdir(folder))
            if os.path.isdir(d)]


# Clears the screen
def cls():
    os.system(['clear', 'cls'][os.name == 'nt'])


# Create binary ground truth images
def create_ground_truth_ims(path_to_folder):

    folders = list_dirs(path_to_folder)
    file_types = ["bmp", "jpg", "png"]

    # Loop through input folders (days)
    for day, directory in enumerate(folders, 1):

        # Get list of images in the folder
        images = []
        for file_type in file_types:
            images = images + glob.glob(directory + "\*." + file_type)

        # Loop through images
        for i, im_name in enumerate(images):

            # Open current image
            img = Image.open(im_name)

            # Path to image's annotation file
            path_to_annotation_file = (os.path.splitext(im_name)[0] +
                                       "_annotations.ini")

            # If file does not exist --> skip to next image
            if not os.path.isfile(path_to_annotation_file):
                continue

            # If file does exist --> open it
            else:
                config = ConfigParser.RawConfigParser()
                config.read(path_to_annotation_file)

            # Create new black image where annotations are marked with white
            marked_annotations = Image.new(img.mode, img.size, "black")

            # Go through every annotation and extract it from original image
            for section in config.sections():

                ulc = int(float(config.get(section, "ulc")))
                ulr = int(float(config.get(section, "ulr")))
                lrc = int(float(config.get(section, "lrc")))
                lrr = int(float(config.get(section, "lrr")))

                # Make sure the section options is not not NaN or INF
                if np.isnan(ulc) or np.isnan(ulr) or \
                   np.isnan(lrc) or np.isnan(lrr):
                    break

                # Mark handled annotation with white
                draw = ImageDraw.Draw(marked_annotations)
                draw.rectangle([ulc, ulr, lrc, lrr], fill=255)

            # Save the image
            last_slash_ix = im_name.rfind("\\")
            last_dot_ix = im_name.rfind(".")
            output_name = im_name[last_slash_ix+1:last_dot_ix]

            if not os.path.exists("./groundTruth/" + str(day)):
                os.makedirs("./groundTruth/" + str(day))
            misc.imsave("./groundTruth/" + str(day) + "/" + output_name +
                        im_name[last_dot_ix:], marked_annotations)


def add_subplot_axes(ax, rect, axisbg='w'):
    fig = pl.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position = ax.transAxes.transform(rect[0:2])
    trans_figure = fig.transFigure.inverted()
    infig_position = trans_figure.transform(inax_position)
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]
    subax = fig.add_axes([x, y, width, height], axisbg=axisbg)
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2]**0.5
    y_labelsize *= rect[3]**0.5
    subax.xaxis.set_tick_params(labelsize=0)
    subax.yaxis.set_tick_params(labelsize=0)
    return subax


def draw_hog_params_table(hog_params, axis):
    row_labels = ['winSize',
                  'blockSize',
                  'blockStride',
                  'cellSize',
                  'nbins']

    table_vals = [[hog_params["_winSize"]],
                  [hog_params["_blockSize"]],
                  [hog_params["_blockStride"]],
                  [hog_params["_cellSize"]],
                  [hog_params["_nbins"]]]

    the_table = axis.table(
        cellText=table_vals,
        colWidths=[0.1] * 3,
        rowLabels=row_labels,
        loc='lower left')
    the_table.set_fontsize(12)
    the_table.scale(2, 2)

    pl.subplots_adjust(wspace=0.22)
    pl.show()
    pl.draw()


def save_fig(file_name_prefix, hog_params):
    pl.savefig(r".\%s" % file_name_prefix +
               "_wSi" + str(hog_params["_winSize"]) +
               "_bSi" + str(hog_params["_blockSize"]) +
               "_bSt" + str(hog_params["_blockStride"]) +
               "_cS" + str(hog_params["_cellSize"]) +
               "_nbins" + str(hog_params["_nbins"]) + ".pdf")
