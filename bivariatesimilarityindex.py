# -*- coding: utf-8 -*-


import numpy as np
from PIL import Image as PILImage


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


def bsi(path_to_truth_im, path_to_estim_im):
    
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
