# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image

"""

This function calculates Bivariate Similarity Index.

Inputs: 
pathToTruthIm = string, path to binary ground truth image
pathToEstimIm = string, path to binary estimated image

Outputs:
TET  = value between [0,1]
TEE  = value between [0,1]
dSeq = Segmentation distance

"""

def BSI(pathToTruthIm, pathToEstimIm):
    
    # T = Truth
    # E = Estimate
    
    # a = The number of pixels where the value of T and E is (1,1), e.g., TP
    # b = The number of pixels where the value of T and E is (0,1), e.g., FP
    # c = The number of pixels where the value of T and E is (1,0), e.g., FN
    # d = The number of pixels where the value of T and E is (0,0), e.g., TN        
    
    # Open and identify ground truth & estimated image file
    imTruth = Image.open(pathToTruthIm)
    imEstim = Image.open(pathToEstimIm)
    
    # Save dimensions
    width, height = imTruth.size
    
    # Read from the file 
    pixTruth = imTruth.load()
    pixEstim = imEstim.load()
    
    # Initialize all to zero
    a, b, c, d, T, E = [0.0] * 6
    
    # Loop through every pixel
    for y in range(height):
        for x in range(width):
            
            # Calculate a,b,c,d.
            # For similarity and difference metrics, see:
            # "A Survey of Binary Similarity and Distance Measures"
            # http://www.iiisci.org/journal/CV$/sci/pdfs/GS315JG.pdf
            
            if pixTruth[x,y]==255   and pixEstim[x,y]==255:
                a += 1
            elif pixTruth[x,y]==0   and pixEstim[x,y]==255:
                b += 1
            elif pixTruth[x,y]==255 and pixEstim[x,y]==0:
                c += 1
            elif pixTruth[x,y]==0   and pixEstim[x,y]==0:
                d += 1
            
            # Calculate T and E for Bivariate Similarity Index.
            # For more information, see:
            # "Comparison of segmentation algorithms for fluorescence microscopy images of cells"
            # http://onlinelibrary.wiley.com/doi/10.1002/cyto.a.21079/abstract

            if pixTruth[x,y]==255:
                T += 1
            if pixEstim[x,y]==255:
                E += 1
    
    # Bivariate Similarity Index
    TET = a / T
    TEE = a / E

    # Segmentation distance d_seg
    dSeg = np.sqrt( (1-TET)**2 + (1-TEE)**2 )
    
    return (TET, TEE, dSeg)
