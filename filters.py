# -*- coding: utf-8 -*-


import sys

import numpy as np


"""

Detections inside other detections can be filtered out with this function.
The function returns True if rectangle r is inside rectangle q.

"""


def is_inside(r, q, hog):
    
    if len(r) == 4:
        rx, ry, rw, rh = r
        qx, qy, qw, qh = q
        
    elif len(r) == 2:
        rx, ry = np.double(r)
        rw, rh = hog.winSize
        qx, qy = np.double(q)
        qw, qh = hog.winSize
        
    else:
        print "!!!"
        print "!!! Error: rectangle coordinates have weird size :("
        print "!!!"
        sys.exit(2)
        
    return (rx > qx and
            ry > qy and
            rx + rw < qx + qw and
            ry + rh < qy + qh)


"""

Detections overlapping too much with each other can be filtered out with
this function. The function returns overlap between rectangle r and rectangle q
according to the PAS metric. PAS metric is adopted from PASCAL VOC challenge.

"""


def get_overlap(r, q, hog=None, structure_of_data=None):

    # In case of four coordinates, hog.detectMultiScale function was used, 
    # which returns rectangle start points as well as widths & heights.
    if len(r) == 4:    
        rx, ry, rw, rh = np.double(r)
        qx, qy, qw, qh = np.double(q)
        
    # In case of two coordinates, hog.detect function was used, 
    # which returns only rectangle start points.
    # Insert rectangle widths & heights from hog.winSize.
    elif len(r) == 2:
        rx, ry = np.double(r)
        rw, rh = hog.winSize
        qx, qy = np.double(q)
        qw, qh = hog.winSize
        
    else:
        print "!!!"
        print "!!! Error: rectangle coordinates have weird size :("
        print "!!!"
        sys.exit(2)

    # This overlap function is called in two cases: when filtering detections
    # after sliding window procedure and when building confusion matrix.
    # detectMultiScale returns upper left corner coordinates & widths & heights
    # but annotation software (annotator.py) saved upper left corner
    # coordinates and lower right corner coordinates to INI files. 
    # In future work, annotation software should be changed so that it saves
    # also widths & heights instead of lower right coordinates so that we 
    # could use one straightforward overlap function 
    # instead of the following IF statement:
    if structure_of_data == "corners":
        # Subtract end coordinates from starting coordinates to
        # acquire width & height for both rectangles r & q
        rw = rw-rx
        rh = rh-ry
        qw = qw-qx
        qh = qh-qy
    
    intersection_area = (
        np.max([0, np.min([rx+rw, qx+qw]) - np.max([rx, qx])]) *
        np.max([0, np.min([ry+rh, qy+qh]) - np.max([ry, qy])]))

    r_area = rw*rh
    q_area = qw*qh
    union = r_area + q_area - intersection_area
    overlap = intersection_area / union
    
    return overlap
