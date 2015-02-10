# -*- coding: utf-8 -*-

import os
import glob
import scipy.io
import numpy as np  
import ConfigParser
from PIL import Image, ImageDraw

# ----------------------------------------------------------------
 
# Returns a list of paths to subfolders
def Listdirs(folder):
    return [
        d for d in (os.path.join(folder, d1) for d1 in os.listdir(folder))
        if os.path.isdir(d)
    ] 

# ----------------------------------------------------------------

def Save(pathToFolder):
    
    listOfDayDirs = Listdirs(pathToFolder)
    fileTypes = ["bmp", "jpg", "png"]
    
    # Loop through input folders (days)
    for k,directory in enumerate(listOfDayDirs):

        # Get list of images in the folder
        imageList = []
        for fileType in fileTypes:
            imageList = imageList + glob.glob(directory +  "\*." + fileType)
        
        # Loop through images
        for i,imFileName in enumerate(imageList):
        
            # Open current image
            img = Image.open(imFileName)

            # Path to image's annotation file
            pathToAnnotationFile = os.path.splitext(imFileName)[0] + \
            "_annotations.ini"
            
            # If file does not exist --> skip to next image
            if not os.path.isfile(pathToAnnotationFile):
                continue
            
            # If file does exist --> open it
            else:
                config = ConfigParser.RawConfigParser()
                config.read(pathToAnnotationFile)  
        
            # Create new black image where annotations are marked with white
            markedAnnotations = Image.new(img.mode, img.size, "black")       
            
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
                draw = ImageDraw.Draw(markedAnnotations)
                draw.rectangle([ulc,ulr,lrc,lrr],  fill=255)
    
            # Save the image
            indexOfLastSlash = imFileName.rfind("\\")
            indexOfLastDot = imFileName.rfind(".")
            saveWithThisName = imFileName[indexOfLastSlash+1:indexOfLastDot]

            if not os.path.exists("./groundTruth/" + str(k+1)):
                os.makedirs("./groundTruth/" + str(k+1))
            scipy.misc.imsave("./groundTruth/" + str(k+1) + "/" + \
            saveWithThisName + imFileName[indexOfLastDot:], markedAnnotations)

