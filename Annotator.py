# -*- coding: utf-8 -*-

"""

Annotation software.

Left mouse button click + drag: Create new annotation.
Right mouse button click: Delete annotation.

Tab key: Change between annotating positive and negative examples.

z key: Activate/deactivate zooming mode.
       Left mouse button click + drag to select area to be zoomed in.

Space: Move to the next image.
Esc: Exit the program.

Requires the images to be stored in a folder with subfolders (e.g., one per day).

"""

import glob
import easygui as eg
import os
import ConfigParser
import numpy as np
import copy
import time
import datetime
import sys
import cv2
import Tkinter, tkFileDialog
from PIL import Image

# Show smaller versions of images according to this factor
decimationFactor = 1.2
dragging = False

# -----------------------------------------------------------------------------

def drawBorder(img, width, color):
    
    # Left border
    img[0:img.shape[0], 0:width, 0] = color[0]
    img[0:img.shape[0], 0:width, 1] = color[1]
    img[0:img.shape[0], 0:width, 2] = color[2]
    
    # Right border
    img[0:img.shape[0], img.shape[1]-width:img.shape[1], 0] = color[0]
    img[0:img.shape[0], img.shape[1]-width:img.shape[1], 1] = color[1]
    img[0:img.shape[0], img.shape[1]-width:img.shape[1], 2] = color[2]
    
    # Top border
    img[0:width, 0:img.shape[1], 0] = color[0]
    img[0:width, 0:img.shape[1], 1] = color[1]
    img[0:width, 0:img.shape[1], 2] = color[2]
    
    # Bottom border
    img[img.shape[0]-width:img.shape[0], 0:img.shape[1], 0] = color[0]
    img[img.shape[0]-width:img.shape[0], 0:img.shape[1], 1] = color[1]
    img[img.shape[0]-width:img.shape[0], 0:img.shape[1], 2] = color[2]
    
    return img

# -----------------------------------------------------------------------------

def mouseHandler(event, x, y, flags, param):
    
    global dragging
    global actimouseHandleron
    global startPoint    
    global endPoint
    global mode
    global originalWithAnnotations
    global zoomedImage
    global caption
    global currentPos
    
    # Create new annotation by clicking left mouse button and dragging
    if event == cv2.cv.CV_EVENT_LBUTTONDOWN and not dragging:
        startPoint = (x, y)
        dragging = True
    
    if event == cv2.cv.CV_EVENT_LBUTTONUP and dragging:
        endPoint = (x, y)
        dragging = False
        
    if event == cv2.cv.CV_EVENT_MOUSEMOVE and dragging:
        
        if mode == "Annotate original":
            imgCopy = copy.copy(originalWithAnnotations)  
            
            if annotationType == "positiveExample":
                cv2.rectangle(imgCopy, startPoint, (x,y), (0,255,0))
            else:
                cv2.rectangle(imgCopy, startPoint, (x,y), (0,0,255))
                
            cv2.imshow(caption, imgCopy)
        
        if mode == "Select area to be zoomed in":
            imgCopy = copy.copy(originalWithAnnotations)
            imgCopy = drawBorder(imgCopy, 5, (0,255,255))
            cv2.rectangle(imgCopy, startPoint, (x,y), (0,255,255))
            cv2.imshow(caption, imgCopy)
            
        if mode == "Annotate zoomed":
            zoomedImgCopy = copy.copy(zoomedImage)
            zoomedImgCopy = drawAnnotations(zoomedImgCopy)
            
            if annotationType == "positiveExample":
                cv2.rectangle(zoomedImgCopy, startPoint, (x,y), (0,255,0))
            else:
                cv2.rectangle(zoomedImgCopy, startPoint, (x,y), (0,0,255))
            
            cv2.imshow(caption, zoomedImgCopy)
    
    # If right mouse button is clicked, delete annotations in that coordinate
    if event == cv2.cv.CV_EVENT_RBUTTONUP:
        endPoint = (x,y)
        dragging = False
        
        if zoomedImage == None:
            print "Delete from original"
            mode = "Delete from original"
        else:
            print "Delete from zoomed"
            mode = "Delete from zoomed"
    
# -----------------------------------------------------------------------------
    
def drawAnnotations(image):   
    
    global zoomedCoords
    global zoomRate   
    
    # Go through every section and draw it
    for section in config.sections():       
        
        ulc = int(float(config.get(section, "ulc")) / decimationFactor)
        ulr = int(float(config.get(section, "ulr")) / decimationFactor)
        lrc = int(float(config.get(section, "lrc")) / decimationFactor)
        lrr = int(float(config.get(section, "lrr")) / decimationFactor)
                
        # Make sure the section options is not not NaN or INF
        if np.isnan(ulc) or np.isnan(ulr) or \
           np.isnan(lrc) or np.isnan(lrr):
               
            break
        
        # Draw rectangle to full-sized image (use coordinates as they come)
        if mode == "Annotate original" or mode == "Select area to be zoomed in":
            
            if "positiveExample" in section:
                cv2.rectangle(image, (ulc, ulr), (lrc, lrr), (0,255,0))
            else:
                cv2.rectangle(image, (ulc, ulr), (lrc, lrr), (0,0,255))
        
        # Draw rectangle to zoomed-in image (use modified coordinates)
        elif mode == "Annotate zoomed":
            
            # Check that section is inside zoomed area
            if ulc <= zoomedCoords[2] and ulr <= zoomedCoords[3] and \
            lrc >= zoomedCoords[0] and lrr >= zoomedCoords[1]:

                # Correct coordinates relative to zoomed area
                lrc = (lrc-ulc) * zoomRate + (ulc-zoomedCoords[0]) * zoomRate
                lrr = (lrr-ulr) * zoomRate + (ulr-zoomedCoords[1]) * zoomRate
                ulc = (ulc-zoomedCoords[0]) * zoomRate
                ulr = (ulr-zoomedCoords[1]) * zoomRate
                
                if "positiveExample" in section:
                    cv2.rectangle(image, (ulc, ulr), (lrc, lrr), (0,255,0))
                else:
                    cv2.rectangle(image, (ulc, ulr), (lrc, lrr), (0,0,255))
    
    return image
    
# ----------------------------------------------------------------
 
# Returns a list of paths to subfolders
def listdirs(folder):
    return [
        d for d in (os.path.join(folder, d1) for d1 in os.listdir(folder))
        if os.path.isdir(d)
    ] 
 
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    
    # Exchanges information between mouse event handler and main function
    global annotationType
    annotationType = "positiveExample"     
    caption = ""
    
    zoomedCoords = None
    zoomedImage = None
    zoomRate = 4 # How much do you want to zoom?
    
    # Select input folder dialog
    root = Tkinter.Tk()
    inputDir = tkFileDialog.askdirectory(parent=root, \
    initialdir = os.path.dirname(os.path.abspath(__file__)), \
    title = "Select directory containing images and their annotations")
    if inputDir != "":
        listOfDayDirs = listdirs(inputDir)
    else:
        eg.msgbox("Invalid folder! Press OK to exit.", title="Invalid folder")
        sys.exit()
    
    # Selected folder should have subfolders holding microscope images
    # and one annotation file for each image
    imageFileTypes = ["bmp", "jpg", "png"]
    annotationFileType = "ini"
    
    # Loop through folders
    for index,directory in enumerate(listOfDayDirs):
        
        # Get list of images in the folder
        imageFileList = []
        for fileType in imageFileTypes:
            imageFileList = imageFileList + \
            glob.glob(directory + "\*." + fileType)
            
        # Get list of annotation files in the folder
        annotationFileList = glob.glob(directory + "\*." + annotationFileType)
        
        # Loop through images
        for i,pathToImage in enumerate(imageFileList):
            
            # Image name without file type extension
            imName = pathToImage[pathToImage.rfind("\\")+1 : \
            pathToImage.rfind(".")]
            
            # If new image, open new window
            if caption != pathToImage:
                cv2.destroyAllWindows()
            caption = pathToImage
            
            img = Image.open(pathToImage)
            image = np.array(img.convert('RGB'))
            image = cv2.resize(image, (int(image.shape[1]/decimationFactor), \
            int(image.shape[0]/decimationFactor)))
            
            # New config parser
            config = ConfigParser.RawConfigParser() 
            # Check if annotation file exists
            pathToAnnotationFile = ""
            for file in annotationFileList:
                if imName in file:
                    pathToAnnotationFile = file
            # If file does not exist --> create an empty one
            if pathToAnnotationFile == "":
                print "Warning: No annotations file found for image " + imName
                print "         Creating empty annotation file."
                pathToAnnotationFile = directory + "\\" + imName + "_" \
                "annotations.ini"
                open(pathToAnnotationFile, "a").close()
                config.read(pathToAnnotationFile)
            # If file exists --> open
            else:
                config.read(pathToAnnotationFile)   
        
            mode = "Annotate original"
            operate = True
            
            while operate == True:
                
                nrOfPos = 0
                nrOfNeg = 0
                for section in config.sections():
                    if "positiveExample" in section:
                        nrOfPos += 1
                    else:
                        nrOfNeg +=1
                
                print "Positive annotations in this image: %i" % nrOfPos
                print "Negative annotations in this image: %i \n" % nrOfNeg
                
                if mode == "Annotate original":
                    originalWithAnnotations = drawAnnotations(copy.copy(image))
                    cv2.imshow(caption, originalWithAnnotations)
                elif mode == "Annotate zoomed":
                    zoomedWithAnnotations = drawAnnotations(copy.copy(zoomedImage))
                    cv2.imshow(caption, zoomedWithAnnotations)
                elif mode == "Select area to be zoomed in":
                    originalWithAnnotations = drawAnnotations(copy.copy(image))
                    imgCopy = drawBorder(originalWithAnnotations, 5, (0,255,255))
                    cv2.imshow(caption, imgCopy)
                    
                cv2.setMouseCallback(caption, mouseHandler)
            
                startPoint = None
                endPoint = None
    
                while True:
                    key = cv2.waitKey(10)
                    if key == 27: # Esc key to stop
                        print "Esc pressed, exiting...\n"
                        sys.exit()
                    elif key == 32: # Space key to skip to next image
                        print "Space pressed, moving on to the next image...\n"
                        mode = "Continue"
                        break
                    elif endPoint != None:
                        break
                    elif key == 9: # Tabulator key to change between pos/neg
                        if annotationType == "positiveExample":
                            annotationType = "negativeExample"
                            print "-------------------------------------------"
                            print "You are now annotating negative examples..."
                            print "-------------------------------------------\n"
                        else:
                            annotationType = "positiveExample"
                            print "-------------------------------------------"
                            print "You are now annotating positive examples..."
                            print "-------------------------------------------\n"
                    elif key == 122: # "z" key to zoom in to current mouse coords
                        if mode != "Select area to be zoomed in" and \
                           mode != "Annotate zoomed":
                            imgCopy = copy.copy(originalWithAnnotations)
                            imgCopy = drawBorder(imgCopy, 5, (0,255,255))
                            cv2.imshow(caption, imgCopy)
                            cv2.setMouseCallback(caption, mouseHandler)
                            mode = "Select area to be zoomed in"
                            print "Zooming mode activated\n"
                        else:
                            mode = "Annotate original"
                            originalWithAnnotations = drawAnnotations(copy.copy(image))
                            cv2.imshow(caption, originalWithAnnotations)
                            cv2.setMouseCallback(caption, mouseHandler)
                            zoomedImage = None
                            print "Zooming mode deactivated\n"
            
                if "Annotate" in mode:
                    
                    if len(startPoint) !=2 or len(endPoint) != 2:
                        continue
                    
                    ulc = np.min([startPoint[0], endPoint[0]]) 
                    lrc = np.max([startPoint[0], endPoint[0]])
                    ulr = np.min([startPoint[1], endPoint[1]])
                    lrr = np.max([startPoint[1], endPoint[1]])
                    
                    if mode == "Annotate zoomed":
                        ulc = ulc / zoomRate + zoomedCoords[0]
                        lrc = lrc / zoomRate + zoomedCoords[0]
                        ulr = ulr / zoomRate + zoomedCoords[1]
                        lrr = lrr / zoomRate + zoomedCoords[1]
                    
                    # Make sure that actual rectangle was drawn and not
                    # just a dot, line, or NaN
                    width, height = img.size
                    if ulc>=lrc or ulr>=lrr or \
                       ulc > width or ulc < 0 or \
                       lrc > width or lrc < 0 or \
                       ulr > height or ulr < 0 or \
                       lrr > height or lrr < 0 or \
                       np.isnan(ulc) or np.isnan(ulr) or \
                       np.isnan(lrc) or np.isnan(lrr):
                        continue
                    
                    if annotationType == "positiveExample":
                        secName = imName + "_" + "positiveExample" + "_" + \
                        str(datetime.datetime.now())
                    else:
                        secName = imName + "_" + "negativeExample" + "_" + \
                        str(datetime.datetime.now())
                    
                    if not config.has_section(secName):
                        config.add_section(secName)
                    
                    config.set(secName, "date", time.asctime())                
                    config.set(secName, "ulc", np.ceil(ulc * decimationFactor))
                    config.set(secName, "ulr", np.ceil(ulr * decimationFactor))
                    config.set(secName, "lrc", np.ceil(lrc * decimationFactor))
                    config.set(secName, "lrr", np.ceil(lrr * decimationFactor))
                    
    
                if mode == "Continue":
                    mode = "Annotate original"
                    break
               
                # Go through every annotation/section in the config/.ini and 
                # delete each that have clicked coordinate inside of them
                if "Delete" in mode:
                    
                    for section in config.sections():
                        
                        ulc = float(config.get(section, "ulc"))
                        ulr = float(config.get(section, "ulr"))
                        lrc = float(config.get(section, "lrc"))
                        lrr = float(config.get(section, "lrr"))
                        
                        if mode == "Delete from original":
                            x = endPoint[0] * decimationFactor
                            y = endPoint[1] * decimationFactor
                        else:
                            x = (endPoint[0] / zoomRate + zoomedCoords[0]) * decimationFactor
                            y = (endPoint[1] / zoomRate + zoomedCoords[1]) * decimationFactor
                        
                        if not np.isnan(x) and not np.isnan(y):
                            if (x >= ulc) and (x <= lrc) and \
                               (y >= ulr) and (y <= lrr):
                                #print "You clicked inside annotation!"
                                config.remove_section(section)

                    if mode == "Delete from original":
                        mode = "Annotate original"
                    else:
                        mode = "Annotate zoomed"
                    
                    continue
                    
                
                if mode == "Select area to be zoomed in":
                    zoomedImage = copy.copy(image)
                    
                    ulc = np.min([startPoint[0], endPoint[0]]) 
                    lrc = np.max([startPoint[0], endPoint[0]])
                    ulr = np.min([startPoint[1], endPoint[1]])
                    lrr = np.max([startPoint[1], endPoint[1]])
                    
                    # Make sure that actual rectangle was drawn and not
                    # just a dot, line, or NaN
                    width, height = img.size
                    if ulc>=lrc or ulr>=lrr or \
                       ulc > width or ulc < 0 or \
                       lrc > width or lrc < 0 or \
                       ulr > height or ulr < 0 or \
                       lrr > height or lrr < 0 or \
                       np.isnan(ulc) or np.isnan(ulr) or \
                       np.isnan(lrc) or np.isnan(lrr):
                        continue

                    zoomedImage = zoomedImage[ulr:lrr, ulc:lrc]
                    zoomedImage = cv2.resize(zoomedImage, (zoomedImage.shape[1]*zoomRate, zoomedImage.shape[0]*zoomRate))
                    zoomedCoords = [ulc,ulr,lrc,lrr]
                    
                    mode = "Annotate zoomed"                     
                    zoomedImageWithAnnotations = drawAnnotations(copy.copy(zoomedImage))
                    cv2.imshow(caption, zoomedImageWithAnnotations)
                    cv2.setMouseCallback(caption, mouseHandler)
                    
                # Write result to file
                with open(pathToAnnotationFile, "w") as outfile:
                    config.write(outfile)
    
    # Finally
    eg.msgbox("All images annotated", title="Annotation completed!")
    cv2.destroyAllWindows()
    sys.exit()

    
