# hog-cell-detection-framework

Cell detection framework based on HOG feature descriptors and iterative training. The framework is programmed in Python by using OpenCV's implementation of HOG descriptor and scikit-learn's implementation of SVM.

The framework was produced during a master’s thesis work that was carried out at the Department of Signal Processing, Tampere University of Technology (TUT), during 2014. The master's thesis is available online: http://urn.fi/URN:NBN:fi:tty-201412021562

A research article _"Training based cell detection from bright-field microscope images"_ summing up the essence of the thesis was accepted to the Image and Signal Processing and Analysis (ISPA) 2015 conference: http://dx.doi.org/10.1109/ISPA.2015.7306051.

## Prerequisites

* Python 2.7.12
* OpenCV 2.4.11
* NumPy 1.11.1
* scikit-learn 0.17.1
* EasyGui 0.96
* PIL 1.1.7

## Data

Images and annotations for training and testing:

https://drive.google.com/folderview?id=0BwxRd8rtJl9UT183S20wWG81TmM&usp=sharing

All sharpest images from each day:

https://drive.google.com/folderview?id=0BwxRd8rtJl9UVnJkVExQTko4Mmc&usp=sharing

## Citing

You can freely use and edit the code, but please cite the source.

* BibTeX entry for the master's thesis:
       
        @mastersthesis{tikkanen2014cell,
          title={Cell Detection from Microscope Images Using Histogram of Oriented Gradients},
          author={Tikkanen, Tuomas},
          year={2014},
          school={Tampere University of Technology},
          address={Finland}
        }

* BibTeX entry for the ISPA paper:
    
        @inproceedings{tikkanen2015training,
          title={Training based cell detection from bright-field microscope images},
          author={Tikkanen, Tuomas and Ruusuvuori, Pekka and Latonen, Leena and Huttunen, Heikki},
          booktitle={2015 9th International Symposium on Image and Signal Processing and Analysis (ISPA)},
          pages={160--164},
          year={2015},
          organization={IEEE}
        }
    
## Contact

For any questions related to the work, please contact:

Tuomas Tikkanen

tuomas.tikkanen (at) hotmail.com
