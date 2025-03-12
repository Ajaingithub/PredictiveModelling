# In this project, we will build a neural network to automatically segment tumor region in brain using MRI (Magnetic Resonance imaging) scans
# The MRI scan is one of the most common image modalities that we encounter in the radiology field.
# Other data modalities include:
#     Computer Tomography (CT),
#     Ultrasound
#     X-Rays.

import tensorflow as tf
import keras
import json
import numpy as np
import pandas as pd
import nibabel as nib # nibabel will let us extract the images and labels from the files in our dataset.
from tensorflow.keras import backend as K 
import os

os.chdir("/diazlab/data3/.abhinav/.learning/PredictiveModelling/Brain_Tumor_MRI_Segmentation/")
import util

# 1.2 MRI Data Processing
# We often encounter MR images in the DICOM format.
# The DICOM format is the output format for most commercial MRI scanners. This type of data can be processed using the pydicom Python library.
# In this project, we will be using the data from the Decathlon 10 Challenge. This data has been mostly pre-processed for the competition participants, however in real practice, MRI data needs to be significantly pre-preprocessed before we can use it to train our models.

## 1.3 Exploring the Dataset
# Our dataset is stored in the [NifTI-1 format](https://nifti.nimh.nih.gov/nifti-1/) and we will be using the [NiBabel library](https://github.com/nipy/nibabel) to interact with the files. 
# Each training sample is composed of two separate files:

# The first file is an image file containing a 4D array of MR image in the shape of (240, 240, 155, 4). 
# -  The first 3 dimensions are the X, Y, and Z values for each point in the 3D volume, which is commonly called a voxel. 
# - The 4th dimension is the values for 4 different sequences
#     - 0: FLAIR: "Fluid Attenuated Inversion Recovery" (FLAIR)
#     - 1: T1w: "T1-weighted"
#     - 2: t1gd: "T1-weighted with gadolinium contrast enhancement" (T1-Gd)
#     - 3: T2w: "T2-weighted"

# The second file in each training example is a label file containing a 3D array with the shape of (240, 240, 155).  
# - The integer values in this array indicate the "label" for each voxel in the corresponding image files:
#     - 0: background
#     - 1: edema
#     - 2: non-enhancing tumor
#     - 3: enhancing tumor

# We have access to a total of 484 training images which we will be splitting into a training (80%) and validation (20%) dataset.
# Let's begin by looking at one single case and visualizing the data!

# set home directory and data directory
HOME_DIR = "/diazlab/data3/.abhinav/.learning/PredictiveModelling/Brain_Tumor_MRI_Segmentation/rawdata/Task01_BrainTumour/imagesTr/"
DATA_DIR = HOME_DIR

def load_case(image_nifty_file, label_nifty_file):
    # load the image and label file, get the image content and return a numpy array for each
    image = np.array(nib.load(image_nifty_file).get_fdata())
    label = np.array(nib.load(label_nifty_file).get_fdata())
    
    return image, label