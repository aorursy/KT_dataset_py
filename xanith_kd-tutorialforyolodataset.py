# Section 1 Preprocessing and system setup
#1. Copy from input directory to working directory for file access.  
import os, sys, stat
from distutils.dir_util import remove_tree
from distutils.dir_util import copy_tree

#remove_tree('/kaggle/working/')
os.chdir('/kaggle/input/tutorialforyolodatasetforkshitiz/tutorialforyolodatasetForKshitiz')
fromdirectory = './'
todirectory = '/kaggle/working/'
copy_tree(fromdirectory, todirectory)

os.chdir("/kaggle/working/keras-yolo3-master")
for dirname, _, filenames in os.walk('.'):
     for filename in filenames:
             print(os.path.join(dirname, filename))
 
# 2. Prepare yolo format file
!python xml_to_yolo_for_train.py
!python xml_to_yolo_for_test.py 

# 3. Get anchor information
!python kmeans.py
#  Update anchors in: model_data/yolo_anchors.txt

# 4. Train the model
# Make sure version of tensorflow and keras are same version as huckleberry1
#!pip uninstall tensorflow --yes
#!pip uninstall keras --yes
!pip install tensorflow==1.14.0
!pip install keras==2.3.1
!python -c 'import tensorflow as tf; print(tf.__version__)'
!python -c 'import keras; print(keras.__version__)'

# Section 2 train, evaluation, and prediction
# Train
#!python train.py


# Evaluation
#!python yolo_evaluation.py 


# mAP calculation    
#!python yolo_mAP_Calculation.py 



# Make prediction
#!python yolo_video.py --input ../dataset/images/test_for_detection/IMG_0107.png