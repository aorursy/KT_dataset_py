!pip install dicom
import dicom

import os

import pandas as pd



data_dir = "../input/sample_images/"
patients = os.listdir(data_dir)
patients
labels_df = pd.read_csv("../input/stage1_labels.csv", index_col=0)
labels_df