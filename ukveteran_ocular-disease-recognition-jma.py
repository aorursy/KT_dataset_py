import os
print(os.listdir("../input"))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import PIL
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
image="../input/ocular-disease-recognition-odir5k/ODIR-5K/Training Images/0_right.jpg"
PIL.Image.open(image)
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
data_df = pd.read_excel(open("/kaggle/input/ocular-disease-recognition-odir5k/ODIR-5K/data.xlsx", 'rb'), sheet_name='Sheet1') 
data_df.columns = ["id", 'age', "sex", "left_fundus", "right_fundus", "left_diagnosys", "right_diagnosys", "normal",
                  "diabetes", "glaucoma", "cataract", "amd", "hypertension", "myopia", "other"]
print(data_df.loc[(data_df.cataract==1)].shape)
print(data_df.loc[data_df.cataract==0].shape)
data_df.loc[(data_df.cataract==1)]['left_diagnosys'].value_counts()
data_df.loc[(data_df.cataract==1)]['right_diagnosys'].value_counts()
def has_cataract_mentioned(text):
    if 'cataract' in text:
        return 1
    else:
        return 0
data_df['le_cataract'] = data_df['left_diagnosys'].apply(lambda x: has_cataract_mentioned(x))
data_df['re_cataract'] = data_df['right_diagnosys'].apply(lambda x: has_cataract_mentioned(x))
cataract_le_list = data_df.loc[(data_df.cataract==1) & (data_df.le_cataract==1)]['left_fundus'].values
cataract_re_list = data_df.loc[(data_df.cataract==1) & (data_df.re_cataract==1)]['right_fundus'].values
print(len(cataract_le_list), len(cataract_re_list))
non_cataract_le_list = data_df.loc[(data_df.cataract==0) & (data_df.left_diagnosys=="normal fundus")]['left_fundus'].sample(150).values
non_cataract_re_list = data_df.loc[(data_df.cataract==0) & (data_df.right_diagnosys=="normal fundus")]['right_fundus'].sample(150).values
print(len(non_cataract_le_list), len(non_cataract_re_list))
cataract_list = np.concatenate((cataract_le_list, cataract_re_list), axis = 0)
non_cataract_list = np.concatenate((non_cataract_le_list, non_cataract_re_list), axis = 0)
print(len(non_cataract_list), len(cataract_list))

