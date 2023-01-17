# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
datafolder = '/kaggle/input/bengaliai-cv19/'
# imports

import gc

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.models import load_model
datafolder = '/kaggle/input/bengaliai-cv19/'

# initialize submission dataframe

row_ids = []

pred = []



# initialize variables

roots = ['vowel_diacritic','grapheme_root','consonant_diacritic']



#model = load_model('/kaggle/input/bengaliai-trained-models-mjy-v3/mjy_vowel_diacritic_v4.h5')



# start loop to score test data (we have 4 test datasets [0,3])

start = 0

for z in range(4):

    # load data to dataframe

    img_df = pd.read_parquet(datafolder + 'test_image_data_'+str(z)+'.parquet')

    # get the number of training images from the target\id dataset

    N = img_df.shape[0]

    # drop ids

    #img_df = img_df.drop('image_id', axis = 1)

    # convert to numpy, reshape and normalize

    #x_test = img_df.values.reshape((N,137,236,1)) / 255



    # loop through models, score test data, and update submission dataframe

    for root in roots:

        # get predictions

        pred.extend([1 for k in range(N)])

        # create row id

        row_ids.extend(['Test_'+str(j)+'_'+root for j in range(start,start+N)])

    start += N



    # clean up

    del img_df

    #del x_test

    gc.collect()
submit_df = pd.DataFrame({'row_id':row_ids,'target':pred},

                         columns = ['row_id','target'])

submit_df.head(100)
# clean up

del pred, row_ids

gc.collect()
submit_df.to_csv('submission.csv',index=False)