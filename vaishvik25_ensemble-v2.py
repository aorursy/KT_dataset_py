import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))

# load the two reference submissions 

#xf1 = pd.read_csv('../input/hypercolumns-pneumothorax-fastai-0-831-lb/submission.csv')

xf2 = pd.read_csv('../input/lungv1/leaky_unet_submission.csv')

xf1 = pd.read_csv('../input/unet-with-efficientnet-encoder-in-keras/orig_submission.csv')

# align indices

xf1.columns = ['ImageId', 'enc1']

xf2.columns = ['ImageId', 'enc2']



xf3 = pd.merge(left = xf1, right = xf2, on = 'ImageId', how = 'inner')

print(xf1.shape, xf2.shape, xf3.shape)



# identify the positions where xf1 has empty predictions but xf2 does not

xf3[xf3['enc1'] != xf3['enc2']]

id1 = np.where(xf3['enc1'] == '-1')[0]

id2 = np.where(xf3['enc2'] != '-1')[0]

idx = np.intersect1d(id1,id2)



# map non-empty xf2 slots to empty ones in xf1

xf3['EncodedPixels'] = xf3['enc1']

xf3['EncodedPixels'][idx] = xf3['enc2'][idx]
xf3[['ImageId','EncodedPixels']].to_csv('hybrid_1_2.csv', index = False)