import sys

import pandas as pd

import time

import matplotlib.pyplot as plt



sys.path.insert(0, '../input/siim-acr-pneumothorax-segmentation')



from mask_functions import *
df = pd.read_csv('../input/siim-acr-pneumothorax-segmentation-data/pneumothorax/train-rle.csv', index_col=0)

df.columns = ['map']



def get_totals():

    totals = np.zeros((1024,1024))

    for index,row in df.iterrows():

        if row['map'] != ' -1':

            mask = rle2mask(row['map'], 1024, 1024)

            totals += mask

    totals /= 255

    return totals



%time totals = get_totals()
fig, axes = plt.subplots(2, 2, figsize=(20, 20))



threshold = 20



axes[0][0].imshow(totals.T, cmap=plt.cm.bone)

axes[0][0].set_title('totals')

axes[0][1].imshow(np.clip(totals,0,200).T, cmap=plt.cm.bone)

axes[0][1].set_title('np.clip(totals,0,200)')

axes[1][0].imshow(np.clip(totals,0,100).T, cmap=plt.cm.bone)

axes[1][0].set_title('np.clip(totals,0,100)')

axes[1][1].imshow(totals.T > threshold, cmap=plt.cm.bone)

a = axes[1][1].set_title('totals > %d' % threshold)
super_mask_right = mask2rle(255*np.logical_and(totals > threshold, np.tile(range(1024),1024).reshape((1024,1024)).T >= 512),1024,1024)

super_mask_left = mask2rle(255*np.logical_and(totals > threshold, np.tile(range(1024),1024).reshape((1024,1024)).T < 512),1024,1024)



sub = pd.read_csv('../input/siim-acr-pneumothorax-segmentation/sample_submission.csv')



tmp = sub.groupby('ImageId')['ImageId'].count().reset_index(name='N')

tmp = tmp.loc[tmp.N > 1] #find image id's with more than 1 row -> has pneumothorax mask!

print('number of instances with duplicate ids', len(tmp))



# length 1484 to 1294:

sub = sub.drop(sub.loc[sub.ImageId.isin(tmp.ImageId),:].index)

pd_right = pd.DataFrame({'ImageId': tmp.ImageId, 'EncodedPixels': super_mask_right})

pd_left = pd.DataFrame({'ImageId': tmp.ImageId, 'EncodedPixels': super_mask_left})

# new length is 1450 after adding 2*78:

sub = pd.concat([sub, pd_right, pd_right, pd_left, pd_left], axis=0)



sub.to_csv('sample_submission_on_steroids.csv',index=None)
fig, axes = plt.subplots(1, 2, figsize=(10, 10))

a = axes[0].imshow(rle2mask(super_mask_left,1024,1024).T)

a = axes[1].imshow(rle2mask(super_mask_right,1024,1024).T)