# @Experto

import numpy as np
import sys
import pandas as pd
from skimage.io import imread, imsave
from skimage.color import gray2rgb
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from functools import reduce

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
def rle_encode(im):
    '''
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = im.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def rle_decode(rle_mask):
    '''
    rle_mask: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = rle_mask.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(101*101, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(101,101)
def merge_dataframes(dfs, merge_keys):
    dfs_merged = reduce(lambda left,right: pd.merge(left, right, on=merge_keys), dfs)
    return dfs_merged
df1 = pd.read_csv('../input/submission/crf_c_1.csv').rename(columns={'rle_mask': 'rle_mask'})
df2 = pd.read_csv('../input/submission/crf_c.csv').rename(columns={'rle_mask': 'rle_mask1'})
df3 = pd.read_csv('../input/submission/crf_c_2.csv').rename(columns={'rle_mask': 'rle_mask2'})
df4 = pd.read_csv('../input/submission/sub.csv').rename(columns={'rle_mask': 'rle_mask3'})
#df4 = pd.read_csv('sub.csv')

dfs = [df1,df2,df3,df4]
merge_keys=['id']
df = merge_dataframes(dfs, merge_keys=merge_keys)
df.head()
# Esmaeil Zahedi
res = df1.copy()
i = 0

while True:
    if i == 18000:
        break
        
    if (str(df.loc[i,'rle_mask'])!=str(np.nan)) and (str(df.loc[i,'rle_mask1'])!=str(np.nan)) and (str(df.loc[i,'rle_mask2'])!=str(np.nan))and (str(df.loc[i,'rle_mask3'])!=str(np.nan)): 
        decoded_mask1 = rle_decode(df.loc[i,'rle_mask'])
        decoded_mask2 = rle_decode(df.loc[i,'rle_mask1'])
        decoded_mask3 = rle_decode(df.loc[i,'rle_mask2'])
        decoded_mask4 = rle_decode(df.loc[i,'rle_mask3'])
        
        decoded_mask_all1 = decoded_mask1 + decoded_mask2 + decoded_mask3 + decoded_mask4
        
        decoded_mask_all1[decoded_mask_all1<=2] = 0
        decoded_mask_all1[decoded_mask_all1 >2]  = 1

        mask = rle_encode(decoded_mask_all1)
        
        res.loc[i,'rle_mask'] = mask
    else:
        res.loc[i,'rle_mask'] = df1.loc[i,'rle_mask']
        
    i = i + 1

res.to_csv('blend_ez.csv',index=False)

dft = res
i = 0
plt.figure(figsize=(25,12))
plt.subplots_adjust(bottom=0.3, top=0.9, hspace=0.3) 
j = 0
while True:
    if str(dft.loc[i,'rle_mask'])!=str(np.nan):        
        decoded_mask = rle_decode(dft.loc[i,'rle_mask'])
        plt.subplot(1,7,j+1)
        plt.imshow(decoded_mask)
        plt.title(' ID: '+ df.loc[i,'id'])
        j = j + 1
        if j > 6:
            break
    i = i + 1