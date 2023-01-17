import numpy as np 

import pandas as pd

import os 

import matplotlib.pyplot as plt 

from glob import glob 

import pydicom

from tqdm import * 
def rle2mask(rle, width, height):

    mask= np.zeros(width* height)

    if rle == ' -1': 

        return mask.reshape(width, height)

    array = np.asarray([int(x) for x in rle.split()])

    starts = array[0::2]

    lengths = array[1::2]



    current_position = 0

    for index, start in enumerate(starts):

        current_position += start

        mask[current_position:current_position+lengths[index]] = 255

        current_position += lengths[index]



    return mask.reshape(width, height)



def mask2rle(img, width, height):

    rle = []

    lastColor = 0;

    currentPixel = 0;

    runStart = -1;

    runLength = 0;



    for x in range(width):

        for y in range(height):

            currentColor = img[x][y]

            if currentColor != lastColor:

                if currentColor == 255:

                    runStart = currentPixel;

                    runLength = 1;

                else:

                    rle.append(str(runStart));

                    rle.append(str(runLength));

                    runStart = -1;

                    runLength = 0;

                    currentPixel = 0;

            elif runStart > -1:

                runLength += 1

            lastColor = currentColor;

            currentPixel+=1;



    return " ".join(rle)
os.listdir('../input/pneumothorax/')

train_path = '../input/pneumothorax/dicom-images-train/'

test_path = '../input/pneumothorax/dicom-images-test/'

rle_pd = pd.read_csv('../input/pneumothorax/train-rle.csv')

len(os.listdir(train_path)), len(os.listdir(test_path)), rle_pd.shape  # 10712 train, 1377 test, 
def combine_rle(rle_list): 

    mask = np.zeros((1024,1024)).astype(int)

    for i in rle_list: 

        i_mask = rle2mask(i,1024,1024).astype(int)

        mask = (mask|i_mask).astype(int)

    return  mask2rle(mask, 1024, 1024)



rle_count_pd = rle_pd.ImageId.value_counts()[rle_pd.ImageId.value_counts()>1].index



for i in tqdm_notebook( rle_count_pd ) : 

    combined_mask_rle = combine_rle(  rle_pd[rle_pd.ImageId == i ][' EncodedPixels'].values  )

    rle_pd.loc[ rle_pd[rle_pd.ImageId == i ].index, ' EncodedPixels' ] = combined_mask_rle

rle_pd = rle_pd.drop_duplicates().reset_index()
def list_all_files(rootdir):

    import os

    _files = []

    list = os.listdir(rootdir) 

    for i in range(0,len(list)):

           path = os.path.join(rootdir,list[i])

           if os.path.isdir(path):

              _files.extend(list_all_files(path))

           if os.path.isfile(path):

              _files.append(path)

    return _files



def get_list( path ): 

    _list = []

    for i in tqdm(os.listdir(path)): 

        _list.append(list_all_files( path + i)[0] )

    return _list



def get_train_pd():

    train_list = get_list(train_path)

    rle_pd['path'] = np.zeros_like(rle_pd.ImageId)

    path_list = np.zeros_like(rle_pd.ImageId)

    for i in tqdm_notebook(train_list): 

        SIUID = pydicom.dcmread( i ).get('SOPInstanceUID')

        if SIUID in rle_pd.ImageId.values: 

            path_list[rle_pd[rle_pd.ImageId == SIUID]['path'].index] = i 

    rle_pd['path'] = path_list

    return rle_pd



def get_test_pd():

    test_list = get_list(test_path)

    test_image_id = []

    test_pixel = []

    for i in test_list: 

        test_image_id.append( i[-58:-4] ) 

        test_pixel.append(' -1')    

    test_pd = pd.DataFrame(np.array([test_image_id, test_pixel, test_list]).T, columns = ['ImageId', ' EncodedPixels', 'path']  ) 

    return test_pd



# extracting file name list ... takes about 2 mins...

train_pd = get_train_pd()

test_pd = get_test_pd()

train_pd.shape, test_pd.shape
train_pd[train_pd.columns[1:]]
train_pd[train_pd.columns[1:]].to_csv('train_pd.csv', index = False )

test_pd.to_csv('test_pd.csv', index = False )