import os

import numpy as np

import cv2

import pandas as pd

import pydicom



from PIL import Image

from matplotlib import cm

from matplotlib import pyplot as plt

from matplotlib import patches as patches





print(os.listdir("../input"))
sample_path = '../input/sample images'

df = pd.read_csv(os.path.join(sample_path, 'train-rle-sample.csv'), header=None)



imageId = df[0]

encodedPixels = df[1]



df.head(len(df))
fig, ax = plt.subplots(2, 5, figsize=(20,10))



for i in range(len(df)):

    ds = pydicom.read_file(os.path.join(sample_path, imageId[i] + '.dcm'))

    img = ds.pixel_array

    img_mem = Image.fromarray(img)

    

    if i < 5:

        ax[0][i].imshow(img_mem, cmap='bone')

        ax[0][i].set_title('Index: {}'.format(i))

    else:

        ax[1][i-5].imshow(img_mem, cmap='bone')

        ax[1][i-5].set_title('Index: {}'.format(i))

        

plt.show()
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



def rle2mask(rle, width, height):

    mask= np.zeros(width* height)

    array = np.asarray([int(x) for x in rle.split()])

    starts = array[0::2]

    lengths = array[1::2]



    current_position = 0

    for index, start in enumerate(starts):

        current_position += start

        mask[current_position:current_position+lengths[index]] = 255

        current_position += lengths[index]



    return mask.reshape(width, height)



def bounding_box(img):

    x = np.any(img, axis=1)

    y = np.any(img, axis=0)

    xmin, xmax = np.where(x)[0][[0, -1]]

    ymin, ymax = np.where(y)[0][[0, -1]]



    return xmin, xmax, ymin, ymax
start_idx = 5

num_vis = 3

fig, ax = plt.subplots(num_vis, 3, figsize=(20, 20))



for idx in range(num_vis):

    index = idx + start_idx

    print('Sample Image:', imageId[index] + '.dcm')



    ds = pydicom.read_file(os.path.join(sample_path, imageId[index] + '.dcm'))

    img = ds.pixel_array

    img_mem = Image.fromarray(img)



    # Original image

    ax[idx][0].imshow(img_mem, cmap="bone")

    ax[idx][0].set_title('Original')



    # Masking image

    rleToMask = rle2mask(

        rle=encodedPixels[index],

        width=img.shape[0],

        height=img.shape[1]

    ).T

    ax[idx][1].imshow(img_mem, cmap="bone")

    ax[idx][1].imshow(rleToMask, alpha=0.3, cmap="Reds")

    ax[idx][1].set_title('Masking')



    # Bounding box

    xmin, xmax, ymin, ymax = bounding_box(img=rleToMask)

    rect = patches.Rectangle((ymin, xmin), ymax-ymin, xmax-xmin, linewidth=2, edgecolor='y', facecolor='none')

    ax[idx][2].add_patch(rect)

    ax[idx][2].imshow(img_mem, cmap="bone")

    ax[idx][2].set_title('Bounding Box')



plt.show()
convert_path = './convert_dir'

if not os.path.isdir(convert_path):

    os.mkdir(convert_path)

else:

    pass



for f in os.listdir(sample_path):

    if f[-3:] == 'dcm':

        ds = pydicom.read_file(sample_path + '/' + f)

        img = ds.pixel_array

        cv2.imwrite(convert_path + '/' + f.replace('.dcm', '.png'), img)

        

os.listdir(convert_path)
mask_path = convert_path + '/mask_dir'

if not os.path.isdir(mask_path):

    os.mkdir(mask_path)

else:

    pass



for index in range(len(df)):

    if encodedPixels[index] != '-1':

#         img = cv2.imread(os.path.join(convert_path, imageId[index] + '.png'))

        ds = pydicom.read_file(os.path.join(sample_path, imageId[index] + '.dcm'))

        img = ds.pixel_array

        img_mem = Image.fromarray(img)

        img_size = img.shape

        

        rleToMask = rle2mask(

            rle=encodedPixels[index],

            width=img_size[0],

            height=img_size[1]

        ).T

        rleToMask = rleToMask.astype('int32')   

        

        cv2.imwrite(mask_path + '/{}_mask.png'.format(imageId[index]), rleToMask)

        

    elif encodedPixels[index] == '-1':

#         img = cv2.imread(os.path.join(convert_path, imageId[index] + '.png'))

        ds = pydicom.read_file(os.path.join(sample_path, imageId[index] + '.dcm'))

        img = ds.pixel_array

        img_mem = Image.fromarray(img)

        

        mask_0 = np.zeros((img.shape[:2]))

        mask_0 = mask_0.astype('int32')

        

        cv2.imwrite(mask_path + '/{}_mask.png'.format(imageId[index]), mask_0)

        

os.listdir(mask_path)
mask_list = os.listdir(mask_path)

fig, ax = plt.subplots(2, 5, figsize=(20, 10))



for i in range(len(mask_list)):

    img = cv2.imread(os.path.join(mask_path, imageId[i] + '_mask.png'))

#     img = img[:,:,0]

    if i < 5:

        ax[0][i].imshow(img)

        ax[0][i].set_title('Index: {}'.format(i))

    else:

        ax[1][i-5].imshow(img)

        ax[1][i-5].set_title('Index: {}'.format(i))

        

plt.show()
! rm -rf convert_dir