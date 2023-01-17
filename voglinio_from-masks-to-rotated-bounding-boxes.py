import os
import cv2
from tqdm import tqdm
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from skimage.measure import label, regionprops
ship_dir = '../input'
train_image_dir = os.path.join(ship_dir, 'train')
test_image_dir = os.path.join(ship_dir, 'test')

from skimage.morphology import label
def multi_rle_encode(img):
    labels = label(img[:, :, 0])
    return [rle_encode(labels==k) for k in np.unique(labels[labels>0])]

# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction

def masks_as_image(in_mask_list, all_masks=None):
    # Take the individual ship masks and create a single mask array for all ships
    if all_masks is None:
        all_masks = np.zeros((768, 768), dtype = np.int16)
    #if isinstance(in_mask_list, list):
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks += rle_decode(mask)
    return np.expand_dims(all_masks, -1)
masks = pd.read_csv(os.path.join('../input/',
                                 'train_ship_segmentations.csv'))
print(masks.shape[0], 'masks found')
print(masks['ImageId'].value_counts().shape[0])
masks.head()
images_with_ship = masks.ImageId[masks.EncodedPixels.isnull()==False]
images_with_ship = np.unique(images_with_ship.values)
print('There are ' +str(len(images_with_ship)) + ' image files with masks')
plotme = 1

for i in tqdm(range(0, 10)):
    image = images_with_ship[i]

    if plotme == 1:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (15, 5))
    img_0 = cv2.imread(train_image_dir+'/' + image)
    rle_0 = masks.query('ImageId=="'+image+'"')['EncodedPixels']
    mask_0 = masks_as_image(rle_0)
    #
    # 
    lbl_0, lbl_cnt = label(mask_0, return_num=True) 
    #props = regionprops(lbl_0)
    img_1 = img_0.copy()
    #print ('Image', image, lbl_cnt)
    for i in range(1, lbl_cnt+1):
        mask = np.array((lbl_0 == i).astype('uint8')[..., 0])
        mask = cv2.resize(mask, (768, 768))
        _, cnts, hierarchy = cv2.findContours((255*mask).astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        rect = cv2.minAreaRect(cnts[0])
        if rect[1][1]>rect[1][0]:
            angle = 90-rect[2]
        else:
            angle = -rect[2]
        box = cv2.boxPoints(rect)
        #print (box)
        box = np.int0(box)

        if plotme == 1:
            cv2.drawContours(img_1,[box],0,(0,191,255),2)
        x = int(rect[0][0])
        y = int(rect[0][1])
        #print (rect, angle, 360*((props[i-1].orientation + np.pi/2)/(2*np.pi)), x, y)
        if plotme == 1:
            cv2.circle(img_1, ( x, y ), 5, (255, 0, 0), 3)
        print(str(rect[0][0]) + ' ' + str(rect[0][1]) + ' ' + str(rect[1][0]) + ' ' + str(rect[1][1]) + ' ' + '1' + ' ' + str(angle) + '\n' )

    if plotme == 1:
        ax1.imshow(img_0)
        ax1.set_title('Image')
        ax2.set_title('Mask')
        ax3.set_title('Image with derived bounding box')
        ax2.imshow(mask_0[...,0], cmap='gray')
        ax3.imshow(img_1)
        plt.show()
plotme = 0
rot_bboxes_dict = {}

for i in tqdm(range(0, len(images_with_ship))):
    image = images_with_ship[i]

    if plotme == 1:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (15, 5))
    img_0 = cv2.imread(train_image_dir+'/' + image)
    rle_0 = masks.query('ImageId=="'+image+'"')['EncodedPixels']
    mask_0 = masks_as_image(rle_0)
    #
    # 
    lbl_0, lbl_cnt = label(mask_0, return_num=True) 
    #props = regionprops(lbl_0)
    img_1 = img_0.copy()
    bboxes = []
    for i in range(1, lbl_cnt+1):
        mask = np.array((lbl_0 == i).astype('uint8')[..., 0])
        mask = cv2.resize(mask, (768, 768))
        _, cnts, hierarchy = cv2.findContours((255*mask).astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        rect = cv2.minAreaRect(cnts[0])
        if rect[1][1]>rect[1][0]:
            angle = 90-rect[2]
        else:
            angle = -rect[2]
        box = cv2.boxPoints(rect)
        #print (box)
        box = np.int0(box)

        if plotme == 1:
            cv2.drawContours(img_1,[box],0,(0,191,255),2)
        x = int(rect[0][0])
        y = int(rect[0][1])
        if plotme == 1:
            cv2.circle(img_1, ( x, y ), 5, (255, 0, 0), 3)
        bboxes.append([rect[0][0], rect[0][1], rect[1][0], rect[1][1], angle])
        
    rot_bboxes_dict[image] = bboxes.copy()
    if plotme == 1:
        ax1.imshow(img_0)
        ax1.set_title('Image')
        ax2.set_title('Mask')
        ax3.set_title('Image with derived bounding box')
        ax2.imshow(mask_0[...,0], cmap='gray')
        ax3.imshow(img_1)
        plt.show()
bboxes_df = pd.DataFrame([rot_bboxes_dict])
bboxes_df = bboxes_df.transpose()
bboxes_df.columns = ['bbox_list']
bboxes_df.head()
bboxes_df.to_csv('rotated_bbox_dictionary.csv')
 