import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
ship_dir = '../input'
train_image_dir = os.path.join(ship_dir, 'train')
test_image_dir = os.path.join(ship_dir, 'test')
boundaries = pd.read_csv(os.path.join(ship_dir, 'train_ship_segmentations.csv'))
boundaries.head()
not_empty = pd.notna(boundaries.EncodedPixels)
print("{} images(has ships) with {} masks".format(boundaries[not_empty].ImageId.nunique(), not_empty.sum()))
print('{} images(with no ships) in {} total images'.format((~not_empty).sum(), boundaries.ImageId.nunique()))
# GET IMAGE PATHS
image_paths = []
for root, dirs, files in os.walk(train_image_dir):
    for imgname in files:
        image_paths.append(os.path.join(root, imgname))
print(len(image_paths), "Images found")
print("Sample image path:", image_paths[0])
# PARAMETERS
MASK_SHAPE = (768, 768)
SAMPLE_SIZE = 20
# Boundaries to MASK per image
def rle_to_mask_per_image(boundaries):
    mask_dict = {}
    for (imgID, rle) in boundaries.itertuples(index=False):
        
        # add new key
        if imgID not in mask_dict.keys():
            mask_dict[imgID] = []
        else:
            pass
        
        # Create empty mask
        _mask = np.zeros(MASK_SHAPE, dtype=np.uint8)
        
        # create mask from boundaries
        if str(rle) == 'nan':
            mask_dict[imgID].append(_mask)
        else:
            rle = rle.split(' ')
            for i in range(0, len(rle)-1, 2):
                _mask = _mask.flatten()
                # whiten given pixels
                _mask[int(rle[i]): int(rle[i])+int(rle[i+1])] = 255
            mask_dict[imgID].append(_mask.reshape(MASK_SHAPE))
    
    # merge masks into one mask of single image
    mask_per_image_dict = {}        
    for key in mask_dict.keys():
        # Empty mask for joining all masks
        mastermask = np.zeros(MASK_SHAPE, dtype=np.uint8)
        for mask in mask_dict[key]:
            mastermask |= mask.T
        mask_per_image_dict[key] = mastermask
    return mask_per_image_dict
# GET SAMPLE IMAGES AND MASKS
sample_masks_dict = rle_to_mask_per_image(boundaries.head(SAMPLE_SIZE))
# SHARPEN IMAGES
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
def sharpen(img):
    return cv2.filter2D(img, -1, kernel)
for key in sample_masks_dict.keys():
    plt.figure(figsize=(15,20))
    plt.subplot(1,2,1)
    plt.axis('off')
    plt.title("Image")
    img = plt.imread(os.path.join(train_image_dir, key))
    # Make image sharp if needed
    #img = sharpen(img)
    plt.imshow(img)
    plt.subplot(1,2,2)
    plt.axis('off')
    plt.title("Mask")
    plt.imshow(sample_masks_dict[key], cmap='gray')
    plt.show()
    
