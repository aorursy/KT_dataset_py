import os
import cv2
import numpy as np
import pandas as pd 
import warnings
warnings.filterwarnings("ignore")
print(os.listdir("../input"))

DATA_PATH = '../input'
TEST_DATA = os.path.join(DATA_PATH, "test")
TRAIN_DATA = os.path.join(DATA_PATH, "train")
TRAIN_MASKS_DATA = os.path.join(TRAIN_DATA, "masks")

df = pd.read_csv(DATA_PATH+'/train_ship_segmentations.csv')

path_train = '../input/train/'
path_test = '../input/test/'
train_ids = df.ImageId.values

df = df.set_index('ImageId')
def get_filename(image_id, image_type):
    check_dir = False
    if "Train" == image_type:
        data_path = TRAIN_DATA
    elif "mask" in image_type:
        data_path = TRAIN_MASKS_DATA
    elif "Test" in image_type:
        data_path = TEST_DATA
    else:
        raise Exception("Image type '%s' is not recognized" % image_type)

    if check_dir and not os.path.exists(data_path):
        os.makedirs(data_path)

    return os.path.join(data_path, "{}".format(image_id))

def get_image_data(image_id, image_type, **kwargs):
    img = _get_image_data_opencv(image_id, image_type, **kwargs)
    img = img.astype('uint8')
    return img

def _get_image_data_opencv(image_id, image_type, **kwargs):
    fname = get_filename(image_id, image_type)
    img = cv2.imread(fname)
    assert img is not None, "Failed to read image : %s, %s" % (image_id, image_type)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img