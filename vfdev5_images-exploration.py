# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import os

from glob import glob

TRAIN_DATA = "../input/train-tif"

TEST_DATA = "../input/test-tif"



train_image_files = glob(os.path.join(TRAIN_DATA, "*.tif"))

train_image_ids = np.array([s[len(os.path.join(TRAIN_DATA, 'train_')):-4] for s in train_image_files])



test_image_files = glob(os.path.join(TEST_DATA, "*.tif"))

test_image_ids = np.array([s[len(os.path.join(TEST_DATA, 'test_')):-4] for s in test_image_files])





print(len(train_image_files), len(test_image_files))

print("\n Train images \n", train_image_files[:10])

print("\n Test images \n", test_image_files[:10])



import tifffile



def get_filename(image_id, image_type, ext='tif'):

    """

    Method to get image file path from its id and type   

    """

    if image_type == "Train":

        data_path = TRAIN_DATA

        prefix='train'

    elif image_type == "Test":

        data_path = TEST_DATA

        prefix='test'

    else:

        raise Exception("Image type '%s' is not recognized" % image_type)

    return os.path.join(data_path, "{}_{}.{}".format(prefix, image_id, ext))





def get_image_data(image_id, image_type):

    """

    Method to get image data as np.array specifying image id and type

    """

    fname = get_filename(image_id, image_type)

    img = tifffile.imread(fname)

    return img





def normalize(in_img, q_min=0.5, q_max=99.5, return_mins_maxs=False):

    """

    Normalize image in [0.0, 1.0]

    mins is array of minima

    maxs is array of differences between maxima and minima

    """

    init_shape = in_img.shape

    if len(init_shape) == 2:

        in_img = np.expand_dims(in_img, axis=2)

    w, h, d = in_img.shape

    img = in_img.copy()

    img = np.reshape(img, [w * h, d]).astype(np.float64)

    mins = np.percentile(img, q_min, axis=0)

    maxs = np.percentile(img, q_max, axis=0) - mins

    maxs[(maxs < 0.0001) & (maxs > -0.0001)] = 0.0001

    img = (img - mins[None, :]) / maxs[None, :]

    img = img.clip(0.0, 1.0)

    img = np.reshape(img, [w, h, d])

    if init_shape != img.shape:

        img = img.reshape(init_shape)

    if return_mins_maxs:

        return img, mins, maxs

    return img





def scale_percentile(matrix, q_min=0.5, q_max=99.5):

    is_gray = False

    if len(matrix.shape) == 2:

        is_gray = True

        matrix = matrix.reshape(matrix.shape + (1,))

    matrix = (255*normalize(matrix, q_min, q_max)).astype(np.uint8)

    if is_gray:

        matrix = matrix.reshape(matrix.shape[:2])

    return matrix  





import matplotlib.pylab as plt





def plt_st(l1,l2):

    plt.figure(figsize=(l1,l2))
import cv2



tile_size = (256, 256)

n = 15

ll = min(len(train_image_ids), 1000)



m = int(np.ceil(ll * 1.0 / n))

complete_image = np.zeros((m*(tile_size[0]+2), n*(tile_size[1]+2), 4), dtype=np.uint16)

train_ids = sorted(train_image_ids)

counter = 0

for i in range(m):

    ys = i*(tile_size[1] + 2)

    ye = ys + tile_size[1]

    for j in range(n):

        xs = j*(tile_size[0] + 2)

        xe = xs + tile_size[0]

        if counter == len(train_ids):

            break

        image_id = train_ids[counter]; counter+=1

        try:

            img = get_image_data(image_id, 'Train')

        except ValueError as e:

            print(e, "Problem with image: ", image_id)

        if img.shape[:2] != tile_size:

            img = cv2.resize(img, dsize=tile_size)

        img = cv2.putText(img, image_id, (5,img.shape[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), thickness=3)

        complete_image[ys:ye, xs:xe, :] = img[:,:,:]

    if counter == len(train_ids):

        break

m = complete_image.shape[0] / (tile_size[0] + 2)

n = int(np.ceil(m / 20.0))

for i in range(n):

    plt_st(20, 20)

    ys = i*(tile_size[0] + 2)*20

    ye = min((i+1)*(tile_size[0] + 2)*20, complete_image.shape[0])

    plt.imshow(scale_percentile(complete_image[ys:ye,:,:3]))

    plt.title("Training dataset, part %i" % i)

from PIL import Image



img_pil = Image.open('../input/train-jpg/train_1000.jpg')

img = np.asarray(img_pil)

print(img.shape, img.dtype)

img2 = get_image_data('1000', 'Train')

print(img2.shape, img2.dtype)

plt_st(12, 12)

plt.subplot(121)

plt.imshow(scale_percentile(img[:,:,:3]))

plt.subplot(122)

plt.imshow(scale_percentile(img2[:,:,:3]))