# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import glob
print(os.listdir("../input/ucfcrowdcountingdataset_cvpr13_with_people_density_map/UCF_CC_50"))
DATA_PATH = "../input/ucfcrowdcountingdataset_cvpr13_with_people_density_map/UCF_CC_50/"

MODEL_PATH = "single_image_random_crop_experiment_model_1.model"

image_list = glob.glob(DATA_PATH+"*.jpg")

density_list = list(map(lambda s: s.replace(".jpg", ".h5"), image_list))
# from tensorflow.keras.applications.vgg16 import VGG16

from tensorflow.python.keras.applications.vgg16 import VGG16

# from tensorflow.keras.models import Model

# from tensorflow.keras.layers import Conv2D, UpSampling2D



from tensorflow.python.keras.models import Model

from tensorflow.python.keras.layers import Conv2D, UpSampling2D



import numpy as np



from tensorflow.python.keras.optimizers import SGD

# from keras import optimizers





def build_model():

    sgd = SGD(lr=1e-8, decay=5*1e-4, momentum=0.9)

    vgg16_model = VGG16(weights='imagenet', include_top=False)

    # model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)

    x = vgg16_model.get_layer('block4_conv3').output

    x = UpSampling2D(size=(8, 8))(x)

    x = Conv2D(filters=512, kernel_size=(3, 3), dilation_rate=2, padding='same',activation='relu')(x)

    x = Conv2D(filters=512, kernel_size=(3, 3), dilation_rate=2, padding='same', activation='relu')(x)

    x = Conv2D(filters=512, kernel_size=(3, 3), dilation_rate=2, padding='same', activation='relu')(x)

    x = Conv2D(filters=256, kernel_size=(3, 3), dilation_rate=2, padding='same', activation='relu')(x)

    x = Conv2D(filters=128, kernel_size=(3, 3), dilation_rate=2, padding='same', activation='relu')(x)

    x = Conv2D(filters=64, kernel_size=(3, 3), dilation_rate=2, padding='same', activation='relu')(x)

    x = Conv2D(filters=1, kernel_size=(1, 1), dilation_rate=1, padding='same', activation='relu')(x)

    model = Model(inputs=vgg16_model.input, outputs=x)

    model.compile(optimizer=sgd,

                  loss="mean_squared_error", metrics=None)

    return model
model = build_model()
x = np.random.rand(1, 224, 224, 3)

pred = model.predict(x)

print(pred.shape)
# TODO: write keras.utils.Sequence() to load image

from tensorflow.python.keras.utils import Sequence



import numpy as np

import h5py

import PIL.Image as Image





def load_density(file_path):

    gt_file = h5py.File(file_path, 'r')

    groundtruth = np.asarray(gt_file['density'])

    return groundtruth





def random_crop(img, density_map, random_crop_size):

    """

    adapt from https://jkjung-avt.github.io/keras-image-cropping/

    :param img: image matrix (h, w, channel)

    :param density_map: (h, w, channel)

    :param random_crop_size: h_crop, w_crop

    :return:

    """

    # Note: image_data_format is 'channel_last'

    assert img.shape[2] == 3

    height, width = img.shape[0], img.shape[1]

    dy, dx = random_crop_size

    x = np.random.randint(0, width - dx + 1)

    y = np.random.randint(0, height - dy + 1)



    return img[y:(y+dy), x:(x+dx), :], density_map[y:(y+dy), x:(x+dx), :]





# TODO: write keras.utils.Sequence() to load image

from tensorflow.python.keras.utils import Sequence



import numpy as np

import h5py

import PIL.Image as Image





def load_density(file_path):

    gt_file = h5py.File(file_path, 'r')

    groundtruth = np.asarray(gt_file['density'])

    return groundtruth





def random_crop(img, density_map, random_crop_size):

    """

    adapt from https://jkjung-avt.github.io/keras-image-cropping/

    :param img: image matrix (h, w, channel)

    :param density_map: (h, w, channel)

    :param random_crop_size: h_crop, w_crop

    :return:

    """

    # Note: image_data_format is 'channel_last'

    assert img.shape[2] == 3

    height, width = img.shape[0], img.shape[1]

    dy, dx = random_crop_size

    x = np.random.randint(0, width - dx + 1)

    y = np.random.randint(0, height - dy + 1)



    return img[y:(y+dy), x:(x+dx), :], density_map[y:(y+dy), x:(x+dx), :]





class DatasetSequence(Sequence):



    def __init__(self, image_path_list, density_path_list, random_crop_size=None):

        self.image_path_list = image_path_list

        self.density_path_list = density_path_list

        self.random_crop_size = random_crop_size



    def __len__(self):

        return len(self.image_path_list)



    def __getitem__(self, idx):

        image_path = self.image_path_list[idx]

        density_path = self.density_path_list[idx]



        density = load_density(density_path)

        image = np.array(Image.open(image_path, "r").convert("RGB"))

        density = np.expand_dims(density, axis=3)  # add channel dim



        if self.random_crop_size is not None:

            image, density = random_crop(image, density, self.random_crop_size)



        image = np.expand_dims(image, axis=0) # add batch dim

        density = np.expand_dims(density, axis=0) # add batch dim



        return image, density



    def get_random_crop_image(self, idx):

        image_path = self.image_path_list[idx]

        density_path = self.density_path_list[idx]



        density = load_density(density_path)

        image = np.array(Image.open(image_path, "r").convert("RGB"))

        density = np.expand_dims(density, axis=3)  # add channel dim



        if self.random_crop_size is not None:

            # print("crop ", self.random_crop_size)

            image, density = random_crop(image, density, self.random_crop_size)



        image = np.expand_dims(image, axis=0)  # add batch dim

        density = np.expand_dims(density, axis=0)  # add batch dim



        return image, density



    def get_random_crop_image_batch(self, idx, batch_size):

        image_batch = []

        density_batch = []

        for i in range(batch_size):

            image, density = self.get_random_crop_image(idx)

            image_batch.append(image)

            density_batch.append(density)

        images = np.concatenate(image_batch, axis=0)

        densities = np.concatenate(density_batch, axis=0)

        return images, densities

    

    def get_all(self, crop_per_img):

        image_list = []

        density_list = []

        for i in range(len(self.image_path_list)):

            image, density = self.get_random_crop_image_batch(i, crop_per_img)

            image_list.append(image)

            density_list.append(density)

        image_mat = np.concatenate(image_list, axis = 0)

        density_mat = np.concatenate(density_list, axis = 0)

        return image_mat, density_mat



dataset = DatasetSequence(image_list, density_list, random_crop_size=(224, 224))

img_mat, density_mat = dataset.get_all(10)

print(img_mat.shape)

print(density_mat.shape)
dataset = DatasetSequence(image_list, density_list, random_crop_size=(224, 224))

model = build_model()



# for image, density in dataset:

#     model.fit(image, density)

epoch = 10

n_sample = len(dataset)

print("total epoch ", epoch)

print("sample ", n_sample)

img_train, density_train = dataset.get_all(10)

model.fit(img_train, density_train, batch_size=8, shuffle=True, epochs = epoch)

model.save(MODEL_PATH)
print("meow")
from tensorflow.python.keras.models import load_model

import glob

import PIL.Image as Image

from matplotlib import pyplot as plt

from matplotlib import cm as CM

import os

import numpy as np





def save_density_map(density_map, name):

    plt.figure(dpi=600)

    plt.axis('off')

    plt.margins(0, 0)

    plt.imshow(density_map, cmap=CM.jet)

    plt.savefig(name, dpi=600, bbox_inches='tight', pad_inches=0)

dataset.random_crop_size=(448, 448)

img_train, density_train = dataset.get_random_crop_image(3)

pil_img = Image.fromarray(img_train[0])







model = load_model(MODEL_PATH)



print("label ", density_train.sum())





pred = model.predict(img_train)





print("predict ", np.squeeze(pred[0], axis=2).shape, np.squeeze(pred[0], axis=2).sum())



print("------------")
pil_img.save("train.png")

from matplotlib import pyplot as plt





plt.figure(dpi=600)

plt.axis('off')

plt.margins(0,0)

plt.imshow(Image.open("train.png"))
save_density_map(np.squeeze(density_train[0], axis=2), "label.png")
np.squeeze(pred[0], axis=2)
save_density_map(np.squeeze(pred[0], axis=2), "predict.png")
pred[0].shape
np.savetxt("pred_np.txt",np.squeeze(density_train[0], axis=2))