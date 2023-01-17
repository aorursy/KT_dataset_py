import os 



os.listdir('../input')
!pip install ../input/efficientnetgithub/efficientnet-1.0.0/
import tensorflow as tf
tf.__version__
# general packages

import os

import cv2

import gc

import math

import random

import warnings

import time

import numpy as np

import pandas as pd

from PIL import Image

from glob import glob

import matplotlib.pyplot as plt

import seaborn as sns

# from tqdm.notebook import tqdm

from tqdm import tqdm



#sklearns 

from sklearn.metrics import cohen_kappa_score, accuracy_score, recall_score

from sklearn.model_selection import train_test_split 



from tensorflow.keras.optimizers import Adam, Nadam, SGD

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Model, load_model, Sequential

from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Conv2D, GlobalMaxPooling2D, concatenate

from tensorflow.keras.layers import (MaxPooling2D, Input, Average, Activation, MaxPool2D,

                          Flatten, LeakyReLU, BatchNormalization)

from tensorflow.keras import models

from tensorflow.keras import layers

from tensorflow.keras.preprocessing.image import load_img

from tensorflow.keras.preprocessing.image import img_to_array



from tensorflow.keras.utils import Sequence

from tensorflow.keras import utils as np_utils

from tensorflow.keras.callbacks import (Callback, ModelCheckpoint,

                                        LearningRateScheduler,EarlyStopping, 

                                        ReduceLROnPlateau,CSVLogger)

# from keras_tqdm import TQDMNotebookCallback



warnings.simplefilter('ignore')

sns.set_style('whitegrid')
# declare some parameter

SEED = int(time.time())

epoch = 3

batch_size = 32 

dim = (125, 125)

SIZE = 125

stats = (0.0692, 0.2051)

HEIGHT = 137 

WIDTH = 236



def seed_all(SEED):

    random.seed(SEED)

    np.random.seed(SEED)

    

# seed all

seed_all(SEED)



# load files

im_path = '../input/grapheme-imgs-128x128/'

train = pd.read_csv('../input/bengaliai-cv19/train.csv')

test = pd.read_csv('../input/bengaliai-cv19/test.csv')

train['filename'] = train.image_id.apply(lambda filename: im_path + filename + '.png')



# top 5 samples

train.head()
## Grid Mask

# code takesn from https://www.kaggle.com/haqishen/gridmask



import albumentations

from albumentations.core.transforms_interface import DualTransform, ImageOnlyTransform

from albumentations.augmentations import functional as F



class GridMask(DualTransform):

    """GridMask augmentation for image classification and object detection.



    Args:

        num_grid (int): number of grid in a row or column.

        fill_value (int, float, lisf of int, list of float): value for dropped pixels.

        rotate ((int, int) or int): range from which a random angle is picked. If rotate is a single int

            an angle is picked from (-rotate, rotate). Default: (-90, 90)

        mode (int):

            0 - cropout a quarter of the square of each grid (left top)

            1 - reserve a quarter of the square of each grid (left top)

            2 - cropout 2 quarter of the square of each grid (left top & right bottom)



    Targets:

        image, mask



    Image types:

        uint8, float32



    Reference:

    |  https://arxiv.org/abs/2001.04086

    |  https://github.com/akuxcw/GridMask

    """



    def __init__(self, num_grid=3, fill_value=0, rotate=0, mode=0, always_apply=False, p=0.5):

        super(GridMask, self).__init__(always_apply, p)

        if isinstance(num_grid, int):

            num_grid = (num_grid, num_grid)

        if isinstance(rotate, int):

            rotate = (-rotate, rotate)

        self.num_grid = num_grid

        self.fill_value = fill_value

        self.rotate = rotate

        self.mode = mode

        self.masks = None

        self.rand_h_max = []

        self.rand_w_max = []



    def init_masks(self, height, width):

        if self.masks is None:

            self.masks = []

            n_masks = self.num_grid[1] - self.num_grid[0] + 1

            for n, n_g in enumerate(range(self.num_grid[0], self.num_grid[1] + 1, 1)):

                grid_h = height / n_g

                grid_w = width / n_g

                this_mask = np.ones((int((n_g + 1) * grid_h), int((n_g + 1) * grid_w))).astype(np.uint8)

                for i in range(n_g + 1):

                    for j in range(n_g + 1):

                        this_mask[

                             int(i * grid_h) : int(i * grid_h + grid_h / 2),

                             int(j * grid_w) : int(j * grid_w + grid_w / 2)

                        ] = self.fill_value

                        if self.mode == 2:

                            this_mask[

                                 int(i * grid_h + grid_h / 2) : int(i * grid_h + grid_h),

                                 int(j * grid_w + grid_w / 2) : int(j * grid_w + grid_w)

                            ] = self.fill_value

                

                if self.mode == 1:

                    this_mask = 1 - this_mask



                self.masks.append(this_mask)

                self.rand_h_max.append(grid_h)

                self.rand_w_max.append(grid_w)



    def apply(self, image, mask, rand_h, rand_w, angle, **params):

        h, w = image.shape[:2]

        mask = F.rotate(mask, angle) if self.rotate[1] > 0 else mask

        mask = mask[:,:,np.newaxis] if image.ndim == 3 else mask

        image *= mask[rand_h:rand_h+h, rand_w:rand_w+w].astype(image.dtype)

        return image



    def get_params_dependent_on_targets(self, params):

        img = params['image']

        height, width = img.shape[:2]

        self.init_masks(height, width)



        mid = np.random.randint(len(self.masks))

        mask = self.masks[mid]

        rand_h = np.random.randint(self.rand_h_max[mid])

        rand_w = np.random.randint(self.rand_w_max[mid])

        angle = np.random.randint(self.rotate[0], self.rotate[1]) if self.rotate[1] > 0 else 0



        return {'mask': mask, 'rand_h': rand_h, 'rand_w': rand_w, 'angle': angle}



    @property

    def targets_as_params(self):

        return ['image']



    def get_transform_init_args_names(self):

        return ('num_grid', 'fill_value', 'rotate', 'mode')
# augmix : https://github.com/google-research/augmix



from PIL import Image

from PIL import ImageOps

import numpy as np



def int_parameter(level, maxval):

    """Helper function to scale `val` between 0 and maxval .

    Args:

    level: Level of the operation that will be between [0, `PARAMETER_MAX`].

    maxval: Maximum value that the operation can have. This will be scaled to

      level/PARAMETER_MAX.

    Returns:

    An int that results from scaling `maxval` according to `level`.

    """

    return int(level * maxval / 10)





def float_parameter(level, maxval):

    """Helper function to scale `val` between 0 and maxval.

    Args:

    level: Level of the operation that will be between [0, `PARAMETER_MAX`].

    maxval: Maximum value that the operation can have. This will be scaled to

      level/PARAMETER_MAX.

    Returns:

    A float that results from scaling `maxval` according to `level`.

    """

    return float(level) * maxval / 10.



def sample_level(n):

    return np.random.uniform(low=0.1, high=n)



def autocontrast(pil_img, _):

    return ImageOps.autocontrast(pil_img)



def equalize(pil_img, _):

    return ImageOps.equalize(pil_img)



def posterize(pil_img, level):

    level = int_parameter(sample_level(level), 4)

    return ImageOps.posterize(pil_img, 4 - level)



def rotate(pil_img, level):

    degrees = int_parameter(sample_level(level), 30)

    if np.random.uniform() > 0.5:

        degrees = -degrees

    return pil_img.rotate(degrees, resample=Image.BILINEAR)



def solarize(pil_img, level):

    level = int_parameter(sample_level(level), 256)

    return ImageOps.solarize(pil_img, 256 - level)



def shear_x(pil_img, level):

    level = float_parameter(sample_level(level), 0.3)

    if np.random.uniform() > 0.5:

        level = -level

    return pil_img.transform((SIZE, SIZE),

                           Image.AFFINE, (1, level, 0, 0, 1, 0),

                           resample=Image.BILINEAR)



def shear_y(pil_img, level):

    level = float_parameter(sample_level(level), 0.3)

    if np.random.uniform() > 0.5:

        level = -level

    return pil_img.transform((SIZE, SIZE),

                           Image.AFFINE, (1, 0, 0, level, 1, 0),

                           resample=Image.BILINEAR)



def translate_x(pil_img, level):

    level = int_parameter(sample_level(level), SIZE / 3)

    if np.random.random() > 0.5:

        level = -level

    return pil_img.transform((SIZE, SIZE),

                           Image.AFFINE, (1, 0, level, 0, 1, 0),

                           resample=Image.BILINEAR)





def translate_y(pil_img, level):

    level = int_parameter(sample_level(level), SIZE / 3)

    if np.random.random() > 0.5:

        level = -level

    return pil_img.transform((SIZE, SIZE),

                           Image.AFFINE, (1, 0, 0, 0, 1, level),

                           resample=Image.BILINEAR)



augmentations = [

    autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,

    translate_x, translate_y

]



# taken from https://www.kaggle.com/iafoss/image-preprocessing-128x128

MEAN = [ 0.06922848809290576,  0.06922848809290576,  0.06922848809290576]

STD = [ 0.20515700083327537,  0.20515700083327537,  0.20515700083327537]



def normalize(image):

    """Normalize input image channel-wise to zero mean and unit variance."""

    image = image.transpose(2, 0, 1)  # Switch to channel-first

    mean, std = np.array(MEAN), np.array(STD)

    image = (image - mean[:, None, None]) / std[:, None, None]

    return image.transpose(1, 2, 0)





def apply_op(image, op, severity):

    image = np.clip(image * 255., 0, 255).astype(np.uint8)

    pil_img = Image.fromarray(image)  # Convert to PIL.Image

    pil_img = op(pil_img, severity)

    return np.asarray(pil_img) / 255.





def augment_and_mix(image, severity=1, width=3, depth=1, alpha=1.):

    """Perform AugMix augmentations and compute mixture.

    Args:

    image: Raw input image as float32 np.ndarray of shape (h, w, c)

    severity: Severity of underlying augmentation operators (between 1 to 10).

    width: Width of augmentation chain

    depth: Depth of augmentation chain. -1 enables stochastic depth uniformly

      from [1, 3]

    alpha: Probability coefficient for Beta and Dirichlet distributions.

    Returns:

    mixed: Augmented and mixed image.

  """

    ws = np.float32(

      np.random.dirichlet([alpha] * width))

    m = np.float32(np.random.beta(alpha, alpha))



    mix = np.zeros_like(image)

    for i in range(width):

        image_aug = image.copy()

        depth = depth if depth > 0 else np.random.randint(1, 4)

        

        for _ in range(depth):

            op = np.random.choice(augmentations)

            image_aug = apply_op(image_aug, op, severity)

        mix = np.add(mix, ws[i] * normalize(image_aug), out=mix, 

                     casting="unsafe")



    mixed = (1 - m) * normalize(image) + m * mix

    return mixed
# helper for mixup

def get_rand_bbox(width, height, l):

    r_x = np.random.randint(width)

    r_y = np.random.randint(height)

    r_l = np.sqrt(1 - l)

    r_w = np.int(width * r_l)

    r_h = np.int(height * r_l)

    return r_x, r_y, r_l, r_w, r_h



class GraphemeGenerator(Sequence):

    def __init__(self, data, batch_size, dim, shuffle=True, transform=None, mix_up_alpha = 0.0, cutmix_alpha = 0.0):

        self._data = data

        self._label_1 = pd.get_dummies(self._data['grapheme_root'], 

                                       columns = ['grapheme_root'])

        self._label_2 = pd.get_dummies(self._data['vowel_diacritic'], 

                                       columns = ['vowel_diacritic'])

        self._label_3 = pd.get_dummies(self._data['consonant_diacritic'], 

                                       columns = ['consonant_diacritic'])

        self._list_idx = data.index.values

        self._batch_size = batch_size

        self._dim = dim

        self._shuffle = shuffle

        self.transform = transform

        self.on_epoch_end()

        

        # Mix-up

        assert mix_up_alpha >= 0.0

        self.mix_up_alpha = mix_up_alpha

        

        # Cutmix

        assert cutmix_alpha >= 0.0

        self.cutmix_alpha = cutmix_alpha

        

    def __len__(self):

        return int(np.floor(len(self._data)/self._batch_size))

    

    def __getitem__(self, index):

        batch_idx = self._indices[index * self._batch_size:(index+1) * self._batch_size]

        next_batch_idx = self._indices[(index + 1) * self._batch_size:(index+2) * self._batch_size if index>(len(self._data)-2) 

                                      else (index) * self._batch_size:(index+1) * self._batch_size]

        

        _idx = [self._list_idx[k] for k in batch_idx]

        _next_idx = [self._list_idx[k] for k in next_batch_idx]

        

        X1, y1 = self.__get_data__(_idx)

        X2, y2 = self.__get_data__(_next_idx)

        

        if self.transform is not None:

            randInt = np.random.rand()

            if randInt <= 0.5:

                return self.mix_up(np.array(X1), y1, np.array(X2), y2)

            else:

                return self.cutmix(np.array(X1), y1, np.array(X2), y2)

        else:

            return X1, y1

        

    def on_epoch_end(self):

        self._indices = np.arange(len(self._list_idx))

        if self._shuffle:

            np.random.shuffle(self._indices)

    

    def __get_data__(self, _idx):

        Data     = np.empty((self._batch_size, *self._dim, 1))

        Target_1 = np.empty((self._batch_size, 168), dtype = int)

        Target_2 = np.empty((self._batch_size, 11 ), dtype = int)

        Target_3 = np.empty((self._batch_size,  7 ), dtype = int)

        

        for i, k in enumerate(_idx):

            # load the image file using cv2

            image = cv2.imread(im_path + self._data['image_id'][k] + '.png')

            image = (cv2.resize(image,  self._dim) / 255.0 - stats[0])/stats[1] 

            

            if self.transform is not None:

                randint = np.random.rand()

                if randint <= 0.4:

                    #pass

                    # albumentation : grid mask

                    res = self.transform(image=image)

                    image = res['image']

                elif randint > 0.4 and randint <=0.7:

                    #pass

                    # augmix augmentation

                    image = augment_and_mix(image)

                else:

                    pass

            

            # gray scaling 

            gray = lambda rgb : np.dot(rgb[... , :3] , [0.299 , 0.587, 0.114]) 

            image = gray(image)  

            

            # expand the axises 

            image = image[:, :, np.newaxis]

            Data[i,:, :, :] =  image

        

            Target_1[i,:] = self._label_1.loc[k, :].values

            Target_2[i,:] = self._label_2.loc[k, :].values

            Target_3[i,:] = self._label_3.loc[k, :].values



        return Data, [Target_1, Target_2, Target_3]

    

    def mix_up(self, X1, y1, X2, y2):

        assert X1.shape[0] == X2.shape[0]

        batch_size = X1.shape[0]

        l = np.random.beta(self.mix_up_alpha, self.mix_up_alpha, batch_size)

        X_l = l.reshape(batch_size, 1, 1, 1)

        y_l = l.reshape(batch_size, 1)

        X = X1 * X_l + X2 * (1-X_l)

        return X, y1

    

    def cutmix(self, X1, y1, X2, y2):

        assert X1.shape[0] == X2.shape[0]

        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)

        width = X1.shape[1]

        height = X1.shape[0]

        r_x, r_y, r_l, r_w, r_h = get_rand_bbox(width, height, lam)

        bx1 = np.clip(r_x - r_w // 2, 0, width)

        by1 = np.clip(r_y - r_h // 2, 0, height)

        bx2 = np.clip(r_x + r_w // 2, 0, width)

        by2 = np.clip(r_y + r_h // 2, 0, height)

        X1[:, bx1:bx2, by1:by2, :] = X2[:, bx1:bx2, by1:by2, :]

        X = X1

        return X, y1
import efficientnet.tfkeras as enf



# enf.__dict__

wg = '../input/efficientnet-tfkeras-weights-b0/efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5'

efnet = enf.EfficientNetB0(weights=wg,

                           include_top = False,

                           input_shape=(125, 125, 3))
def create_model(input_dim, output_dim, base_model):

    

    input_tensor = Input(input_dim)

    

    x = Conv2D(3, (3, 3), padding='same',  kernel_initializer='he_uniform', 

               bias_initializer='zeros')(input_tensor)

    curr_output = base_model(x)

    curr_output = GlobalAveragePooling2D()(curr_output)

    curr_output = Dropout(0.3)(curr_output)

    curr_output = Dense(512, activation='elu')(curr_output)

    curr_output = Dropout(0.5)(curr_output)

        

    oputput1 = Dense(168,  activation='softmax', name='gra') (curr_output)

    oputput2 = Dense(11,  activation='softmax', name='vow') (curr_output)

    oputput3 = Dense(7,  activation='softmax', name='cons') (curr_output)

    output_tensor = [oputput1, oputput2, oputput3]



    model = Model(input_tensor, output_tensor)

    

    return model



# building the complete model

model = create_model(input_dim=(125,125,1), output_dim=(168,11,7), base_model = efnet)

model.summary()
# compiling    

model.compile(

    

    optimizer = Adam(learning_rate=0.0001), 

    

    loss = {'gra' : 'categorical_crossentropy', 

            'vow' : 'categorical_crossentropy', 

            'cons': 'categorical_crossentropy'},

    

    loss_weights = {'gra' : 0.5,

                    'vow' : 0.25,

                    'cons': 0.25},

    

    metrics={'gra' : ['accuracy'], 

             'vow' : ['accuracy'], 

             'cons': ['accuracy']}

)
# grid mask augmentation

transforms_train = albumentations.Compose([

    GridMask(num_grid=3, rotate=15, p=1),

])



# for way one - data generator

train_labels, val_labels = train_test_split(train, test_size = 0.20, 

                                            random_state = SEED)



# training generator

train_generator = GraphemeGenerator(train_labels, batch_size, dim, 

                                shuffle = True, transform=transforms_train, mix_up_alpha = 0.4, cutmix_alpha = 0.4)



# validation generator: no shuffle , not augmentation

val_generator = GraphemeGenerator(val_labels, batch_size, dim, 

                              shuffle = False)
def macro_recall(y_true, y_pred):

    return recall_score(y_true, y_pred, average='macro')



class CustomCallback(tf.keras.callbacks.Callback):

    def __init__(self, val_data, batch_size = 32):

        super().__init__()

        self.valid_data = val_data

        self.batch_size = batch_size

    

    def on_epoch_begin(self,epoch, logs={}):

        self.recall_scores = []

        

    def on_epoch_end(self, epoch, logs={}):

        batches = len(self.valid_data)

        total = batches * self.batch_size

        self.val_recalls = {0: [], 1:[], 2:[]}

        

        for batch in range(batches):

            xVal, yVal = self.valid_data.__getitem__(batch)

            val_preds = self.model.predict(xVal)

            

            for i in range(3):

                preds = np.argmax(val_preds[i], axis=1)

                true = np.argmax(yVal[i], axis=1)

                self.val_recalls[i].append(macro_recall(true, preds))

        

        for i in range(3):

            self.recall_scores.append(np.average(self.val_recalls[i]))

            

        print("validation recall score: ", np.average(self.recall_scores, weights=[2, 1, 1]))

        

        return 
from tensorflow.keras.callbacks import (ModelCheckpoint, LearningRateScheduler,

                             EarlyStopping, ReduceLROnPlateau, TensorBoard, CSVLogger)



# some call back function; feel free to add more for experiment

def Call_Back():

    learning_rate_reduction_root = ReduceLROnPlateau(monitor='gra_accuracy', 

                                            patience=3, 

                                            verbose=1,

                                            factor=0.5, 

                                            min_lr=0.000001)

    learning_rate_reduction_vowel = ReduceLROnPlateau(monitor='vow_accuracy', 

                                            patience=3, 

                                            verbose=1,

                                            factor=0.5, 

                                            min_lr=0.000001)

    learning_rate_reduction_consonant = ReduceLROnPlateau(monitor='cons_accuracy', 

                                            patience=3, 

                                            verbose=1,

                                            factor=0.5, 

                                            min_lr=0.000001)

    #lr_scheduler = LearningRateScheduler(schedule=lambda epoch: 0.001 * (0.9 ** epoch))

    csv_logger = CSVLogger('E0Grapheme-B0-1-epochs.csv')

    

    custom_callback = CustomCallback(val_generator)

    

    return [csv_logger,custom_callback, 

            learning_rate_reduction_root, learning_rate_reduction_vowel, learning_rate_reduction_consonant]
# epoch size 

epochs = 1 # increase the number, ex.: 100/200

training = True # setting it true for training the model



# calling all callbacks 

callbacks = Call_Back()



if training:

    # acatual training (fitting)

    train_history = model.fit(

        train_generator,

        steps_per_epoch=int(len(train_labels)/batch_size), 

        validation_data=val_generator,

        validation_steps = int(len(val_labels)/batch_size),

        epochs=epochs,

        callbacks=callbacks, 

    )