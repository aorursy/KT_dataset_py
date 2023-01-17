import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, ReLU, MaxPool2D, Add, Dense, Dropout, Flatten, GlobalAveragePooling2D
import tensorflow.compat.v1 as tf1
from tensorflow.keras.mixed_precision import experimental as mixed_precision


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import cv2

import numpy as np

import PIL
from PIL import Image, ImageDraw, ImageEnhance

import albumentations as albu

from tqdm.auto import tqdm

import random
random.seed(42)

from warnings import filterwarnings

%matplotlib inline
filterwarnings('ignore')
# config = tf1.ConfigProto()
# config.gpu_options.allow_growth=True
# sess = tf1.Session(config=config)
# tf1.keras.backend.set_session(session=sess)
# gpu_options = tf1.GPUOptions(per_process_gpu_memory_fraction=0.9999)
# sess = tf1.Session(config=tf1.ConfigProto(gpu_options=gpu_options))
# tf1.keras.backend.set_session(session=sess)
policy = mixed_precision.Policy('mixed_bfloat16')
mixed_precision.set_policy(policy)
# OpenCV with GPU variable
use_GPU = False
df_train = pd.read_csv("../input/global-wheat-detection/train.csv")
df_train.head(5)
df_train.isnull().sum()
image_id_values = df_train["image_id"].unique()
len(image_id_values)
train_image_ids = image_id_values[0:3363]
val_image_ids = image_id_values[3363:3373]
def group_boxes(group):
    boundaries = group['bbox'].str.split(',', expand=True)
    boundaries[0] = boundaries[0].str.slice(start=1)
    boundaries[3] = boundaries[3].str.slice(stop=-1)
    
    return boundaries.values.astype(float)

bboxes = df_train.groupby('image_id').apply(group_boxes)
bboxes['bce2fdc4d'][:5]
def load_image(image_id):
    
    global use_GPU
    if use_GPU:
        image = cv2.UMat(cv2.imread('../input/global-wheat-detection/train/'+image_id+'.jpg'))
    else:
        image = cv2.imread('../input/global-wheat-detection/train/'+image_id+'.jpg')
        
    if use_GPU:
        if len(cv2.UMat.get(image)) == 0:
            raise ValueError(f"Image could not be located")
    else:
        if len(image) == 0:
            raise ValueError(f"Image could not be located")
    image = cv2.resize(image, (256, 256))
    
    if use_GPU:
        return cv2.UMat.get(image)
    else:
        return image
def load_image(image_id):
    image = Image.open('../input/global-wheat-detection/train/' + image_id + ".jpg")
    image = image.resize((256, 256))
    
    return np.asarray(image)
train_pixels = {}
train_labels = {}

for image_id in tqdm(train_image_ids):
    
    train_pixels[image_id] = load_image(image_id)
    train_labels[image_id] = bboxes[image_id].copy() / 4
val_pixels = {}
val_labels = {}

for image_id in tqdm(val_image_ids):
    val_pixels[image_id] = load_image(image_id)
    val_labels[image_id] = bboxes[image_id].copy() / 4
type(train_pixels['b6ab77fd7'])
len(train_labels['b6ab77fd7'])
def draw_bboxes(image_id, bboxes, source='train'):
    if use_GPU:
        image = cv2.UMat(cv2.imread('../input/global-wheat-detection/'+source+'/'+image_id+'.jpg'))
    else:
        image = cv2.imread('../input/global-wheat-detection/'+source+'/'+image_id+'.jpg')
    image = cv2.resize(image, (256, 256))
    
#     cv2.imshow('image', image)
#     cv2.waitKey()
#     cv2.destroyAllWindows()
    
    for bbox in bboxes:
        image = draw_bbox(image, bbox)
        
    return image
def draw_bbox(image, bbox):
    x, y, w, h = bbox
    image = cv2.rectangle(image,
                          (int(x), int(y)),
                          (int(x+w), int(y+h)),
                          (225,0,0), 
                          1)
    return image
def show_images(image_ids, bboxes, source='train'):
    pixels = []
    global use_GPU
    
    for image_id in image_ids:
        pixels.append(
            draw_bboxes(image_id, bboxes[image_id], source)
        )
        
    num_of_images = len(image_ids)
    fig, axes = plt.subplots(1, num_of_images, figsize=(10*num_of_images, 10*num_of_images))
    
    for i, image_pixels in enumerate(pixels):
        if use_GPU:
            axes[i].imshow(cv2.UMat.get(image_pixels))
        else:
            axes[i].imshow(image_pixels)
    #plt.show()
%%time
show_images((train_image_ids[100:102]),(train_labels))
def draw_grid_lines(image_id, labels, grid_line_distance=32, source='train'):
    global use_GPU
    
    image_path = '../input/global-wheat-detection/'+source+'/'+image_id+'.jpg'
    if use_GPU:
        image = cv2.UMat(draw_bboxes(image_id, labels))
    else:
        image = draw_bboxes(image_id, labels)
    image = cv2.resize(image, (256, 256))
    
    # Vertical Lines
    prev_x = 0
    prev_y = 0
    for dist in range(0, 256, grid_line_distance):
        image = cv2.line(image, (prev_x, prev_y+(dist-prev_y)), (prev_x+256, prev_y+(dist-prev_y)), (0, 0, 255), 1, 1)
        prev_x = prev_x
        prev_y = dist
    
    # Horizontal Lines
    prev_x = 0
    prev_y = 0
    for dist in range(0, 256, grid_line_distance):
        image = cv2.line(image, (prev_x+(dist-prev_x), prev_y), (prev_x+(dist-prev_x), prev_y+256), (0, 0, 255), 1, 1)
        prev_x = dist
        prev_y = prev_y
        
    if use_GPU:
        image = cv2.UMat.get(image)
        
    else:
        image = image

    return image
        
fig, ax = plt.subplots(1, 2, figsize=(10*2, 10*2))
image_id = train_image_ids[3000]
ax[0].set_title('Grid size: 32*32')
ax[0].imshow(draw_grid_lines(image_id, train_labels[image_id], grid_line_distance=32 )) # 32*32 grid
ax[1].set_title('Grid size: 16*16')
ax[1].imshow(draw_grid_lines(image_id, train_labels[image_id], grid_line_distance=16 )) # 16*16 grid
fig.show()
train_image_ids[100]
tiny_boxes = []

for i, image_id in enumerate(train_image_ids):
    for label in train_labels[image_id]:
        if label[2]*label[3] <= 10 and label[2]*label[3] != 0:
            tiny_boxes.append(i)
print(str(len(tiny_boxes)) + ' tiny boxes found')
def clean_labels(train_image_ids, train_labels):
    good_labels = {}
    
    for i, image_id in enumerate(train_image_ids):
        good_labels[image_id] = []
        
        for j, label in enumerate(train_labels[image_id]):
            
            if label[2]*label[3] > 8000 and i not in [1079, 1371, 2020]:
                continue
                
            elif label[2]<5 or label[3]<5:
                continue
                
            else:
                good_labels[image_id].append(
                    train_labels[image_id][j]
                )
    return good_labels
train_labels = clean_labels(train_image_ids, train_labels)
class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, image_ids, image_pixels, labels=None, batch_size=1, shuffle=False, augment=False):
        self.image_ids = image_ids
        self.image_pixels = image_pixels
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.on_epoch_end()
        
        self.image_grid = self.form_image_grid()
        
        
    def form_image_grid(self):    
        image_grid = np.zeros((32, 32, 4))

        # x, y, width, height
        cell = [0, 0, 256 / 32, 256 / 32] 

        for i in range(0, 32):
            for j in range(0, 32):
                image_grid[i,j] = cell

                cell[0] = cell[0] + cell[2]

            cell[0] = 0
            cell[1] = cell[1] + cell[3]

        return image_grid
def __len__(self):
    return int(np.floor(len(self.image_ids) / self.batch_size))


def on_epoch_end(self):
    self.indexes = np.arange(len(self.image_ids))

    if self.shuffle == True:
        np.random.shuffle(self.indexes)


DataGenerator.__len__ = __len__
DataGenerator.on_epoch_end = on_epoch_end
DataGenerator.train_augmentations = albu.Compose([
        albu.RandomSizedCrop(
            min_max_height=(200, 200), 
            height=256, 
            width=256, 
            p=0.8
        ),
        albu.OneOf([
            albu.Flip(),
            albu.RandomRotate90(),
        ], p=1),
        albu.OneOf([
            albu.HueSaturationValue(),
            albu.RandomBrightnessContrast()
        ], p=1),
        albu.OneOf([
            albu.GaussNoise(),
            albu.GlassBlur(),
            albu.ISONoise(),
            albu.MultiplicativeNoise(),
        ], p=0.5),
        albu.Cutout(
            num_holes=8, 
            max_h_size=16, 
            max_w_size=16, 
            fill_value=0, 
            p=0.5
        ),
        albu.CLAHE(p=1),
        albu.ToGray(p=1),
    ], 
    bbox_params={'format': 'coco', 'label_fields': ['labels']})

DataGenerator.val_augmentations = albu.Compose([
    albu.CLAHE(p=1),
    albu.ToGray(p=1),
])
def __getitem__(self, index):
    indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

    batch_ids = [self.image_ids[i] for i in indexes]

    X, y = self.__data_generation(batch_ids)

    return X, y


def __data_generation(self, batch_ids):
    X, y = [], []

    # Generate data
    for i, image_id in enumerate(batch_ids):
        pixels = self.image_pixels[image_id]
        bboxes = self.labels[image_id]

        if self.augment:     
            pixels, bboxes = self.augment_image(pixels, bboxes)
        else:
            pixels = self.contrast_image(pixels)
            bboxes = self.form_label_grid(bboxes)

        X.append(pixels)
        y.append(bboxes)

    return np.array(X), np.array(y)


def augment_image(self, pixels, bboxes):
    bbox_labels = np.ones(len(bboxes))

    aug_result = self.train_augmentations(image=pixels, bboxes=bboxes, labels=bbox_labels)

    bboxes = self.form_label_grid(aug_result['bboxes'])

    return np.array(aug_result['image']) / 255, bboxes


def contrast_image(self, pixels):        
    aug_result = self.val_augmentations(image=pixels)
    return np.array(aug_result['image']) / 255


DataGenerator.__getitem__ = __getitem__
DataGenerator.__data_generation = __data_generation
DataGenerator.augment_image = augment_image
DataGenerator.contrast_image = contrast_image
def form_label_grid(self, bboxes):
    label_grid = np.zeros((32, 32, 10))

    for i in range(0, 32):
        for j in range(0, 32):
            cell = self.image_grid[i,j]
            label_grid[i,j] = self.rect_intersect(cell, bboxes)

    return label_grid


def rect_intersect(self, cell, bboxes): 
    cell_x, cell_y, cell_width, cell_height = cell
    cell_x_max = cell_x + cell_width 
    cell_y_max = cell_y + cell_height
    
    anchor_one = np.array([0, 0, 0, 0, 0])
    anchor_two = np.array([0, 0, 0, 0, 0])

    # check all boxes
    for bbox in bboxes:
        box_x, box_y, box_width, box_height = bbox
        box_x_centre = box_x + (box_width / 2)
        box_y_centre = box_y + (box_height / 2)

        if(box_x_centre >= cell_x and box_x_centre < cell_x_max and box_y_centre >= cell_y and box_y_centre < cell_y_max):
            
            if anchor_one[0] == 0:
                anchor_one = self.yolo_shape(
                    [box_x, box_y, box_width, box_height], 
                    [cell_x, cell_y, cell_width, cell_height]
                )
            
            elif anchor_two[0] == 0:
                anchor_two = self.yolo_shape(
                    [box_x, box_y, box_width, box_height], 
                    [cell_x, cell_y, cell_width, cell_height]
                )
                
            else:
                break

    return np.concatenate((anchor_one, anchor_two), axis=None)


def yolo_shape(self, box, cell):
    box_x, box_y, box_width, box_height = box
    cell_x, cell_y, cell_width, cell_height = cell

    # top left x,y to centre x,y
    box_x = box_x + (box_width / 2)
    box_y = box_y + (box_height / 2)

    # offset bbox x,y to cell x,y
    box_x = (box_x - cell_x) / cell_width
    box_y = (box_y - cell_y) / cell_height

    # bbox width,height relative to cell width,height
    box_width = box_width / 256
    box_height = box_height / 256

    return [1, box_x, box_y, box_width, box_height]


DataGenerator.form_label_grid = form_label_grid
DataGenerator.rect_intersect = rect_intersect
DataGenerator.yolo_shape = yolo_shape
BATCH_SIZE = 6

train_generator = DataGenerator(
    train_image_ids,
    train_pixels,
    train_labels, 
    batch_size=BATCH_SIZE, 
    shuffle=True,
    augment=True
)

val_generator = DataGenerator(
    val_image_ids, 
    val_pixels,
    val_labels, 
    batch_size=10,
    shuffle=False,
    augment=False
)

image_grid = train_generator.image_grid
### Using Kaggle TPU for taining.
### For Using Kaggle TPU for taining.
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)

tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
def add_main_block(num_times_to_add_block: int, x, x_shortcut, filters: int, kernel_size: tuple, strides: tuple, block_number=None, padding: str ='same', alpha: str = 0.1):
    
    for _ in range(num_times_to_add_block):
        x = Conv2D(filters, kernel_size, strides=strides, padding=padding)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=alpha)(x)
    
        x = Conv2D(filters*2, kernel_size, strides=strides, padding=padding)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=alpha)(x)
    
        x = Add()([x_shortcut, x])
        x = LeakyReLU(alpha=alpha)(x)
        
        x_shortcut = x
        
    return x, x_shortcut



def add_res_block(x, x_shortcut=None, filters: int = None, kernel_size: tuple = None, strides: tuple = None, block_number=None, padding: str = 'same', alpha: float = 0.1):
    x = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding=padding)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=alpha)(x)

    x_shortcut = x
    
    return x, x_shortcut


def custom_loss(y_true, y_pred):
    
    global BATCH_SIZE
    
    #------------- For GPU Computing ----------------#
#     binary_crossentropy = prob_loss = tf.keras.losses.BinaryCrossentropy(
#         reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
#     )

#     prob_loss = binary_crossentropy(
#         tf.concat([y_true[:,:,:,0], y_true[:,:,:,5]], axis=0), 
#         tf.concat([y_pred[:,:,:,0], y_pred[:,:,:,5]], axis=0)
#     )
    #-------------- End of GPU Computing ---------------#
    
    
    
    #------------ For TPU Computing ------------#
    binary_crossentropy = prob_loss = tf.keras.losses.BinaryCrossentropy(
        reduction=tf.keras.losses.Reduction.NONE)
    
    # Binary Cross Entropy loss for TPU Computing.
    reduce_sum = tf.reduce_sum(binary_crossentropy(
        tf.concat([y_true[:,:,:,0], y_true[:,:,:,5]], axis=0),
        tf.concat([y_pred[:,:,:,0], y_pred[:,:,:,5]], axis=0)
    ))
    
    prob_loss = reduce_sum * (1. / BATCH_SIZE)
    #-------------- End of TPU Computing --------------#
    
    xy_loss = tf.keras.losses.MSE(
        tf.concat([y_true[:,:,:,1:3], y_true[:,:,:,6:8]], axis=0), 
        tf.concat([y_pred[:,:,:,1:3], y_pred[:,:,:,6:8]], axis=0)
    )
    
    wh_loss = tf.keras.losses.MSE(
        tf.concat([y_true[:,:,:,3:5], y_true[:,:,:,8:10]], axis=0), 
        tf.concat([y_pred[:,:,:,3:5], y_pred[:,:,:,8:10]], axis=0)
    )
    
    bboxes_mask = get_mask(y_true)
    
    xy_loss = xy_loss * bboxes_mask
    wh_loss = wh_loss * bboxes_mask
    
    return prob_loss + xy_loss + wh_loss


def get_mask(y_true):
    anchor_one_mask = tf.where(
        y_true[:,:,:,0] == 0, 
        0.5, 
        5.0
    )
    
    anchor_two_mask = tf.where(
        y_true[:,:,:,5] == 0, 
        0.5, 
        5.0
    )
    
    bboxes_mask = tf.concat(
        [anchor_one_mask,anchor_two_mask],
        axis=0
    )
    
    return bboxes_mask


with tpu_strategy.scope():


    x_input = Input(shape=(256, 256, 3))

    x = Conv2D(32, (3, 3), strides=(1, 1), padding='same')(x_input)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)


    #------------ Block 1 -------------#
    #-- Res-block-1 --#
    x, x_shortcut = add_res_block(x, filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', alpha=0.1)
    #-- Main-Block-1 --#
    x, x_shortcut = add_main_block(2, x, x_shortcut, filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', alpha=0.1)


    #------------ Block 2 -------------#
    #-- Res-block-2 --#
    x, x_shortcut = add_res_block(x, x_shortcut=x_shortcut, filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', alpha=0.1)
    #-- Main-Block 2 --#
    x, x_shortcut = add_main_block(2, x, x_shortcut, filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', alpha=0.1)


    #------------ Block 3 -------------#
    #-- Res-block 3 --#
    x, x_shortcut = add_res_block(x, x_shortcut, filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', alpha=0.1)
    #-- Main-Block --#
    x, x_shortcut = add_main_block(8, x, x_shortcut, filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', alpha=0.1)


    #------------ Block 4 -------------#
    #-- Res-Block 4 --#
    x, x_shortcut = add_res_block(x, x_shortcut, filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', alpha=0.1)
    #-- Main-Block 4 --#
    x, x_shortcut = add_main_block(8, x, x_shortcut, filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', alpha=0.1)


    #------------ Block 5 -------------#
    #-- Res-Block 5 --#
    x, x_shortcut = add_res_block(x, x_shortcut, filters=1024, kernel_size=(3, 3), strides=(1, 1), padding='same', alpha=0.1)
    #-- Main-Block 5 --#
    x, x_shortcut = add_main_block(4, x, x_shortcut, filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', alpha=0.1)


    #------------ Output Layers -------------#
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

    predictions= x = Conv2D(10, (1, 1), strides=(1, 1), activation='sigmoid', dtype="float32")(x)
    
    optimiser = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model = tf.keras.Model(inputs=x_input, outputs=predictions)
    

    model.compile(
        optimizer=optimiser, 
        loss=custom_loss
    )
x_input = tf.keras.Input(shape=(256,256,3))

x = tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same')(x_input)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

########## block 1 ##########
x = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

x_shortcut = x

for i in range(2):
    x = tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    x = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    x = tf.keras.layers.Add()([x_shortcut, x])
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    x_shortcut = x


########## block 2 ##########
x = tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

x_shortcut = x

for i in range(2):
    x = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    x = tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    x = tf.keras.layers.Add()([x_shortcut, x])
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    x_shortcut = x

########## block 3 ##########
x = tf.keras.layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

x_shortcut = x

for i in range(8):
    x = tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    x = tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    x = tf.keras.layers.Add()([x_shortcut, x])
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    x_shortcut = x

    
########## block 4 ##########
x = tf.keras.layers.Conv2D(512, (3, 3), strides=(2, 2), padding='same')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

x_shortcut = x

for i in range(8):
    x = tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    x = tf.keras.layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    x = tf.keras.layers.Add()([x_shortcut, x])
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    x_shortcut = x

########## block 5 ##########
x = tf.keras.layers.Conv2D(1024, (3, 3), strides=(2, 2), padding='same')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

x_shortcut = x

for i in range(4):
    x = tf.keras.layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    x = tf.keras.layers.Conv2D(1024, (3, 3), strides=(1, 1), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    x = tf.keras.layers.Add()([x_shortcut, x])
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    x_shortcut = x

########## output layers ##########
x = tf.keras.layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

x = tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

x = tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

predictions = tf.keras.layers.Conv2D(10, (1, 1), strides=(1, 1), activation='sigmoid')(x)

model = tf.keras.Model(inputs=x_input, outputs=predictions)
def custom_loss(y_true, y_pred):
    binary_crossentropy = prob_loss = tf.keras.losses.BinaryCrossentropy(
        reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
    )
    
    prob_loss = binary_crossentropy(
        tf.concat([y_true[:,:,:,0], y_true[:,:,:,5]], axis=0), 
        tf.concat([y_pred[:,:,:,0], y_pred[:,:,:,5]], axis=0)
    )
    
    xy_loss = tf.keras.losses.MSE(
        tf.concat([y_true[:,:,:,1:3], y_true[:,:,:,6:8]], axis=0), 
        tf.concat([y_pred[:,:,:,1:3], y_pred[:,:,:,6:8]], axis=0)
    )
    
    wh_loss = tf.keras.losses.MSE(
        tf.concat([y_true[:,:,:,3:5], y_true[:,:,:,8:10]], axis=0), 
        tf.concat([y_pred[:,:,:,3:5], y_pred[:,:,:,8:10]], axis=0)
    )
    
    bboxes_mask = get_mask(y_true)
    
    xy_loss = xy_loss * bboxes_mask
    wh_loss = wh_loss * bboxes_mask
    
    return prob_loss + xy_loss + wh_loss


def get_mask(y_true):
    anchor_one_mask = tf.where(
        y_true[:,:,:,0] == 0, 
        0.5, 
        5.0
    )
    
    anchor_two_mask = tf.where(
        y_true[:,:,:,5] == 0, 
        0.5, 
        5.0
    )
    
    bboxes_mask = tf.concat(
        [anchor_one_mask,anchor_two_mask],
        axis=0
    )
    
    return bboxes_mask
optimiser = tf.keras.optimizers.Adam(learning_rate=0.0001)

model.compile(
    optimizer=optimiser, 
    loss=custom_loss
)
callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=2, verbose=1),
    tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, verbose=1, restore_best_weights=True),
    keras.callbacks.ModelCheckpoint(filepath="Model_save_diff_user.h5", save_best_only=True, monitor='loss', save_weights_only=True, verbose=1)
]
model.summary()
model.load_weights("./Model_save_diff_user.h5")
model_json = model.to_json()
with open("Model_save_structure_diff_user.h5", "w") as json_file:
    json_file.write(model_json)
model.save_weights("Model_save_diff_user.h5")
history = model.fit_generator(
    train_generator,
    validation_data=val_generator,
    epochs=80,
    callbacks=callbacks
)
import tensorflow as tf
print(tf.__version__)
import itertools

def gen(): 
  for i in itertools.count(1): 
    yield (i, [1] * i) 

dataset = tf.data.Dataset.from_generator( 
     gen, 
     (tf.int64, tf.int64), 
     (tf.TensorShape([]), tf.TensorShape([None]))) 

list(dataset.take(3).as_numpy_iterator())
!pwd
from numba import cuda
device = cuda.get_current_device()
device.reset()
del cuda
