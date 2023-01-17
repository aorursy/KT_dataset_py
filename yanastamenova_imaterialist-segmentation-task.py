# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
#!git config --global http.proxy http://proxyuser:proxypwd@proxy.server.com:8080
# Import required libraries
import gc
import sys
import json
import random
from pathlib import Path
from PIL import Image
#import time

import cv2 # for image manipulation
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from imgaug import augmenters as iaa

import seaborn as sns
import matplotlib.image as mpimg
from matplotlib import pyplot as plt

import pickle
from tqdm import tqdm, tqdm_pandas
#from scipy.signal import argrelextrema

import itertools
import h5py #read .h5
import glob
#!pip install urllib3==1.7
!pip install tensorflow==1.5 #need that because Mask-RCNN version
!pip install keras==2.1.5

import tensorflow
print(tensorflow.__version__)
import keras
print(keras.__version__)
image = Image.open("/kaggle/input/imaterialist-fashion-2020-fgvc7/train/8617b2102bb25fbb0a93fb7f11e6397c.jpg")
image = np.array(image)
print(image.shape)
plt.show()
plt.imshow(image)
plt.axis('off')
dataDir = "/kaggle/input/imaterialist-fashion-2020-fgvc7/"
workDir = "/kaggle/working/"
os.listdir(dataDir)

#alternatively
#!ls /kaggle/input/imaterialist-fashion-2020-fgvc7/
sample_submission = pd.read_csv(os.path.join(dataDir, 'sample_submission.csv'))
sample_submission.sample(1)
train_data = pd.read_csv(os.path.join(dataDir, 'train.csv'))
train_data.sample(5)
train_data.info()
sample_submission.info()
sample_submission.head()
print('Train data shape: {0} \nUnique number of train images: {1}'
      .format(train_data.shape, train_data["ImageId"].nunique()))
print('Test data shape: {0} \nUnique number of test images: {1}'
      .format(sample_submission.shape, sample_submission["ImageId"].nunique()))
pd.DataFrame(train_data['Height'].describe()).T.drop(columns = ['std','25%', '50%', '75%'])
pd.DataFrame(train_data['Width'].describe()).T.drop(columns = ['std','25%', '50%', '75%'])
plt.figure(figsize = (100,10))
max_height = list(set(train_data[train_data['Height'] == train_data['Height'].max()]['ImageId']))[0]
image = mpimg.imread('{0}/train/{1}.jpg'.format(dataDir, max_height))                     
plt.imshow(image)
plt.axis('off')
plt.show()
#Extract information from the .json file
with open(os.path.join(dataDir, 'label_descriptions.json'), 'r') as file:
    label_description = json.load(file)
label_description
n_classes = len(label_description['categories'])
n_attributes = len(label_description['attributes'])
print('Classes: {0} \nAttributes: {1}'.
     format(str(n_classes), str(n_attributes)))
categories_data = pd.DataFrame(label_description['categories'])
attributes_data = pd.DataFrame(label_description['attributes'])
categories_data
attributes_data
categories_data.supercategory.unique()
attributes_data.supercategory.unique()
def show_images(size = 4, figsize = (12, 12)):
    #get the images
    image_ids = train_data['ImageId'].unique()[:size]
    images = []
    
    for image_id in image_ids:
        images.append(mpimg.imread('{0}/train/{1}.jpg'.format(dataDir, image_id)))
        
    count = 0
    
    fig, ax = plt.subplots(nrows = 2, ncols = 2, figsize = figsize)
    for row in ax:
        for col in row:
            col.imshow(images[count])
            col.axis('off')
            count += 1
    plt.show()
    gc.collect()
show_images()
#Function to create mask
def create_mask(size):
    image_ids = train_data['ImageId'].unique()[:size] #get a number of images
    images_meta = [] #to be added in this array
    
    for image_id in image_ids:
        img = mpimg.imread('{0}/train/{1}.jpg'.format(dataDir, image_id))
        images_meta.append({
            'image': img,
            'shape': img.shape,
            'encoded_pixels': train_data[train_data['ImageId'] == image_id]['EncodedPixels'],
            'class_ids': train_data[train_data['ImageId'] == image_id]['ClassId']
        })
        
    masks = []
    
    for image in images_meta:
        shape = image.get('shape') #get via key
        encoded_pixels = list(image.get('encoded_pixels')) 
        class_ids = list(image.get('class_ids'))
        
        #Initialize numpy array with shape same as image size
        height, width = shape[:2] 
        mask = np.zeros((height, width)).reshape(-1) 
        # (-1) 'The new shape should be compatible with the original shape'
        # numpy allow us to give one of new shape parameter as -1 but not (-1, -1)).
        # It means that it is an unknown dimension and we want numpy to figure it out.
        # And numpy will figure this by looking at the 'length of the array and remaining
        # dimensions' and making sure it satisfies the above mentioned criteria
        
        #Iterate over encoded pixels and create mask
        for segment, (pixel_str, class_id) in enumerate(zip(encoded_pixels, class_ids)):
            splitted_pixels = list(map(int, pixel_str.split()))      #split the pixels string
            pixel_starts = splitted_pixels[::2]                      #choose every second element
            run_lengths = splitted_pixels[1::2]                      #start from 1 with step size 2
            assert max(pixel_starts) < mask.shape[0]                 #make sure it is ok
            
            for pixel_start, run_length in zip(pixel_starts, run_lengths):
                pixel_start = int(pixel_start) - 1
                run_length = int(run_length)
                mask[pixel_start:pixel_start+run_length] = 255 - class_id 
        masks.append(mask.reshape((height, width), order = 'F'))
    
    return masks, images_meta

def plot_segmented_images(size = 4, figsize = (14, 14)):
    #First, create masks from given segments
    masks, images_meta = create_mask(size)
    
    #Plot images
    
    count = 0
    
    fig, ax = plt.subplots(nrows = 2, ncols = 2, figsize = figsize)
    for row in ax:
        for col in row:
            col.imshow(images_meta[count]['image'])
            col.imshow(masks[count], alpha = 0.50)
            col.axis('off')
            count += 1
    plt.show()
    gc.collect()
plot_segmented_images()
images_data = train_data.groupby('ImageId')['EncodedPixels', 'ClassId'].agg(lambda x: list(x))
dimensions_data = train_data.groupby('ImageId')['Height', 'Width'].mean()
images_data = images_data.join(dimensions_data, on='ImageId')

images_data.head()
print("Total images: ", len(images_data))
!git clone https://www.github.com/matterport/Mask_RCNN.git

os.chdir('Mask_RCNN')

!rm -rf .git # to prevent an error when the kernel is committed
!rm -rf images assets # to prevent displaying images at the bottom of a kernel
!wget --quiet https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5
!ls -lh mask_rcnn_coco.h5

COCO_WEIGHTS_PATH = 'mask_rcnn_coco.h5'
# sys.path.append(workDir/'Mask_RCNN') # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
# Directory to save logs and trained model
modelDir = os.path.join(workDir, "logs")
class FashionImagesConfig(Config):
    """Configuration for training on the iMaterialist dataset.
    Derives from the base Config class and overrides values specific
    to the dataset.
    """
    # Give the configuration a recognizable name
    NAME = "fashion2020"

    # Train on 1 GPU and 3 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 3

    # Number of classes (including background)
    NUM_CLASSES = 1 + len(categories_data)  # background + 46 shapes
    BACKBONE = 'resnet50'

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. 
    TRAIN_ROIS_PER_IMAGE = 16

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5
    
config = FashionImagesConfig()
config.display()
class Fashion2020Dataset(utils.Dataset):
    def __init__(self, data):
        super().__init__(self)
        
        self.IMAGE_SIZE = 256
        self.DIMENSIONS = (256, 256)
        
        for category in label_description['categories']:
            self.add_class('fashion2020', category.get('id'), category.get('name'))
            
        for i, row in data.iterrows():
            self.add_image('fashion2020',
                          image_id = row.name,
                          path = str('{0}/train/{1}.jpg'.format(dataDir, row.name)),
                          labels = row['ClassId'],
                          annotations = row['EncodedPixels'],
                          height = row['Height'],
                          width = row['Width'])
            
    def modify_image(self, image_path):
        #dims = (self.IMAGE_SIZE, self.IMAGE_SIZE)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.DIMENSIONS, interpolation = cv2.INTER_AREA)
        
        return img
    
    def load_image(self, image_id):
        img = self.image_info[image_id]['path']
        return self.modify_image(img)
    
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path'], [x for x in info['labels']]
    
    def load_mask(self, image_id):
        info = self.image_info[image_id]
        mask = np.zeros((self.IMAGE_SIZE, self.IMAGE_SIZE, len(info['annotations'])), dtype = np.uint8)
        
        labels = []
        
        for (m, (annotation, label)) in enumerate(zip(info['annotations'], info['labels'])):
            sub_mask = np.full(info['height']*info['width'], 0, dtype=np.uint8)
            annotation = [int(x) for x in annotation.split(' ')]
            
            for i, start_pixel in enumerate(annotation[::2]):
                sub_mask[start_pixel: start_pixel + annotation[2*i+1]] = 1
                
            sub_mask = sub_mask.reshape((info['height'], info['width']), order = 'F')
            sub_mask = cv2.resize(sub_mask, self.DIMENSIONS, interpolation = cv2.INTER_NEAREST)
            
            mask[:, :, m] = sub_mask
            labels.append(int(label)+1)
        return mask, np.array(labels)
dataset = Fashion2020Dataset(images_data)
dataset.prepare()
#Show masks separately
for i in range(10):
    image_id = random.choice(dataset.image_ids)
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, dataset.class_names, limit = 5)
images_data.info()
# Load random image and mask.
image_id = random.choice(dataset.image_ids)
image = dataset.load_image(image_id)
mask, class_ids = dataset.load_mask(image_id)
# Compute Bounding box
bbox = utils.extract_bboxes(mask)

# Display image and additional stats
print("image_id ", image_id, dataset.image_reference(image_id))
log("image", image)
log("mask", mask)
log("class_ids", class_ids)
log("bbox", bbox)
# Display image and instances
visualize.display_instances(image, bbox, mask, class_ids, dataset.class_names)
#Split to training and validation data
from sklearn.utils import shuffle

random.seed(42)
images_data_shuffled = shuffle(images_data)
val_size = int(0.05 * len(images_data_shuffled['ClassId']))
image_data_val = images_data_shuffled[:val_size]
image_data_train = images_data_shuffled[val_size:]

print(len(image_data_train), len(image_data_val))
print(f'Training set: {image_data_train.shape} \nValidation set: {image_data_val.shape}')
# prepare the training dataset
dataset_train = Fashion2020Dataset(image_data_train)
dataset_train.prepare()
dataset_val = Fashion2020Dataset(image_data_val)
dataset_val.prepare()
class_ids = [0]
while class_ids[0] == 0:  ## look for a mask
    image_id = random.choice(dataset_train.image_ids)
    image_fp = dataset_train.image_reference(image_id)
    image = dataset_train.load_image(image_id)
    mask, class_ids = dataset_train.load_mask(image_id)

print(image.shape)

plt.figure(figsize=(12, 12))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.axis('off')

plt.subplot(1, 2, 2)
masked = np.zeros(image.shape[:2])
for i in range(mask.shape[2]):
    masked += image[:, :, 0] * mask[:, :, i]
plt.imshow(masked, cmap = 'gray', alpha = 0.75)
plt.axis('off')

print(image_fp)
print(class_ids)
#Apply some image augmentation, incl. flipping, rotation, blurring, etc.
augmentation = iaa.Sequential([
    iaa.OneOf([ ## geometric transform
        iaa.Affine(
            scale={"x": (0.98, 1.02), "y": (0.98, 1.04)},
            translate_percent={"x": (-0.02, 0.02), "y": (-0.06, 0.06)},
            rotate=(-3, 3),
        ),
        iaa.Fliplr(0.2)
    ]),
    iaa.OneOf([ ## brightness or contrast or blur
        iaa.Multiply((0.9, 1.1)),
        iaa.ContrastNormalization((0.7, 1.1)),
        iaa.GaussianBlur(sigma=(0.0, 0.2)),
    ]),
])

# test augmentation on image
imggrid = augmentation.draw_grid(image[:, :, 0], cols=5, rows=2)
plt.figure(figsize=(30, 12))
plt.axis('off')
_ = plt.imshow(imggrid[:, :, 0], cmap='gray')
model = modellib.MaskRCNN(mode = 'training', config = config, model_dir = workDir)
model.load_weights(COCO_WEIGHTS_PATH, by_name=True, exclude=[
    "mrcnn_class_logits", "mrcnn_bbox_fc",
    "mrcnn_bbox", "mrcnn_mask"])
#copy this class, only to uncomment the tensorboard callback in the train function
class MaskRCNN():
    def train(self, train_dataset, val_dataset, learning_rate, epochs, layers,
              augmentation=None, custom_callbacks=None, no_augmentation_sources=None):
        """Train the model.
        train_dataset, val_dataset: Training and validation Dataset objects.
        learning_rate: The learning rate to train with
        epochs: Number of training epochs. Note that previous training epochs
                are considered to be done alreay, so this actually determines
                the epochs to train in total rather than in this particaular
                call.
        layers: Allows selecting wich layers to train. It can be:
            - A regular expression to match layer names to train
            - One of these predefined values:
              heads: The RPN, classifier and mask heads of the network
              all: All the layers
              3+: Train Resnet stage 3 and up
              4+: Train Resnet stage 4 and up
              5+: Train Resnet stage 5 and up
        augmentation: Optional. An imgaug (https://github.com/aleju/imgaug)
            augmentation. For example, passing imgaug.augmenters.Fliplr(0.5)
            flips images right/left 50% of the time. You can pass complex
            augmentations as well. This augmentation applies 50% of the
            time, and when it does it flips images right/left half the time
            and adds a Gaussian blur with a random sigma in range 0 to 5.

                augmentation = imgaug.augmenters.Sometimes(0.5, [
                    imgaug.augmenters.Fliplr(0.5),
                    imgaug.augmenters.GaussianBlur(sigma=(0.0, 5.0))
                ])
	    custom_callbacks: Optional. Add custom callbacks to be called
	        with the keras fit_generator method. Must be list of type keras.callbacks.
        no_augmentation_sources: Optional. List of sources to exclude for
            augmentation. A source is string that identifies a dataset and is
            defined in the Dataset class.
        """
        assert self.mode == "training", "Create model in training mode."

        # Pre-defined layer regular expressions
        layer_regex = {
            # all layers but the backbone
            "heads": r"(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # From a specific Resnet stage and up
            "3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "5+": r"(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # All layers
            "all": ".*",
        }
        if layers in layer_regex.keys():
            layers = layer_regex[layers]

        # Data generators
        train_generator = data_generator(train_dataset, self.config, shuffle=True,
                                         augmentation=augmentation,
                                         batch_size=self.config.BATCH_SIZE,
                                         no_augmentation_sources=no_augmentation_sources)
        val_generator = data_generator(val_dataset, self.config, shuffle=True,
                                       batch_size=self.config.BATCH_SIZE)

        # Create log_dir if it does not exist
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Callbacks
        callbacks = [
            keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                        histogram_freq=0, write_graph=True, write_images=False),
            keras.callbacks.ModelCheckpoint(self.checkpoint_path,
                                            verbose=1, save_weights_only=True),
        ]

        # Add custom callbacks to the list
        if custom_callbacks:
            callbacks += custom_callbacks

        # Train
        log("\nStarting at epoch {}. LR={}\n".format(self.epoch, learning_rate))
        log("Checkpoint Path: {}".format(self.checkpoint_path))
        self.set_trainable(layers)
        self.compile(learning_rate, self.config.LEARNING_MOMENTUM)

        # Work-around for Windows: Keras fails on Windows when using
        # multiprocessing workers. See discussion here:
        # https://github.com/matterport/Mask_RCNN/issues/13#issuecomment-353124009
        if os.name is 'nt':
            workers = 0
        else:
            workers = multiprocessing.cpu_count()
        print(workers)
        self.keras_model.fit_generator(
            train_generator,
            initial_epoch=self.epoch,
            epochs=epochs,
            steps_per_epoch=self.config.STEPS_PER_EPOCH,
            callbacks=callbacks,
            validation_data=val_generator,
            validation_steps=self.config.VALIDATION_STEPS,
            max_queue_size=100,
            workers=1,
            use_multiprocessing=False,
        )
        self.epoch = max(self.epoch, epochs)
# These may not be optimal parameters just to run it!
LEARNING_RATE = 0.005
LEARNING_RATE_TUNE = 0.0001 #for the last 3 epochs
EPOCHS = [1, 3, 5, 8] 

import warnings 
warnings.filterwarnings("ignore")
# train heads with higher lr for more learning speed
"""model.train(dataset_train, dataset_val,
            learning_rate = LEARNING_RATE,
            epochs = EPOCHS[0],
            layers = 'heads',
            augmentation = None)  """
#history = model.keras_model.history.history
"""%cd ..
model.keras_model.save_weights('modelHead.h5')
pickle.dump(history, open('modelHead.pkl', 'ab'))
%cd $workDir"""
#model.load_weights('/kaggle/input/head-saved-weights/modelHead.h5')
#now with all layers and augmentation included 2 more epochs
"""model.train(dataset_train, dataset_val,
            learning_rate = LEARNING_RATE,
            epochs = EPOCHS[1],
            layers = 'all',
            augmentation = augmentation)"""
#load history
#history = pickle.load(open('/kaggle/input/head-saved-weights/modelHead.pkl', 'rb'))
"""new_history = model.keras_model.history.history
for k in new_history: history[k] = history[k] + new_history[k]"""
"""%cd ..
model.keras_model.save_weights('modelAll1.h5')
pickle.dump(history, open('modelAll1.pkl', 'ab'))
%cd $workDir"""
#decrease learning rate and train for 2 more epochs
"""model.train(dataset_train, dataset_val,
            learning_rate = LEARNING_RATE/5,
            epochs = EPOCHS[2],
            layers = 'all',
            augmentation = augmentation)"""
"""new_history = model.keras_model.history.history
for k in new_history: history[k] = history[k] + new_history[k]"""
"""%cd ..
model.keras_model.save_weights('modelAll2.h5')
pickle.dump(history, open('modelAll2.pkl', 'ab'))
%cd $workDir"""
#the last three epochs train with LR = 1e-4
"""model.train(dataset_train, dataset_val,
            learning_rate = LEARNING_RATE_TUNE,
            epochs = EPOCHS[3],
            layers = 'all',
            augmentation = augmentation)"""
"""new_history = model.keras_model.history.history
for k in new_history: history[k] = history[k] + new_history[k]"""
"""model.keras_model.save_weights('modelAll3.h5')
pickle.dump(history, open('modelAll3.pkl', 'wb'))"""
history = pickle.load(open('/kaggle/input/head-saved-weights/modelAll3.pkl', 'rb'))
epochs = range(1, len(next(iter(history.values())))+1) #get number of epochs
history_data = pd.DataFrame(history, index=epochs)
"""%cd ..
history_data.to_csv('History data from Mask_RCNN training' + '.csv')
%cd $workDir"""
history_data
plt.figure(figsize=(40,8))

plt.subplot(141)
plt.plot(epochs, history_data["loss"], label="Train loss")
plt.plot(epochs, history_data["val_loss"], label="Valid loss")
plt.legend()

plt.subplot(142)
plt.plot(epochs, history_data["mrcnn_class_loss"], label="Train class loss")
plt.plot(epochs, history_data["val_mrcnn_class_loss"], label="Valid class loss")
plt.legend()

plt.show()

plt.figure(figsize=(40,8))

plt.subplot(141)
plt.plot(epochs, history_data["mrcnn_bbox_loss"], label="Train box loss")
plt.plot(epochs, history_data["val_mrcnn_bbox_loss"], label="Valid box loss")
plt.legend()

plt.subplot(142)
plt.plot(epochs, history_data['mrcnn_mask_loss'], label="Train mask loss")
plt.plot(epochs, history_data['val_mrcnn_mask_loss'], label="Valid mask loss")
plt.legend()

plt.show()
best_epoch = np.argmin(history['val_loss'])
print("Best epoch: ", best_epoch+1, history['val_loss'][best_epoch])
#select trained model
glob_list = glob.glob(f'/kaggle/input/head-saved-weights/mask_rcnn_fashion2020_{(best_epoch+1):04d}.h5')
model_path = glob_list[0] if glob_list else ''
model_path
class InferenceConfig(FashionImagesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

#Recreate the model in inference mode
model = modellib.MaskRCNN(mode = 'inference',
                         config = inference_config,
                         model_dir = workDir)

#Load trained weights 
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name = True)
sample_data = sample_submission
sample_data.head()
# Convert data to run-length encoding
def to_rle(bits):
    rle = []
    pos = 0
    for bit, group in itertools.groupby(bits):
        group_list = list(group)
        if bit:
            rle.extend([pos, sum(group_list)])
        pos += len(group_list)
    return rle
# Fix overlapped masks
def fix_masks(masks, rois):
    areas = np.sum(masks.reshape(-1, masks.shape[-1]), axis = 0)
    mask_index = np.argsort(areas)
    union_mask = np.zeros(masks.shape[:-1], dtype = bool)
    for m in mask_index:
        masks[:, :, m] = np.logical_and(masks[:, :, m], np.logical_not(union_mask))
        union_mask = np.logical_or(masks[:, :, m], union_mask)
    for m in range(masks.shape[-1]):
        mask_position = np.where(masks[:, :, m] == True)
        if np.any(mask_position):
            y1, x1 = np.min(mask_position, axis = 1)
            y2, x2 = np.max(mask_position, axis = 1)
            rois[m, :] = [y1, x1, y2, x2]
    return masks, rois
IMAGE_SIZE = 256
def resize_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation = cv2.INTER_AREA)  
    return img
"""%%time
submission_list = []
missing_count = 0

for i, row in tqdm(sample_data.iterrows(), total = len(sample_data)):
    image = resize_image(str(dataDir + '/test/' + row['ImageId']) + '.jpg')
    result = model.detect([image])[0]
    if result['masks'].size > 0:
        masks, _ = fix_masks(result['masks'], result['rois'])
        for m in range(masks.shape[-1]):
            mask = masks[:, :, m].ravel(order = 'F')
            rle = to_rle(mask)
            label = result['class_ids'][m] - 1
            submission_list.append([row['ImageId'], ' '.join(list(map(str, rle))), label, np.NaN])
    else:
        # The system does not allow missing ids
        submission_list.append([row['ImageId'], '1 1', 23, np.NaN])
        missing_count += 1
    """
"""validation_pred_df = pd.DataFrame(submission_list)
validation_pred_df.columns = ['ImageId', 'EncodedPixels', 'ClassId']
validation_pred_df = validation_pred_df.groupby('ImageId')['EncodedPixels', 'ClassId'].agg(lambda x: list(x))

ImageId = pd.Series(validation_pred_df.index)
validation_pred_df.index = pd.Index(list(range(len(validation_pred_df))))
validation_pred_df['ImageId'] = ImageId

validation_pred_df"""
#sample_submission.columns
"""submission_data = pd.DataFrame(submission_list, columns=sample_submission.columns.values)
print("Total image results: ", submission_data['ImageId'].nunique())
print("Missing Images: ", missing_count)
submission_data.head()"""
#submission_data.to_csv('submission.csv', index=False)
submission_raw = pd.read_csv("/kaggle/input/submissionraw/submission_non_grouped.csv")
submission_raw.head()
submission = submission_raw.groupby('ImageId')['EncodedPixels', 'ClassId', 'AttributesIds'].agg(lambda x: list(x))
submission.sample(5)
submission.info()
submission.to_csv('submission.csv', index=False)