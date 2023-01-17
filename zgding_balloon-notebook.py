#文件路径



import os

print("相关数据：", os.listdir("../input/"))

MODEL_PATH = "../input/pretrainmodel/mask_rcnn_balloon.h5"

print("预训练参数文件：", os.listdir("../input/pretrainmodel/"))

MRCNN_LIB_PATH = "../input/mrcnn-lib2/mrcnn/"

print("mrcnn的库文件：", os.listdir(MRCNN_LIB_PATH + "/mrcnn/"))

TRAIN_DATA = "../input/train-data-b/balloon_dataset/balloon/"

print("训练和验证数据：", os.listdir(TRAIN_DATA))

TEST_DATA = "../input/test-data2/test_data/test_data/"

print("测试数据：",os.listdir(TEST_DATA))

LOGS = "./"

print("输出文件夹：",os.listdir(LOGS))

cmd = "rm -fr " + LOGS + "/.*" #注意rm操作：清除磁盘空间

os.system(cmd)

print("清空输出文件夹：", os.listdir(LOGS))
#加载mask_rcnn源码，以及对气球分割模型的定制接口



import sys

import json

import datetime

import numpy as np

import skimage.draw



# Root directory of the project

ROOT_DIR = os.path.abspath(MRCNN_LIB_PATH)



# Import Mask RCNN

sys.path.append(ROOT_DIR)  # To find local version of the library

from mrcnn.config import Config

from mrcnn import model as modellib, utils



# Path to trained weights file

COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")



# Directory to save logs and model checkpoints, if not provided

# through the command line argument --logs

DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "./")



############################################################

#  Configurations

############################################################





class BalloonConfig(Config):

    """Configuration for training on the toy  dataset.

    Derives from the base Config class and overrides some values.

    """

    # Give the configuration a recognizable name

    NAME = "balloon"



    # We use a GPU with 12GB memory, which can fit two images.

    # Adjust down if you use a smaller GPU.

    IMAGES_PER_GPU = 2



    # Number of classes (including background)

    NUM_CLASSES = 1 + 1  # Background + balloon



    # Number of training steps per epoch

    STEPS_PER_EPOCH = 2#1000



    # Skip detections with < 90% confidence

    DETECTION_MIN_CONFIDENCE = 0.9

    

############################################################

#  Dataset

############################################################



class BalloonDataset(utils.Dataset):



    def load_balloon(self, dataset_dir, subset):

        """Load a subset of the Balloon dataset.

        dataset_dir: Root directory of the dataset.

        subset: Subset to load: train or val

        """

        # Add classes. We have only one class to add.

        self.add_class("balloon", 1, "balloon")



        # Train or validation dataset?

        assert subset in ["train", "val"]

        dataset_dir = os.path.join(dataset_dir, subset)



        # Load annotations

        # VGG Image Annotator (up to version 1.6) saves each image in the form:

        # { 'filename': '28503151_5b5b7ec140_b.jpg',

        #   'regions': {

        #       '0': {

        #           'region_attributes': {},

        #           'shape_attributes': {

        #               'all_points_x': [...],

        #               'all_points_y': [...],

        #               'name': 'polygon'}},

        #       ... more regions ...

        #   },

        #   'size': 100202

        # }

        # We mostly care about the x and y coordinates of each region

        # Note: In VIA 2.0, regions was changed from a dict to a list.

        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))

        annotations = list(annotations.values())  # don't need the dict keys



        # The VIA tool saves images in the JSON even if they don't have any

        # annotations. Skip unannotated images.

        annotations = [a for a in annotations if a['regions']]



        # Add images

        for a in annotations:

            # Get the x, y coordinaets of points of the polygons that make up

            # the outline of each object instance. These are stores in the

            # shape_attributes (see json format above)

            # The if condition is needed to support VIA versions 1.x and 2.x.

            if type(a['regions']) is dict:

                polygons = [r['shape_attributes'] for r in a['regions'].values()]

            else:

                polygons = [r['shape_attributes'] for r in a['regions']] 



            # load_mask() needs the image size to convert polygons to masks.

            # Unfortunately, VIA doesn't include it in JSON, so we must read

            # the image. This is only managable since the dataset is tiny.

            image_path = os.path.join(dataset_dir, a['filename'])

            image = skimage.io.imread(image_path)

            height, width = image.shape[:2]



            self.add_image(

                "balloon",

                image_id=a['filename'],  # use file name as a unique image id

                path=image_path,

                width=width, height=height,

                polygons=polygons)



    def load_mask(self, image_id):

        """Generate instance masks for an image.

       Returns:

        masks: A bool array of shape [height, width, instance count] with

            one mask per instance.

        class_ids: a 1D array of class IDs of the instance masks.

        """

        # If not a balloon dataset image, delegate to parent class.

        image_info = self.image_info[image_id]

        if image_info["source"] != "balloon":

            return super(self.__class__, self).load_mask(image_id)



        # Convert polygons to a bitmap mask of shape

        # [height, width, instance_count]

        info = self.image_info[image_id]

        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],

                        dtype=np.uint8)

        for i, p in enumerate(info["polygons"]):

            # Get indexes of pixels inside the polygon and set them to 1

            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])

            mask[rr, cc, i] = 1



        # Return mask, and array of class IDs of each instance. Since we have

        # one class ID only, we return an array of 1s

        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)



    def image_reference(self, image_id):

        """Return the path of the image."""

        info = self.image_info[image_id]

        if info["source"] == "balloon":

            return info["path"]

        else:

            super(self.__class__, self).image_reference(image_id)





def train(model):

    """Train the model."""

    # Training dataset.

    dataset_train = BalloonDataset()

    dataset_train.load_balloon(args_dataset, "train")

    dataset_train.prepare()



    # Validation dataset

    dataset_val = BalloonDataset()

    dataset_val.load_balloon(args_dataset, "val")

    dataset_val.prepare()



    # *** This training schedule is an example. Update to your needs ***

    # Since we're using a very small dataset, and starting from

    # COCO trained weights, we don't need to train too long. Also,

    # no need to train all layers, just the heads should do it.

    print("Training network heads")

    model.train(dataset_train, dataset_val,

                learning_rate=config.LEARNING_RATE,

                epochs=2,

                layers='heads')





def color_splash(image, mask):

    """Apply color splash effect.

    image: RGB image [height, width, 3]

    mask: instance segmentation mask [height, width, instance count]



    Returns result image.

    """

    # Make a grayscale copy of the image. The grayscale copy still

    # has 3 RGB channels, though.

    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255

    # Copy color pixels from the original color image where mask is set

    if mask.shape[-1] > 0:

        # We're treating all instances as one, so collapse the mask into one layer

        mask = (np.sum(mask, -1, keepdims=True) >= 1)

        splash = np.where(mask, image, gray).astype(np.uint8)

    else:

        splash = gray.astype(np.uint8)

    return splash





def detect_and_color_splash(model, image_path=None, video_path=None):

    assert image_path or video_path





    # Run model detection and generate the color splash effect

    print("Running on {}".format(args_image))

    # Read image

    image = skimage.io.imread(args_image + os.listdir(args_image)[0])

    # Detect objects

    r = model.detect([image], verbose=1)[0]

    # Color splash

    splash = color_splash(image, r['masks'])

    # Save output

    #file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())

    #skimage.io.imsave(file_name, splash)

    return image, splash



#查看训练集合



import matplotlib.pyplot as plt



args_dataset = TRAIN_DATA

args_weights = MODEL_PATH

args_logs = LOGS

args_image = TEST_DATA

args_video = None



# Validate arguments

print("Weights: ", args_weights)

print("Dataset: ", args_dataset)

print("Logs: ", args_logs)





# Training dataset.

dataset_train = BalloonDataset()

dataset_train.load_balloon(args_dataset, "train")

dataset_train.prepare()



image_ids = dataset_train.image_ids

for cnt, image_id in enumerate(image_ids):

    if cnt < 3:

        continue

    image = dataset_train.load_image(image_id)

    mask, class_ids = dataset_train.load_mask(image_id)

    idx = 0

    print ("image.shape:",image.shape)

    print ("mask.shape:",mask.shape)

    print ("cls_list:",class_ids)

    print ("mask:")

    

    

    plt.figure(num='astronaut',figsize=(16,16))

    plt.subplot(6,6,1)

    plt.imshow(image[:,:,:])

    

    for ii in range(mask.shape[2]):

        plt.subplot(6,6,ii+2)

        plt.title('mask_{}'.format(ii))

        plt.imshow(mask[:,:,ii])

    break

    idx += 1

    







#inference of pretrained model



# Configurations

class InferenceConfig(BalloonConfig):

    # Set batch size to 1 since we'll be running inference on

    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU

    GPU_COUNT = 1

    IMAGES_PER_GPU = 1

config = InferenceConfig()

config.display()



# Create model

model = modellib.MaskRCNN(mode="inference", config=config,

                              model_dir=args_logs)



# Load your weights

print("Loading your weights ", args_weights)

model.load_weights(args_weights, by_name=True)



#inference

image_src, image_splash = detect_and_color_splash(model, image_path=args_image, video_path=args_video)

#show

plt.figure(num='astronaut2',figsize=(16,16))

plt.subplot(6,6,1)

plt.imshow(image_src[:,:,:])

plt.subplot(6,6,2)

plt.imshow(image_splash[:,:,:])
# train



config = BalloonConfig()

config.display()



# Create model

model = modellib.MaskRCNN(mode="training", config=config, model_dir=args_logs)



# Load weights

print("Loading weights ", args_weights)

model.load_weights(args_weights, by_name=True)



# Train or evaluate

train(model)
#查看训练结果

LOGS = "./"

print(os.listdir(LOGS + "balloon20190509T0633/"))

#inference of your model



# Configurations

class InferenceConfig(BalloonConfig):

    # Set batch size to 1 since we'll be running inference on

    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU

    GPU_COUNT = 1

    IMAGES_PER_GPU = 1

config = InferenceConfig()

config.display()



# Create model

model = modellib.MaskRCNN(mode="inference", config=config,

                              model_dir=args_logs)



# Load your weights

args_your_weights = LOGS + "balloon20190509T0633/mask_rcnn_balloon_0002.h5" #"path to your weight" #TODO

print("Loading your weights ", args_your_weights)

model.load_weights(args_weights, by_name=True)



#inference

image_src, image_splash = detect_and_color_splash(model, image_path=args_image, video_path=args_video)

#show

plt.figure(num='astronaut2',figsize=(16,16))

plt.subplot(6,6,1)

plt.imshow(image_src[:,:,:])

plt.subplot(6,6,2)

plt.imshow(image_splash[:,:,:])