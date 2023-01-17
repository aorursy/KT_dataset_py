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
import tensorflow as tf

from keras import backend as K

from keras.models import load_model, Model

import PIL

import scipy.io

import scipy.misc

import matplotlib.pyplot as plt

import colorsys

import imghdr

import os

import random

from keras import backend as K



import numpy as np

from PIL import Image, ImageDraw, ImageFont



def read_classes(classes_path):

    with open(classes_path) as f:

        class_names = f.readlines()

    class_names = [c.strip() for c in class_names]

    return class_names



def read_anchors(anchors_path):

    with open(anchors_path) as f:

        anchors = f.readline()

        anchors = [float(x) for x in anchors.split(',')]

        anchors = np.array(anchors).reshape(-1, 2)

    return anchors



def generate_colors(class_names):

    hsv_tuples = [(x / len(class_names), 1., 1.) for x in range(len(class_names))]

    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))

    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(10101)  # Fixed seed for consistent colors across runs.

    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.

    random.seed(None)  # Reset seed to default.

    return colors



def scale_boxes(boxes, image_shape):

    """ Scales the predicted boxes in order to be drawable on the image"""

    height = image_shape[0]

    width = image_shape[1]

    image_dims = K.stack([height, width, height, width])

    image_dims = K.reshape(image_dims, [1, 4])

    boxes = boxes * image_dims

    return boxes



def preprocess_image(img_path, model_image_size):

    image_type = imghdr.what(img_path)

    image = Image.open(img_path)

    resized_image = image.resize(tuple(reversed(model_image_size)), Image.BICUBIC)

    image_data = np.array(resized_image, dtype='float32')

    image_data /= 255.

    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

    return image, image_data



def draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors):

    

    font = ImageFont.truetype(font='../input/fonttt/FiraMono-Medium.otf',size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))

    thickness = (image.size[0] + image.size[1]) // 300



    for i, c in reversed(list(enumerate(out_classes))):

        predicted_class = class_names[c]

        box = out_boxes[i]

        score = out_scores[i]



        label = '{} {:.2f}'.format(predicted_class, score)



        draw = ImageDraw.Draw(image)

        label_size = draw.textsize(label, font)



        top, left, bottom, right = box

        top = max(0, np.floor(top + 0.5).astype('int32'))

        left = max(0, np.floor(left + 0.5).astype('int32'))

        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))

        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

        print(label, (left, top), (right, bottom))



        if top - label_size[1] >= 0:

            text_origin = np.array([left, top - label_size[1]])

        else:

            text_origin = np.array([left, top + 1])



        # My kingdom for a good redistributable image drawing library.

        for i in range(thickness):

            draw.rectangle([left + i, top + i, right - i, bottom - i], outline=colors[c])

        draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=colors[c])

        draw.text(text_origin, label, fill=(0, 0, 0), font=font)

        del draw

def yolo_head(feats, anchors, num_classes):

    """Convert final layer features to bounding box parameters.



    Parameters

    ----------

    feats : tensor

        Final convolutional layer features.

    anchors : array-like

        Anchor box widths and heights.

    num_classes : int

        Number of target classes.



    Returns

    -------

    box_xy : tensor

        x, y box predictions adjusted by spatial location in conv layer.

    box_wh : tensor

        w, h box predictions adjusted by anchors and conv spatial resolution.

    box_conf : tensor

        Probability estimate for whether each box contains any object.

    box_class_pred : tensor

        Probability distribution estimate for each box over class labels.

    """

    num_anchors = len(anchors)

    # Reshape to batch, height, width, num_anchors, box_params.

    anchors_tensor = K.reshape(K.variable(anchors), [1, 1, 1, num_anchors, 2])

    # Static implementation for fixed models.

    # TODO: Remove or add option for static implementation.

    # _, conv_height, conv_width, _ = K.int_shape(feats)

    # conv_dims = K.variable([conv_width, conv_height])



    # Dynamic implementation of conv dims for fully convolutional model.

    conv_dims = K.shape(feats)[1:3]  # assuming channels last

    # In YOLO the height index is the inner most iteration.

    conv_height_index = K.arange(0, stop=conv_dims[0])

    conv_width_index = K.arange(0, stop=conv_dims[1])

    conv_height_index = K.tile(conv_height_index, [conv_dims[1]])



    # TODO: Repeat_elements and tf.split doesn't support dynamic splits.

    # conv_width_index = K.repeat_elements(conv_width_index, conv_dims[1], axis=0)

    conv_width_index = K.tile(K.expand_dims(conv_width_index, 0), [conv_dims[0], 1])

    conv_width_index = K.flatten(K.transpose(conv_width_index))

    conv_index = K.transpose(K.stack([conv_height_index, conv_width_index]))

    conv_index = K.reshape(conv_index, [1, conv_dims[0], conv_dims[1], 1, 2])

    conv_index = K.cast(conv_index, K.dtype(feats))

    

    feats = K.reshape(feats, [-1, conv_dims[0], conv_dims[1], num_anchors, num_classes + 5])

    conv_dims = K.cast(K.reshape(conv_dims, [1, 1, 1, 1, 2]), K.dtype(feats))



    # Static generation of conv_index:

    # conv_index = np.array([_ for _ in np.ndindex(conv_width, conv_height)])

    # conv_index = conv_index[:, [1, 0]]  # swap columns for YOLO ordering.

    # conv_index = K.variable(

    #     conv_index.reshape(1, conv_height, conv_width, 1, 2))

    # feats = Reshape(

    #     (conv_dims[0], conv_dims[1], num_anchors, num_classes + 5))(feats)



    box_confidence = K.sigmoid(feats[..., 4:5])

    box_xy = K.sigmoid(feats[..., :2])

    box_wh = K.exp(feats[..., 2:4])

    box_class_probs = K.softmax(feats[..., 5:])



    # Adjust preditions to each spatial grid point and anchor size.

    # Note: YOLO iterates over height index before width index.

    box_xy = (box_xy + conv_index) / conv_dims

    box_wh = box_wh * anchors_tensor / conv_dims



    return box_confidence, box_xy, box_wh, box_class_probs





def yolo_boxes_to_corners(box_xy, box_wh):

    """Convert YOLO box predictions to bounding box corners."""

    box_mins = box_xy - (box_wh / 2.)

    box_maxes = box_xy + (box_wh / 2.)



    return K.concatenate([

        box_mins[..., 1:2],  # y_min

        box_mins[..., 0:1],  # x_min

        box_maxes[..., 1:2],  # y_max

        box_maxes[..., 0:1]  # x_max

    ])

        
yolo_model = load_model("../input/yolo-h5-file/yolo.h5")
yolo_model.summary()
# GRADED FUNCTION: yolo_filter_boxes



def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = .6):

   

    box_scores = box_confidence*box_class_probs

    box_classes = K.argmax(box_scores, axis=-1)

    box_class_scores = K.max(box_scores, axis=-1)

    filtering_mask = box_class_scores>threshold



    scores = tf.boolean_mask(box_class_scores,filtering_mask,name="scores")

    boxes = tf.boolean_mask(boxes, filtering_mask, name='boxes')

    classes = tf.boolean_mask(box_classes,filtering_mask, name="boxes")

    return scores, boxes, classes
# GRADED FUNCTION: yolo_non_max_suppression



def yolo_non_max_suppression(scores, boxes, classes, max_boxes = 10, iou_threshold = 0.5):



    max_boxes_tensor = K.variable(max_boxes, dtype='int32')     # tensor to be used in tf.image.non_max_suppression()

    K.get_session().run(tf.variables_initializer([max_boxes_tensor])) # initialize variable max_boxes_tensor

    nms_indices = tf.image.non_max_suppression(boxes,scores,max_boxes,iou_threshold)



    scores = K.gather(scores, nms_indices)

    boxes = K.gather(boxes,nms_indices)

    classes = K.gather(classes,nms_indices)

    return scores, boxes, classes
def yolo_eval(yolo_outputs, image_shape = (720., 1280.), max_boxes=10, score_threshold=.6, iou_threshold=.5):

     

    box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs

    boxes = yolo_boxes_to_corners(box_xy, box_wh)

    scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = 0.5)

    boxes = scale_boxes(boxes, image_shape)

    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes)

    



    

    return scores, boxes, classes
sess = K.get_session()
class_names = read_classes("../input/model-data/coco_classes.txt")

anchors = read_anchors("../input//model-data/yolo_anchors.txt")

image_shape = (720., 1280.)  
yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)
def predict(sess, image_file):



    image, image_data = preprocess_image("../input/imagefiles/" + image_file, model_image_size = (608, 608))

    out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes], feed_dict={yolo_model.input: image_data, K.learning_phase(): 0})



    print('Found {} boxes for {}'.format(len(out_boxes), image_file))

    colors = generate_colors(class_names)



    draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)

    

    plt.imshow(image)

    

    return out_scores, out_boxes, out_classes
out_scores, out_boxes, out_classes = predict(sess, "test.jpg")