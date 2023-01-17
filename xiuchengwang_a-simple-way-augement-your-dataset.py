import tensorflow as tf
import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import random
path_jpg = '../input/time.jpg'
image_raw_data = tf.gfile.FastGFile(path_jpg,'rb').read()
with tf.Session() as sess:

    

    

    img_data = tf.image.decode_jpeg(image_raw_data)

    image_float = tf.image.convert_image_dtype(img_data, tf.float32)

    resized = tf.image.resize_images(image_float, [200, 300], method=0)

    plt.imshow(resized.eval())

    plt.show()
def distort_color(image, color_ordering=0):

    lower=random.uniform(-32.0/255.0, 32.0/255.0)

    upper=random.uniform(0.4, 1.6)

    a=random.uniform(-0.3, 0.3)

    b=random.uniform(0.4, 1.6)

    if color_ordering == 0:

        image = tf.image.adjust_brightness(image, lower)

        image = tf.image.adjust_saturation(image, upper)

        image = tf.image.adjust_hue(image,a)

        image = tf.image.adjust_contrast(image, b)

    else:

        image = tf.image.adjust_saturation(image, upper)

        image = tf.image.adjust_brightness(image, lower)

        image = tf.image.adjust_contrast(image, b)

        image = tf.image.adjust_hue(image,a)



    return tf.clip_by_value(image, 0.0, 1.0)
def preprocess_for_train(image, height, width, bbox):

    if bbox is None:

        bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])

    if image.dtype != tf.float32:

        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(

        tf.shape(image), bounding_boxes=bbox, min_object_covered=0.75)

    distorted_image = tf.slice(image, bbox_begin, bbox_size)



    

    distorted_image = tf.image.resize_images(distorted_image, [height, width], method=np.random.randint(4))

    distorted_image = tf.image.random_flip_left_right(distorted_image)

    distorted_image = distort_color(distorted_image, np.random.randint(2))

    return distorted_image
image_raw_data = tf.gfile.FastGFile(path_jpg,'rb').read()

with tf.Session() as sess:

    img_data = tf.image.decode_jpeg(image_raw_data)

    boxes = tf.constant([[[0.05, 0.05, 0.9, 0.7], [0.15, 0.17, 0.85, 0.96]]])

    for i in range(9):

        result = preprocess_for_train(img_data, 200, 300,boxes)

        #print(result.eval())

        plt.imshow(result.eval())



        plt.show()

        image_float = tf.image.convert_image_dtype(result, tf.uint8)

        encoded_image = tf.image.encode_jpeg(image_float.eval())

        l=i

        l=str(l)

        with tf.gfile.GFile("cat"+l+".jpg", "wb") as f:

            f.write(encoded_image.eval())
