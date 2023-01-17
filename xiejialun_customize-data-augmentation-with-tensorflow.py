import os

import numpy as np

import tensorflow as tf

import matplotlib.pyplot as plt

from kaggle_datasets import KaggleDatasets



print('Tensorflow version : {}'.format(tf.__version__))
GCS_DS_PATH = KaggleDatasets().get_gcs_path('flower-classification-with-tpus')

GCS_PATH = os.path.join(GCS_DS_PATH, 'tfrecords-jpeg-512x512')

TEST_FNS = tf.io.gfile.glob(os.path.join(GCS_PATH, 'test/*.tfrec'))

IMG_DIM = (512, 512)
AugParams = {

    'scale_factor':0.5,

    'scale_prob':0.5,

    'rot_range':90,

    'rot_prob':0.5,

    'blockout_sl':0.1,

    'blockout_sh':0.2,

    'blockout_rl':0.4,

    'blockout_prob':0.5,

    'blur_ksize':3,

    'blur_sigma':1,

    'blur_prob':0.5

}
def image_rotate(image, angle):



    if len(image.get_shape().as_list()) != 3:

        raise ValueError('`image_rotate` only support image with 3 dimension(h, w, c)`')



    angle = tf.cast(angle, tf.float32)

    h, w, c = IMG_DIM[0], IMG_DIM[1], 3

    cy, cx = h//2, w//2



    ys = tf.range(h)

    xs = tf.range(w)



    ys_vec = tf.tile(ys, [w])

    xs_vec = tf.reshape( tf.tile(xs, [h]), [h,w] )

    xs_vec = tf.reshape( tf.transpose(xs_vec, [1,0]), [-1])



    ys_vec_centered, xs_vec_centered = ys_vec - cy, xs_vec - cx

    new_coord_centered = tf.cast(tf.stack([ys_vec_centered, xs_vec_centered]), tf.float32)



    inv_rot_mat = tf.reshape( tf.dynamic_stitch([0,1,2,3], [tf.cos(angle), tf.sin(angle), -tf.sin(angle), tf.cos(angle)]), [2,2])

    old_coord_centered = tf.matmul(inv_rot_mat, new_coord_centered)



    old_ys_vec_centered, old_xs_vec_centered = old_coord_centered[0,:], old_coord_centered[1,:]

    old_ys_vec = tf.cast( tf.round(old_ys_vec_centered+cy), tf.int32)

    old_xs_vec = tf.cast( tf.round(old_xs_vec_centered+cx), tf.int32)



    outside_ind = tf.logical_or( tf.logical_or(old_ys_vec > h-1 , old_ys_vec < 0), tf.logical_or(old_xs_vec > w-1 , old_xs_vec<0))



    old_ys_vec = tf.boolean_mask(old_ys_vec, tf.logical_not(outside_ind))

    old_xs_vec = tf.boolean_mask(old_xs_vec, tf.logical_not(outside_ind))



    ys_vec = tf.boolean_mask(ys_vec, tf.logical_not(outside_ind))

    xs_vec = tf.boolean_mask(xs_vec, tf.logical_not(outside_ind))



    old_coord = tf.cast(tf.transpose(tf.stack([old_ys_vec, old_xs_vec]), [1,0]), tf.int32)

    new_coord = tf.cast(tf.transpose(tf.stack([ys_vec, xs_vec]), [1,0]), tf.int64)



    channel_vals = tf.split(image, c, axis=-1)

    rotated_channel_vals = list()

    for channel_val in channel_vals:

        rotated_channel_val = tf.gather_nd(channel_val, old_coord)



        sparse_rotated_channel_val = tf.SparseTensor(new_coord, tf.squeeze(rotated_channel_val,axis=-1), [h, w])

        rotated_channel_vals.append(tf.sparse.to_dense(sparse_rotated_channel_val, default_value=0, validate_indices=False))



    rotated_image = tf.transpose(tf.stack(rotated_channel_vals), [1, 2, 0])

    return rotated_image

    

def random_blockout(img, sl=0.1, sh=0.2, rl=0.4):



    h, w, c = IMG_DIM[0], IMG_DIM[1], 3

    origin_area = tf.cast(h*w, tf.float32)



    e_size_l = tf.cast(tf.round(tf.sqrt(origin_area * sl * rl)), tf.int32)

    e_size_h = tf.cast(tf.round(tf.sqrt(origin_area * sh / rl)), tf.int32)



    e_height_h = tf.minimum(e_size_h, h)

    e_width_h = tf.minimum(e_size_h, w)



    erase_height = tf.random.uniform(shape=[], minval=e_size_l, maxval=e_height_h, dtype=tf.int32)

    erase_width = tf.random.uniform(shape=[], minval=e_size_l, maxval=e_width_h, dtype=tf.int32)



    erase_area = tf.zeros(shape=[erase_height, erase_width, c])

    erase_area = tf.cast(erase_area, tf.uint8)



    pad_h = h - erase_height

    pad_top = tf.random.uniform(shape=[], minval=0, maxval=pad_h, dtype=tf.int32)

    pad_bottom = pad_h - pad_top



    pad_w = w - erase_width

    pad_left = tf.random.uniform(shape=[], minval=0, maxval=pad_w, dtype=tf.int32)

    pad_right = pad_w - pad_left



    erase_mask = tf.pad([erase_area], [[0,0],[pad_top, pad_bottom], [pad_left, pad_right], [0,0]], constant_values=1)

    erase_mask = tf.squeeze(erase_mask, axis=0)

    erased_img = tf.multiply(tf.cast(img,tf.float32), tf.cast(erase_mask, tf.float32))



    return tf.cast(erased_img, img.dtype)

    

def zoom_out(x, scale_factor):



    resize_x = tf.random.uniform(shape=[], minval=tf.cast(IMG_DIM[1]//(1/scale_factor), tf.int32), maxval=IMG_DIM[1], dtype=tf.int32)

    resize_y = tf.random.uniform(shape=[], minval=tf.cast(IMG_DIM[0]//(1/scale_factor), tf.int32), maxval=IMG_DIM[0], dtype=tf.int32)

    top_pad = (IMG_DIM[0] - resize_y) // 2

    bottom_pad = IMG_DIM[0] - resize_y - top_pad

    left_pad = (IMG_DIM[1] - resize_x ) // 2

    right_pad = IMG_DIM[1] - resize_x - left_pad

        

    x = tf.image.resize(x, (resize_y, resize_x))

    x = tf.pad([x], [[0,0], [top_pad, bottom_pad], [left_pad, right_pad], [0,0]])

    x = tf.image.resize(x, IMG_DIM)

    return tf.squeeze(x, axis=0)

    

def zoom_in(x, scale_factor):



    scales = list(np.arange(0.5, 1.0, 0.05))

    boxes = np.zeros((len(scales),4))

            

    for i, scale in enumerate(scales):

        x_min = y_min = 0.5 - (0.5*scale)

        x_max = y_max = 0.5 + (0.5*scale)

        boxes[i] = [x_min, y_min, x_max, y_max]

        

    def random_crop(x):

        crop = tf.image.crop_and_resize([x], boxes=boxes, box_indices=np.zeros(len(boxes)), crop_size=IMG_DIM)

        return crop[tf.random.uniform(shape=[], minval=0, maxval=len(scales), dtype=tf.int32)]

        

    return random_crop(x)



def gaussian_blur(img, ksize=5, sigma=1):

    

    def gaussian_kernel(size=3, sigma=1):



        x_range = tf.range(-(size-1)//2, (size-1)//2 + 1, 1)

        y_range = tf.range((size-1)//2, -(size-1)//2 - 1, -1)



        xs, ys = tf.meshgrid(x_range, y_range)

        kernel = tf.exp(-(xs**2 + ys**2)/(2*(sigma**2))) / (2*np.pi*(sigma**2))

        return tf.cast( kernel / tf.reduce_sum(kernel), tf.float32)

    

    kernel = gaussian_kernel(ksize, sigma)

    kernel = tf.expand_dims(tf.expand_dims(kernel, axis=-1), axis=-1)

    

    r, g, b = tf.split(img, [1,1,1], axis=-1)

    r_blur = tf.nn.conv2d([r], kernel, [1,1,1,1], 'SAME')

    g_blur = tf.nn.conv2d([g], kernel, [1,1,1,1], 'SAME')

    b_blur = tf.nn.conv2d([b], kernel, [1,1,1,1], 'SAME')



    blur_image = tf.concat([r_blur, g_blur, b_blur], axis=-1)

    return tf.squeeze(blur_image, axis=0)
def augmentation(image):

    image = tf.cast(image, tf.float32)

    #Gaussian blur

    if tf.random.uniform(shape=[], minval=0.0, maxval=1.0) > AugParams['blur_prob']:

        image = gaussian_blur(image, AugParams['blur_ksize'], AugParams['blur_sigma'])

    

    #Random block out

    if tf.random.uniform(shape=[], minval=0.0, maxval=1.0) > AugParams['blockout_prob']:

        image = random_blockout(image, AugParams['blockout_sl'], AugParams['blockout_sh'], AugParams['blockout_rl'])

        

    #Random scale

    if tf.random.uniform(shape=[], minval=0.0, maxval=1.0) > AugParams['scale_prob']:

        if tf.random.uniform(shape=[], minval=0.0, maxval=1.0) > 0.5:

            image = zoom_in(image, AugParams['scale_factor'])

        else:

            image = zoom_out(image, AugParams['scale_factor'])

    #Random rotate

    if tf.random.uniform(shape=[], minval=0.0, maxval=1.0) > AugParams['rot_prob']:

        angle = tf.random.uniform(shape=[], minval=-AugParams['rot_range'], maxval=AugParams['rot_range'], dtype=tf.int32)

        image = image_rotate(image,angle)

    

    return tf.cast(image, tf.uint8)

    

def decoded_example(example):

    example = tf.io.parse_single_example(example, { "image" : tf.io.FixedLenFeature([], tf.string) })

    bits = example['image']

    image = tf.image.decode_jpeg(bits)

    return image



ds = tf.data.TFRecordDataset(TEST_FNS)

ds = ds.map(decoded_example)

ds = ds.map(augmentation)

ds = ds.batch(25)
images = next(iter(ds))

images = images.numpy()

plt.figure(figsize=(24,24))

col = 5

row = 5

for idx, image in enumerate(images):

    plt.subplot(row, col, idx+1)

    plt.imshow(image)

plt.show()