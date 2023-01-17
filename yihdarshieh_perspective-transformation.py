import tensorflow as tf

import numpy as np

from matplotlib import pyplot as plt

import re

import math

import datetime

from kaggle_datasets import KaggleDatasets

import tensorflow.keras.backend as K

AUTO = tf.data.experimental.AUTOTUNE
M = tf.convert_to_tensor([[1, 2, 3], [3, 4, 5], [4, 5, 6]], dtype=tf.float64)

mask = tf.convert_to_tensor([[0, 0, 0], [0, 1, 1], [0, 1, 1]])

indices = tf.convert_to_tensor(

    [

        [[1, 1], [1, 2]],

        [[2, 1], [2, 2]]

    ]

)

tf.gather_nd(M, indices)

indices = tf.convert_to_tensor(

    [

        [

            [[1, 1], [1, 2]],

            [[2, 1], [2, 2]]

        ],

        [

            [[1, 0], [1, 2]],

            [[2, 0], [2, 2]]

        ],

        [

            [[1, 0], [1, 1]],

            [[2, 0], [2, 1]]

        ],

        [

            [[0, 1], [0, 2]],

            [[2, 1], [2, 2]]

        ],

        [

            [[0, 0], [0, 2]],

            [[2, 0], [2, 2]]

        ],

        [

            [[0, 0], [0, 1]],

            [[2, 0], [2, 1]]

        ],

        [

            [[0, 1], [0, 2]],

            [[1, 1], [1, 2]]

        ],

        [

            [[0, 0], [0, 2]],

            [[1, 0], [1, 2]]

        ],

        [

            [[0, 0], [0, 1]],

            [[1, 0], [1, 1]]

        ]         

    ]

)

indices = tf.reshape(indices, shape=(3, 3, 2, 2, 2))

indices



tf.gather_nd(M, indices)



tf.linalg.det(tf.gather_nd(M, indices))
image_size = 192

IMAGE_SIZE = [image_size, image_size]

BATCH_SIZE = 64

AUG_BATCH = BATCH_SIZE



# Data access

GCS_DS_PATH = KaggleDatasets().get_gcs_path('tpu-getting-started')





GCS_PATH_SELECT = { # available image sizes

    192: GCS_DS_PATH + '/tfrecords-jpeg-192x192'



}

GCS_PATH = GCS_PATH_SELECT[IMAGE_SIZE[0]]



TRAINING_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/train/*.tfrec')



def decode_image(image_data):

    image = tf.image.decode_jpeg(image_data, channels=3)

    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range

    image = tf.reshape(image, [*IMAGE_SIZE, 3]) # explicit size needed for TPU

    return image



def read_labeled_tfrecord(example):

    LABELED_TFREC_FORMAT = {

        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring

        "class": tf.io.FixedLenFeature([], tf.int64),  # shape [] means single element

    }

    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)

    image = decode_image(example['image'])

    label = tf.cast(example['class'], tf.int32)

    return image, label # returns a dataset of (image, label) pairs



def load_dataset(filenames, labeled = True, ordered = False):

    # Read from TFRecords. For optimal performance, reading from multiple files at once and

    # Diregarding data order. Order does not matter since we will be shuffling the data anyway

    

    ignore_order = tf.data.Options()

    if not ordered:

        ignore_order.experimental_deterministic = False # disable order, increase speed

        

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads = AUTO) # automatically interleaves reads from multiple files

    dataset = dataset.with_options(ignore_order) # use data as soon as it streams in, rather than in its original order

    dataset = dataset.map(read_labeled_tfrecord if labeled else read_unlabeled_tfrecord, num_parallel_calls = AUTO) # returns a dataset of (image, label) pairs if labeled = True or (image, id) pair if labeld = False

    return dataset



def count_data_items(filenames):

    # the number of data items is written in the name of the .tfrec files, i.e. flowers00-230.tfrec = 230 data items

    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]

    return np.sum(n)



NUM_TRAINING_IMAGES = int(count_data_items(TRAINING_FILENAMES))



print('Dataset: {} training images'.format(NUM_TRAINING_IMAGES))
def batch_to_numpy_images_and_labels(data):

    

    images = data

    numpy_images = images.numpy()



    # If no labels, only image IDs, return None for labels (this is the case for test data)

    return numpy_images





def display_one_flower(image, title, subplot, red=False, titlesize=16):

    plt.subplot(*subplot)

    plt.axis('off')

    plt.imshow(image)

    if len(title) > 0:

        plt.title(title, fontsize=int(titlesize) if not red else int(titlesize/1.2), color='red' if red else 'black', fontdict={'verticalalignment':'center'}, pad=int(titlesize/1.5))

    return (subplot[0], subplot[1], subplot[2]+1)





def display_batch_of_images(databatch, predictions=None):

    

    """This will work with:

    display_batch_of_images(images)

    """

    

    # data

    images = batch_to_numpy_images_and_labels(databatch)

    labels = None

    

    if labels is None:

        labels = [None for _ in enumerate(images)]

        

    # auto-squaring: this will drop data that does not fit into square or square-ish rectangle

    rows = int(math.sqrt(len(images)))

    cols = len(images)//rows

        

    # size and spacing

    FIGSIZE = 13.0

    SPACING = 0.1

    subplot=(rows,cols,1)

    if rows < cols:

        plt.figure(figsize=(FIGSIZE,FIGSIZE/cols*rows))

    else:

        plt.figure(figsize=(FIGSIZE/rows*cols,FIGSIZE))

    

    # display

    for i, (image, label) in enumerate(zip(images[:rows*cols], labels[:rows*cols])):

        title = '' if label is None else CLASSES[label]

        correct = True

        if predictions is not None:

            title, correct = title_from_label_and_target(predictions[i], label)

        dynamic_titlesize = FIGSIZE*SPACING/max(rows,cols)*40+3 # magic formula tested to work from 1x1 to 10x10 images

        subplot = display_one_flower(image, title, subplot, not correct, titlesize=dynamic_titlesize)

    

    #layout

    plt.tight_layout()

    if label is None and predictions is None:

        plt.subplots_adjust(wspace=0, hspace=0)

    else:

        plt.subplots_adjust(wspace=SPACING, hspace=SPACING)

    plt.show()
def get_training_dataset(dataset, batch_size=None, advanced_aug=True, repeat=True, with_labels=True, drop_remainder=False):

    

    if not with_labels:

        dataset = dataset.map(lambda image, label: image, num_parallel_calls=AUTO)

    

    if advanced_aug:

        dataset = dataset.map(transform, num_parallel_calls=AUTO)

    

    if type(repeat) == bool and repeat:

        dataset = dataset.repeat() # the training dataset must repeat for several epochs

    elif type(repeat) == int and repeat > 0:

        dataset = dataset.repeat(repeat)

    

    dataset = dataset.shuffle(2048)

    

    if batch_size is None:

        batch_size = BATCH_SIZE

    

    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    

    return dataset
dataset = load_dataset(TRAINING_FILENAMES, labeled=True)

training_dataset = get_training_dataset(dataset, advanced_aug=False, repeat=1, with_labels=True)



images, labels = next(iter(training_dataset.take(1)))

print(images.shape)



display_batch_of_images(images)
def random_4_points_2D(height, width):

    """Generate 4 random 2-D points.

    

    The 4 points are inside a rectangle with the same center as the above rectangle but with side length being approximately 1.5 times.

    This choice is to avoid the image being transformed too disruptively.

    

    Each point is created first by making it close to the corresponding corner points determined by the rectangle, i.e

    [0, 0], [0, width], [height, width] and [height, 0] respectively. Then the 4 points are randomly shifted module 4.

    

    Args:

        height: 0-D tensor, height of a reference rectangle.

        width: 0-D tensor, width of a reference rectangle.

        

    Returns:

        points: 2-D tensor of shape [4, 2]

    """



    sy = height // 4

    sx = width // 4

        

    h, w = height, width

    

    y1 = tf.random.uniform(minval = -sy, maxval = sy, shape=[], dtype=tf.int64)

    x1 = tf.random.uniform(minval = -sx, maxval = sx, shape=[], dtype=tf.int64)

    

    y2 = tf.random.uniform(minval = -sy, maxval = sy, shape=[], dtype=tf.int64)

    x2 = tf.random.uniform(minval = 3 * sx, maxval = 5 * sx, shape=[], dtype=tf.int64)



    y3 = tf.random.uniform(minval = 3 * sy, maxval = 5 * sy, shape=[], dtype=tf.int64)

    x3 = tf.random.uniform(minval = 3 * sx, maxval = 5 * sx, shape=[], dtype=tf.int64)    

    

    y4 = tf.random.uniform(minval = 3 * sy, maxval = 5 * sy, shape=[], dtype=tf.int64)

    x4 = tf.random.uniform(minval = -sx, maxval = sx, shape=[], dtype=tf.int64)

                

    points = tf.convert_to_tensor([[y1, x1], [y2, x2], [y3, x3], [y4, x4], [y1, x1], [y2, x2], [y3, x3], [y4, x4]])

    start_index = tf.random.uniform(minval=0, maxval=4, shape=[], dtype=tf.int64)

    points = points[start_index: start_index + 4]

        

    return points





def random_4_point_transform_2D(image):

    """Apply 4 point transformation on 2-D image `image` with randomly generated 4 points on target spaces.

    

    On source space, the 4 points are the corner points, i.e [0, 0], [0, width], [height, width] and [height, 0].

    On target space, the 4 points are randomly generated by `random_4_points_2D()`.

    """

    

    height, width = image.shape[:2]



    # 4 corner points in source image

    # shape = [4, 2]

    src_pts = tf.convert_to_tensor([[0, 0], [0, width], [height, width], [height, 0]])



    # 4 points in target image

    # shape = [4, 2]

    tgt_pts = random_4_points_2D(height, width)

    

    tgt_image = four_point_transform_2D(image, src_pts, tgt_pts)



    return tgt_image





def four_point_transform_2D(image, src_pts, tgt_pts):

    """Apply 4 point transformation determined by `src_pts` and `tgt_pts` on 2-D image `image`.

    

    Args:

        image: 2-D tensor of shape [height, width], or 3-D tensor of shape [height, width, channels]

        src_pts: 2-D tensor of shape [4, 2]

        tgt_pts: 2-D tensor of shape [4, 2]

        

    Returns:

        A tensor with the same shape as `image`.

    """

    

    src_to_tgt_mat = get_src_to_tgt_mat_2D(src_pts, tgt_pts)

    

    tgt_image = transform_by_perspective_matrix_2D(image, src_to_tgt_mat)

    

    return tgt_image





def transform_by_perspective_matrix_2D(image, src_to_tgt_mat):

    """Transform a 2-D image by a prespective transformation matrix

    

    Args:

        image: 2-D tensor of shape [height, width], or 3-D tensor of shape [height, width, channels]

        src_to_tgt_mat: 2-D tensor of shape [3, 3]. This is the transformation matrix mapping the source space to the target space.

        

    Returns:

        A tensor with the same shape as `image`.        

    """



    height, width = image.shape[:2]



    # shape = (3, 3)

    tgt_to_src_mat = tf.linalg.inv(src_to_tgt_mat)

        

    # prepare y coordinates

    # shape = [height * width]

    ys = tf.repeat(tf.range(height), width)  



    # prepare x coordinates

    # shape = [height * width]

    xs = tf.tile(tf.range(width), [height])



    # prepare indices in target space

    # shape = [2, height * width]

    tgt_indices = tf.stack([ys, xs], axis=0)

    

    # Change to projective coordinates in the target space by adding ones

    # shape = [3, height * width]

    tgt_indices_homo = tf.concat([tgt_indices, tf.ones(shape=[1, height * width], dtype=tf.int32)], axis=0)

    

    # Get the corresponding projective coordinate in the source space

    # shape = [3, height * width]

    src_indices_homo = tf.linalg.matmul(tgt_to_src_mat, tf.cast(tgt_indices_homo, dtype=tf.float64))

    

    # normalize the projective coordinates

    # shape = [3, height * width]

    src_indices_normalized = src_indices_homo[:3, :] / src_indices_homo[2:, :]

    

    # Get the affine coordinate by removing ones

    # shape = [2, height * width]

    src_indices_affine = tf.cast(src_indices_normalized, dtype=tf.int64)[:2, :]

    

    # Mask the points outside the range

    # shape = [height * width]

    y_mask = tf.logical_and(src_indices_affine[0] >= 0, src_indices_affine[0] <= height - 1)

    x_mask = tf.logical_and(src_indices_affine[1] >= 0, src_indices_affine[1] <= width - 1)

    mask = tf.logical_and(y_mask, x_mask)

    

    # clip the coordinates

    # shape = [2, height * width]

    src_indices = tf.clip_by_value(src_indices_affine, clip_value_min=0, clip_value_max=[[height - 1], [width - 1]])

    

    # Get a collection of (y_coord, x_coord)

    # shape = [height * width, 2]

    src_indices = tf.transpose(src_indices)

    

    # shape = [height * width, channels]

    tgt_image = tf.gather_nd(image, src_indices)

    

    # Set pixel to 0 by using the mask

    tgt_image = tgt_image * tf.cast(mask[:, tf.newaxis], tf.float32)

    

    # reshape to [height, width, channels]

    tgt_image = tf.reshape(tgt_image, image.shape)



    return tgt_image





def get_src_to_tgt_mat_2D(src_pts, tgt_pts):

    """Get the perspective transformation matrix from the source space to the target space, which maps the 4 source points to the 4 target points.

    

    Args:

        src_pts: 2-D tensor of shape [4, 2]

        tgt_pts: 2-D tensor of shape [4, 2]

        

    Returns:

        2-D tensor of shape [3, 3]

    """

    

    src_pts = tf.cast(src_pts, tf.int64)

    tgt_pts = tf.cast(tgt_pts, tf.int64)

    

    # The perspective transformation matrix mapping basis vectors and (1, 1, 1) to `src_pts`

    # shape = [3, 3]

    src_mat = get_transformation_mat_2D(src_pts)

    

    # The perspective transformation matrix mapping basis vectors and (1, 1, 1) to `tgt_pts`

    # shape = [3, 3]

    tgt_mat = get_transformation_mat_2D(tgt_pts)

    

    # The perspective transformation matrix mapping `src_pts` to `tgt_pts`

    # shape = [3, 3]

    src_to_tgt_mat = tf.linalg.matmul(tgt_mat, tf.linalg.inv(src_mat))

    

    return src_to_tgt_mat

  

    

def get_transformation_mat_2D(four_pts):

    """Get the perspective transformation matrix from a space to another space, which maps the basis vectors and (1, 1, 1) to the 4 points defined by `four_pts`.

    

    Args:

        four_pts: 2-D tensor of shape [4, 2]

        

    Returns:

        2-D tensor of shape [3, 3]        

    """

    

    # Change to projective coordinates by adding ones

    # shape = [3, 4]

    #pts_homo = tf.concat([tf.transpose(four_pts), tf.ones(shape=[1, 4], dtype=tf.int64)], axis=0)

    pts_homo = tf.transpose(tf.concat([four_pts, tf.ones(shape=[4, 1], dtype=tf.int64)], axis=-1))

    

    pts_homo = tf.cast(pts_homo, tf.float64)

    

    # Find `scalars` such that: src_pts_homo[:, 3:] * scalars == src_pts_homo[:, 3:]

    # shape = [3, 3]

    inv_mat = tf.linalg.inv(pts_homo[:, :3])

    # shape = [3, 1]

    scalars = tf.linalg.matmul(inv_mat, pts_homo[:, 3:])

    

    # Get the matrix transforming unit vectors to the 4 source points

    # shape = [3, 3]

    mat = tf.transpose(tf.transpose(pts_homo[:, :3]) * scalars)

    

    return mat
new_image = random_4_point_transform_2D(images[0])

display_batch_of_images(tf.convert_to_tensor([images[0], new_image]))
new_image = random_4_point_transform_2D(images[1])

display_batch_of_images(tf.convert_to_tensor([images[1], new_image]))
new_image = random_4_point_transform_2D(images[2])

display_batch_of_images(tf.convert_to_tensor([images[2], new_image]))
new_image = random_4_point_transform_2D(images[3])

display_batch_of_images(tf.convert_to_tensor([images[3], new_image]))
def random_4_points_2D_batch(height, width, batch_size):

    """Generate `batch_size * 4` random 2-D points.

    

    Each 4 points are inside a rectangle with the same center as the above rectangle but with side length being approximately 1.5 times.

    This choice is to avoid the image being transformed too disruptively.



    Each point is created first by making it close to the corresponding corner points determined by the rectangle, i.e

    [0, 0], [0, width], [height, width] and [height, 0] respectively. Then the 4 points are randomly shifted module 4.

    

    Args:

        height: 0-D tensor, height of a reference rectangle.

        width: 0-D tensor, width of a reference rectangle.

        batch_size: 0-D tensor, the number of 4 points to be generated

        

    Returns:

        points: 3-D tensor of shape [batch_size, 4, 2]

    """



    sy = height // 4

    sx = width // 4

        

    h, w = height, width

    

    y1 = tf.random.uniform(minval = -sy, maxval = sy, shape=[batch_size], dtype=tf.int64)

    x1 = tf.random.uniform(minval = -sx, maxval = sx, shape=[batch_size], dtype=tf.int64)

    

    y2 = tf.random.uniform(minval = -sy, maxval = sy, shape=[batch_size], dtype=tf.int64)

    x2 = tf.random.uniform(minval = 3 * sx, maxval = 5 * sx, shape=[batch_size], dtype=tf.int64)



    y3 = tf.random.uniform(minval = 3 * sy, maxval = 5 * sy, shape=[batch_size], dtype=tf.int64)

    x3 = tf.random.uniform(minval = 3 * sx, maxval = 5 * sx, shape=[batch_size], dtype=tf.int64)    

    

    y4 = tf.random.uniform(minval = 3 * sy, maxval = 5 * sy, shape=[batch_size], dtype=tf.int64)

    x4 = tf.random.uniform(minval = -sx, maxval = sx, shape=[batch_size], dtype=tf.int64)

            

    # shape = [4, 2, batch_size]

    points = tf.convert_to_tensor([[y1, x1], [y2, x2], [y3, x3], [y4, x4], [y1, x1], [y2, x2], [y3, x3], [y4, x4]])

    

    # shape = [batch_size, 4, 2]

    points = tf.transpose(points, perm=[2, 0, 1])

    

    # Trick to get random rotation

    # shape = [batch_size, 8, 2]

    points = tf.tile(points, multiples=[1, 2, 1])    

    # shape = [batch_size]

    start_indices = tf.random.uniform(minval=0, maxval=4, shape=[batch_size], dtype=tf.int64)

    # shape = [batch_size, 4]

    indices = start_indices[:, tf.newaxis] + tf.range(4, dtype=tf.int64)[tf.newaxis, :]

    # shape = [batch_size, 4, 2]

    indices = tf.stack([tf.broadcast_to(tf.range(batch_size, dtype=tf.int64)[:, tf.newaxis], shape=[batch_size, 4]), indices], axis=2)    

    

    # shape = [batch_size, 4, 2]

    points = tf.gather_nd(points, indices)

        

    return points





def random_4_point_transform_2D_batch(images):

    """Apply 4 point transformation on 2-D images `images` with randomly generated 4 points on target spaces.

    

    On source space, the 4 points are the corner points, i.e [0, 0], [0, width], [height, width] and [height, 0].

    On target space, the 4 points are randomly generated by `random_4_points_2D_batch()`.

    """



    batch_size, height, width = images.shape[:3]



    # 4 corner points in source image

    # shape = [batch_size, 4, 2]

    src_pts = tf.convert_to_tensor([[0, 0], [0, width], [height, width], [height, 0]])

    src_pts = tf.broadcast_to(src_pts, shape=[batch_size, 4, 2])



    # 4 points in target image

    # shape = [batch_size, 4, 2]

    tgt_pts = random_4_points_2D_batch(height, width, batch_size)

    

    tgt_images = four_point_transform_2D_batch(images, src_pts, tgt_pts)



    return tgt_images





def four_point_transform_2D_batch(images, src_pts, tgt_pts):

    """Apply 4 point transformation determined by `src_pts` and `tgt_pts` on 2-D images `images`.

    

    Args:

        images: 3-D tensor of shape [batch_size, height, width], or 4-D tensor of shape [batch_size, height, width, channels]

        src_pts: 3-D tensor of shape [batch_size, 4, 2]

        tgt_pts: 3-D tensor of shape [batch_size, 4, 2]

        

    Returns:

        A tensor with the same shape as `images`.

    """

    

    src_to_tgt_mat = get_src_to_tgt_mat_2D_batch(src_pts, tgt_pts)

    

    tgt_images = transform_by_perspective_matrix_2D_batch(images, src_to_tgt_mat)

    

    return tgt_images





def transform_by_perspective_matrix_2D_batch(images, src_to_tgt_mat):

    """Transform 2-D images by prespective transformation matrices

    

    Args:

        images: 3-D tensor of shape [batch_size, height, width], or 4-D tensor of shape [batch_size, height, width, channels]

        src_to_tgt_mat: 3-D tensor of shape [batch_size, 3, 3]. This is the transformation matrix mapping the source space to the target space.

        

    Returns:

        A tensor with the same shape as `image`.        

    """



    batch_size, height, width = images.shape[:3]



    # shape = (3, 3)

    tgt_to_src_mat = tf.linalg.inv(src_to_tgt_mat)

        

    # prepare y coordinates

    # shape = [height * width]

    ys = tf.repeat(tf.range(height), width) 

    

    # prepare x coordinates

    # shape = [height * width]

    xs = tf.tile(tf.range(width), [height])



    # prepare indices in target space

    # shape = [2, height * width]

    tgt_indices = tf.stack([ys, xs], axis=0)

    

    # Change to projective coordinates in the target space by adding ones

    # shape = [3, height * width]

    tgt_indices_homo = tf.concat([tgt_indices, tf.ones(shape=[1, height * width], dtype=tf.int32)], axis=0)

    

    # Get the corresponding projective coordinate in the source space

    # shape = [batch_size, 3, height * width]

    src_indices_homo = tf.linalg.matmul(tgt_to_src_mat, tf.cast(tgt_indices_homo, dtype=tf.float64))

    

    # normalize the projective coordinates

    # shape = [batch_size, 3, height * width]

    src_indices_normalized = src_indices_homo[:, :3, :] / src_indices_homo[:, 2:, :]

    

    # Get the affine coordinate by removing ones

    # shape = [batch_size, 2, height * width]

    src_indices_affine = tf.cast(src_indices_normalized, dtype=tf.int64)[:, :2, :]

    

    # Mask the points outside the range

    # shape = [batch_size, height * width]

    y_mask = tf.logical_and(src_indices_affine[:, 0] >= 0, src_indices_affine[:, 0] <= height - 1)

    x_mask = tf.logical_and(src_indices_affine[:, 1] >= 0, src_indices_affine[:, 1] <= width - 1)

    mask = tf.logical_and(y_mask, x_mask)

    

    # clip the coordinates

    # shape = [batch_size, 2, height * width]

    src_indices = tf.clip_by_value(src_indices_affine, clip_value_min=0, clip_value_max=[[height - 1], [width - 1]])

    

    # Get a collection of (y_coord, x_coord)

    # shape = [batch_size, height * width, 2]

    src_indices = tf.transpose(src_indices, perm=[0, 2, 1])

    

    # shape = [batch_size, height * width, channels]

    tgt_images = tf.gather_nd(images, src_indices, batch_dims=1)

    

    # Set pixel to 0 by using the mask

    tgt_images = tgt_images * tf.cast(mask[:, :, tf.newaxis], tf.float32)

    

    # reshape to [height, width, channels]

    tgt_images = tf.reshape(tgt_images, images.shape)



    return tgt_images





def get_src_to_tgt_mat_2D_batch(src_pts, tgt_pts):

    """Get the perspective transformation matrix from the source space to the target space, which maps the 4 source points to the 4 target points.

    

    Args:

        src_pts: 3-D tensor of shape [batch_size, 4, 2]

        tgt_pts: 3-D tensor of shape [batch_size, 4, 2]

        

    Returns:

        2-D tensor of shape [batch_size, 3, 3]

    """

    

    src_pts = tf.cast(src_pts, tf.int64)

    tgt_pts = tf.cast(tgt_pts, tf.int64)

    

    # The perspective transformation matrix mapping basis vectors and (1, 1, 1) to `src_pts`

    # shape = [batch_size, 3, 3]

    src_mat = get_transformation_mat_2D_batch(src_pts)

    

    # The perspective transformation matrix mapping basis vectors and (1, 1, 1) to `tgt_pts`

    # shape = [3, 3]

    tgt_mat = get_transformation_mat_2D_batch(tgt_pts)

    

    # The perspective transformation matrix mapping `src_pts` to `tgt_pts`

    # shape = [3, 3]

    src_to_tgt_mat = tf.linalg.matmul(tgt_mat, tf.linalg.inv(src_mat))

    

    return src_to_tgt_mat

  

    

def get_transformation_mat_2D_batch(four_pts):

    """Get the perspective transformation matrix from a space to another space, which maps the basis vectors and (1, 1, 1) to the 4 points defined by `four_pts`.

    

    Args:

        four_pts: 3-D tensor of shape [batch_size, 4, 2]

        

    Returns:

        3-D tensor of shape [batch_size, 3, 3]        

    """

    

    batch_size = four_pts.shape[0]

    

    # Change to projective coordinates by adding ones

    # shape = [batch_size, 3, 4]

    pts_homo = tf.transpose(tf.concat([four_pts, tf.ones(shape=[batch_size, 4, 1], dtype=tf.int64)], axis=-1), perm=[0, 2, 1])

    

    pts_homo = tf.cast(pts_homo, tf.float64)

    

    # Find `scalars` such that: src_pts_homo[:, 3:] * scalars == src_pts_homo[:, 3:]

    # shape = [batch_size 3, 3]

    inv_mat = tf.linalg.inv(pts_homo[:, :, :3])

    # shape = [batch_size, 3, 1]

    scalars = tf.linalg.matmul(inv_mat, pts_homo[:, :, 3:])

    

    # Get the matrix transforming unit vectors to the 4 source points

    # shape = [batch_size, 3, 3]    

    mat = tf.transpose(tf.transpose(pts_homo[:, :, :3], perm=[0, 2, 1]) * scalars, perm=[0, 2, 1])

    

    return mat
new_images = random_4_point_transform_2D_batch(tf.repeat(images[:16], axis=0, repeats=4))

display_batch_of_images(new_images)
new_images = random_4_point_transform_2D_batch(tf.repeat(images[16:32], axis=0, repeats=4))

display_batch_of_images(new_images)
new_images = random_4_point_transform_2D_batch(images)

display_batch_of_images(new_images)
n_iter = 1000



start = datetime.datetime.now()

for i in range(n_iter):

    random_4_point_transform_2D_batch(images) 

end = datetime.datetime.now()

timing = (end - start).total_seconds() / n_iter

print(f"batch_4_pt_transformation: {timing}")