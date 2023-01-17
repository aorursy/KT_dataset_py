# 以狗分类为例子，可以自由使用所有方法以及其组合，可以自由更改数据集

# baseline：EfficientNetB0，可以自由更改网络



# 可以选择启用哪些方法



bool_random_flip_left_right=0

bool_random_flip_up_down=0

bool_random_brightness=0

bool_random_contrast=0

bool_random_hue=0

bool_random_saturation=0



bool_rotation_transform=0

cutmix_rate=0.

mixup_rate= 0.

gridmask_rate = 0.



pre_trained='imagenet' # None,'imagenet','noisy-student'

dense_activation='softmax' #'softmax','sigmoid'

bool_lr_scheduler=0



# 交叉验证和tta目前最多只允许使用一个

tta_times=0 #当tta_times=i>0时，使用i+1倍测试集

cross_validation_folds=0 #当tta_times=i>1时，使用i折交叉验证





#focal_loss和label_smoothing最多同时使用一个

bool_focal_loss = 0

label_smoothing_rate=0.



# 划分了训练集和验证集时才有效,该功能暂时作废

# special_monitor='auc'#None,'auc'





# 暂未引入，后续引入

# adversarial validation
#针对前面opts的一些中间处理



#tta只有在使用了data_aug时才允许启用

bool_tta =  tta_times and max(  bool_random_flip_left_right,

                                bool_random_flip_up_down,

                                bool_random_brightness,

                                bool_random_contrast,

                                bool_random_hue,

                                bool_random_saturation)



print(bool_tta)



assert (bool_focal_loss and label_smoothing_rate) == 0 , 'focal_loss和label_smoothing最多同时使用一个'

assert (tta_times and cross_validation_folds) == 0 , 'focal_loss和label_smoothing最多同时使用一个'    
#安装包

!pip install -U efficientnet

!pip install tensorflow_addons
#导入包

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt 

import os

import tensorflow as tf

import random, re, math

from tensorflow.keras.layers import Dense

from tensorflow.keras.models import Model

from tensorflow.keras import optimizers

import tensorflow_addons as tfa



from kaggle_datasets import KaggleDatasets

import efficientnet.tfkeras as efn

from sklearn.model_selection import KFold

from tensorflow.keras.callbacks import ModelCheckpoint



print(tf.__version__)

print(tf.keras.__version__)
#针对不同硬件产生不同配置

AUTO = tf.data.experimental.AUTOTUNE

# Detect hardware, return appropriate distribution strategy

try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

    GCS_DS_PATH = KaggleDatasets().get_gcs_path()

    print(GCS_DS_PATH)

else:

    strategy = tf.distribute.get_strategy() # default distribution strategy in Tensorflow. Works on CPU and single GPU.

print("REPLICAS: ", strategy.num_replicas_in_sync)
#更换数据集时更换整个这大段



#超参数，根据数据和策略调参

BATCH_SIZE = 16 * strategy.num_replicas_in_sync



img_size = 512

IMAGE_SIZE=(img_size, img_size)



#应感觉该是，用的数据增强方法增多，那么需要的EPOCHS也需要增多

EPOCHS = 12 #12,改为1快速测试

lr_if_without_scheduler = 0.0003

nb_classes = 104

print('BATCH_SIZE是：',BATCH_SIZE)



#如果使用了special_monitor，会在后续自动append，无需在此添加

my_metrics = ['accuracy']



GCS_PATH_SELECT = { # available image sizes

    192: GCS_DS_PATH + '/tfrecords-jpeg-192x192',

    224: GCS_DS_PATH + '/tfrecords-jpeg-224x224',

    331: GCS_DS_PATH + '/tfrecords-jpeg-331x331',

    512: GCS_DS_PATH + '/tfrecords-jpeg-512x512'

}





GCS_PATH = GCS_PATH_SELECT[img_size]





TRAINING_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/train/*.tfrec')



print(GCS_PATH)

#应该是这个才对  gs://kds-b2e6cdbc4af76dcf0363776c09c12fe46872cab211d1de9f60ec7aec/tfrecords-jpeg-512x512



VALIDATION_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/val/*.tfrec')

test_paths = tf.io.gfile.glob(GCS_PATH + '/test/*.tfrec') # predictions on this dataset should be submitted for the competition

train_paths=TRAINING_FILENAMES+VALIDATION_FILENAMES



print(TRAINING_FILENAMES)



CLASSES = ['pink primrose',    'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea',     'wild geranium',     'tiger lily',           'moon orchid',              'bird of paradise', 'monkshood',        'globe thistle',         # 00 - 09

           'snapdragon',       "colt's foot",               'king protea',      'spear thistle', 'yellow iris',       'globe-flower',         'purple coneflower',        'peruvian lily',    'balloon flower',   'giant white arum lily', # 10 - 19

           'fire lily',        'pincushion flower',         'fritillary',       'red ginger',    'grape hyacinth',    'corn poppy',           'prince of wales feathers', 'stemless gentian', 'artichoke',        'sweet william',         # 20 - 29

           'carnation',        'garden phlox',              'love in the mist', 'cosmos',        'alpine sea holly',  'ruby-lipped cattleya', 'cape flower',              'great masterwort', 'siam tulip',       'lenten rose',           # 30 - 39

           'barberton daisy',  'daffodil',                  'sword lily',       'poinsettia',    'bolero deep blue',  'wallflower',           'marigold',                 'buttercup',        'daisy',            'common dandelion',      # 40 - 49

           'petunia',          'wild pansy',                'primula',          'sunflower',     'lilac hibiscus',    'bishop of llandaff',   'gaura',                    'geranium',         'orange dahlia',    'pink-yellow dahlia',    # 50 - 59

           'cautleya spicata', 'japanese anemone',          'black-eyed susan', 'silverbush',    'californian poppy', 'osteospermum',         'spring crocus',            'iris',             'windflower',       'tree poppy',            # 60 - 69

           'gazania',          'azalea',                    'water lily',       'rose',          'thorn apple',       'morning glory',        'passion flower',           'lotus',            'toad lily',        'anthurium',             # 70 - 79

           'frangipani',       'clematis',                  'hibiscus',         'columbine',     'desert-rose',       'tree mallow',          'magnolia',                 'cyclamen ',        'watercress',       'canna lily',            # 80 - 89

           'hippeastrum ',     'bee balm',                  'pink quill',       'foxglove',      'bougainvillea',     'camellia',             'mallow',                   'mexican petunia',  'bromelia',         'blanket flower',        # 90 - 99

           'trumpet creeper',  'blackberry lily',           'common tulip',     'wild rose']                                                                                                                                               # 100 - 102







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

    label=tf.one_hot(label,nb_classes)

    return image, label # returns a dataset of (image, label) pairs



def read_unlabeled_tfrecord(example):

    UNLABELED_TFREC_FORMAT = {

        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring

        "id": tf.io.FixedLenFeature([], tf.string),  # shape [] means single element

        # class is missing, this competitions's challenge is to predict flower classes for the test dataset

    }

    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)

    image = decode_image(example['image'])

    idnum = example['id']

    return image, idnum # returns a dataset of image(s)



def load_dataset(filenames, labeled=True, ordered=False):

    # Read from TFRecords. For optimal performance, reading from multiple files at once and

    # disregarding data order. Order does not matter since we will be shuffling the data anyway.



    ignore_order = tf.data.Options()

    if not ordered:

        ignore_order.experimental_deterministic = False # disable order, increase speed



    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO) # automatically interleaves reads from multiple files

    dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order

    dataset = dataset.map(read_labeled_tfrecord if labeled else read_unlabeled_tfrecord, num_parallel_calls=AUTO)

    # returns a dataset of (image, label) pairs if labeled=True or (image, id) pairs if labeled=False

    return dataset
#lr_scheduler

#数值按实际情况设置



LR_START = 0.00003

LR_MAX = 0.0003 * strategy.num_replicas_in_sync

LR_MIN = 0.00003

LR_RAMPUP_EPOCHS = 3

LR_SUSTAIN_EPOCHS = 1

LR_EXP_DECAY = .8



def lrfn(epoch):

    if epoch < LR_RAMPUP_EPOCHS:

        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START

    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:

        lr = LR_MAX

    else:

        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY**(epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN

    return lr

    

lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)



rng = [i for i in range(EPOCHS)]

y = [lrfn(x) for x in rng]

if bool_lr_scheduler:

    plt.plot(rng, y)

    print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1]))
'''

# 这里比较特殊，需要重写decode_image

def decode_image(filename, label=None, image_size=(img_size, img_size)):

    bits = tf.io.read_file(filename)

    image = tf.image.decode_jpeg(bits, channels=3)

    image = tf.cast(image, tf.float32) / 255.0

    image = tf.image.resize(image, image_size)

    if label is None:

        return image

    else:

        return image, label

'''









# 只能写入test data 也能用的 aug   

def data_aug(image, label=None):

    if bool_random_flip_left_right:

        image = tf.image.random_flip_left_right(image)

    if bool_random_flip_up_down:    

        image = tf.image.random_flip_up_down(image)

    if bool_random_brightness:

        image = tf.image.random_brightness(image,0.2)

    if bool_random_contrast:

        image = tf.image.random_contrast(image,0.6,1.4)

    if bool_random_hue:

        image = tf.image.random_hue(image,0.07)

    if bool_random_saturation:

        image = tf.image.random_saturation(image,0.5,1.5)

    

    if label is None:

        return image

    else:

        return image, label
import tensorflow as tf, tensorflow.keras.backend as K

def get_mat(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):

    # returns 3x3 transformmatrix which transforms indicies

        

    # CONVERT DEGREES TO RADIANS

    rotation = math.pi * rotation / 180.

    shear = math.pi * shear / 180.

    

    # ROTATION MATRIX

    c1 = tf.math.cos(rotation)

    s1 = tf.math.sin(rotation)

    one = tf.constant([1],dtype='float32')

    zero = tf.constant([0],dtype='float32')

    rotation_matrix = tf.reshape( tf.concat([c1,s1,zero, -s1,c1,zero, zero,zero,one],axis=0),[3,3] )

        

    # SHEAR MATRIX

    c2 = tf.math.cos(shear)

    s2 = tf.math.sin(shear)

    shear_matrix = tf.reshape( tf.concat([one,s2,zero, zero,c2,zero, zero,zero,one],axis=0),[3,3] )    

    

    # ZOOM MATRIX

    zoom_matrix = tf.reshape( tf.concat([one/height_zoom,zero,zero, zero,one/width_zoom,zero, zero,zero,one],axis=0),[3,3] )

    

    # SHIFT MATRIX

    shift_matrix = tf.reshape( tf.concat([one,zero,height_shift, zero,one,width_shift, zero,zero,one],axis=0),[3,3] )

    

    return K.dot(K.dot(rotation_matrix, shear_matrix), K.dot(zoom_matrix, shift_matrix))





def rotation_transform(image,label):

    # input image - is one image of size [dim,dim,3] not a batch of [b,dim,dim,3]

    # output - image randomly rotated, sheared, zoomed, and shifted

    DIM = IMAGE_SIZE[0]

    XDIM = DIM%2 #fix for size 331

    

    rot = 15. * tf.random.normal([1],dtype='float32')

    shr = 5. * tf.random.normal([1],dtype='float32') 

    h_zoom = 1.0 + tf.random.normal([1],dtype='float32')/10.

    w_zoom = 1.0 + tf.random.normal([1],dtype='float32')/10.

    h_shift = 16. * tf.random.normal([1],dtype='float32') 

    w_shift = 16. * tf.random.normal([1],dtype='float32') 

  

    # GET TRANSFORMATION MATRIX

    m = get_mat(rot,shr,h_zoom,w_zoom,h_shift,w_shift) 



    # LIST DESTINATION PIXEL INDICES

    x = tf.repeat( tf.range(DIM//2,-DIM//2,-1), DIM )

    y = tf.tile( tf.range(-DIM//2,DIM//2),[DIM] )

    z = tf.ones([DIM*DIM],dtype='int32')

    idx = tf.stack( [x,y,z] )

    

    # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS

    idx2 = K.dot(m,tf.cast(idx,dtype='float32'))

    idx2 = K.cast(idx2,dtype='int32')

    idx2 = K.clip(idx2,-DIM//2+XDIM+1,DIM//2)

    

    # FIND ORIGIN PIXEL VALUES           

    idx3 = tf.stack( [DIM//2-idx2[0,], DIM//2-1+idx2[1,]] )

    d = tf.gather_nd(image,tf.transpose(idx3))

        

    return tf.reshape(d,[DIM,DIM,3]),label
# 在batch内部互相随机取图

def cutmix(image, label, PROBABILITY = cutmix_rate):

    # input image - is a batch of images of size [n,dim,dim,3] not a single image of [dim,dim,3]

    # output - a batch of images with cutmix applied

    label=tf.cast(label,tf.float32)

    

    DIM = img_size    

    imgs = []; labs = []

    

    for j in range(BATCH_SIZE):

        

        #random_uniform( shape, minval=0, maxval=None)        

        # DO CUTMIX WITH PROBABILITY DEFINED ABOVE

        P = tf.cast(tf.random.uniform([], 0, 1) <= PROBABILITY, tf.int32)

        

        # CHOOSE RANDOM IMAGE TO CUTMIX WITH

        k = tf.cast(tf.random.uniform([], 0, BATCH_SIZE), tf.int32)

        

        # CHOOSE RANDOM LOCATION

        #选一个随机的中心点

        x = tf.cast(tf.random.uniform([], 0, DIM), tf.int32)

        y = tf.cast(tf.random.uniform([], 0, DIM), tf.int32)

        

        # Beta(1, 1)等于均匀分布

        b = tf.random.uniform([], 0, 1) # this is beta dist with alpha=1.0

        

        #P只随机出0或1，就是裁剪或是不裁剪

        WIDTH = tf.cast(DIM * tf.math.sqrt(1-b),tf.int32) * P

        ya = tf.math.maximum(0,y-WIDTH//2)

        yb = tf.math.minimum(DIM,y+WIDTH//2)

        xa = tf.math.maximum(0,x-WIDTH//2)

        xb = tf.math.minimum(DIM,x+WIDTH//2)

        

        # MAKE CUTMIX IMAGE

        one = image[j,ya:yb,0:xa,:]

        two = image[k,ya:yb,xa:xb,:]

        three = image[j,ya:yb,xb:DIM,:]        

        #得出了ya:yb区间内的输出图像

        middle = tf.concat([one,two,three],axis=1)

        #得到了完整输出图像

        img = tf.concat([image[j,0:ya,:,:],middle,image[j,yb:DIM,:,:]],axis=0)

        imgs.append(img)

        

        # MAKE CUTMIX LABEL

        #按面积来加权的

        a = tf.cast(WIDTH*WIDTH/DIM/DIM,tf.float32)

        lab1 = label[j,]

        lab2 = label[k,]

        labs.append((1-a)*lab1 + a*lab2)



    image2 = tf.reshape(tf.stack(imgs),(BATCH_SIZE,DIM,DIM,3))

    label2 = tf.reshape(tf.stack(labs),(BATCH_SIZE, nb_classes))

    return image2,label2
def mixup(image, label, PROBABILITY = mixup_rate):

    # input image - is a batch of images of size [n,dim,dim,3] not a single image of [dim,dim,3]

    # output - a batch of images with mixup applied

    DIM = img_size

    

    imgs = []; labs = []

    for j in range(BATCH_SIZE):

        

        # CHOOSE RANDOM

        k = tf.cast( tf.random.uniform([],0,BATCH_SIZE),tf.int32)

        a = tf.random.uniform([],0,1) # this is beta dist with alpha=1.0



        #根据概率抽取执不执行mixup

        P = tf.cast(tf.random.uniform([], 0, 1) <= PROBABILITY, tf.int32)

        if P==1:

            a=0.

        

        # MAKE MIXUP IMAGE

        img1 = image[j,]

        img2 = image[k,]

        imgs.append((1-a)*img1 + a*img2)

        

        # MAKE CUTMIX LABEL

        lab1 = label[j,]

        lab2 = label[k,]

        labs.append((1-a)*lab1 + a*lab2)

            

    # RESHAPE HACK SO TPU COMPILER KNOWS SHAPE OF OUTPUT TENSOR (maybe use Python typing instead?)

    image2 = tf.reshape(tf.stack(imgs),(BATCH_SIZE,DIM,DIM,3))

    label2 = tf.reshape(tf.stack(labs),(BATCH_SIZE,nb_classes))

    return image2,label2
# gridmask

def transform(image, inv_mat, image_shape):

    h, w, c = image_shape

    cx, cy = w//2, h//2

    new_xs = tf.repeat( tf.range(-cx, cx, 1), h)

    new_ys = tf.tile( tf.range(-cy, cy, 1), [w])

    new_zs = tf.ones([h*w], dtype=tf.int32)

    old_coords = tf.matmul(inv_mat, tf.cast(tf.stack([new_xs, new_ys, new_zs]), tf.float32))

    old_coords_x, old_coords_y = tf.round(old_coords[0, :] + w//2), tf.round(old_coords[1, :] + h//2)

    clip_mask_x = tf.logical_or(old_coords_x<0, old_coords_x>w-1)

    clip_mask_y = tf.logical_or(old_coords_y<0, old_coords_y>h-1)

    clip_mask = tf.logical_or(clip_mask_x, clip_mask_y)

    old_coords_x = tf.boolean_mask(old_coords_x, tf.logical_not(clip_mask))

    old_coords_y = tf.boolean_mask(old_coords_y, tf.logical_not(clip_mask))

    new_coords_x = tf.boolean_mask(new_xs+cx, tf.logical_not(clip_mask))

    new_coords_y = tf.boolean_mask(new_ys+cy, tf.logical_not(clip_mask))

    old_coords = tf.cast(tf.stack([old_coords_y, old_coords_x]), tf.int32)

    new_coords = tf.cast(tf.stack([new_coords_y, new_coords_x]), tf.int64)

    rotated_image_values = tf.gather_nd(image, tf.transpose(old_coords))

    rotated_image_channel = list()

    for i in range(c):

        vals = rotated_image_values[:,i]

        sparse_channel = tf.SparseTensor(tf.transpose(new_coords), vals, [h, w])

        rotated_image_channel.append(tf.sparse.to_dense(sparse_channel, default_value=0, validate_indices=False))

    return tf.transpose(tf.stack(rotated_image_channel), [1,2,0])



def random_rotate(image, angle, image_shape):

    def get_rotation_mat_inv(angle):

        # transform to radian

        angle = math.pi * angle / 180

        cos_val = tf.math.cos(angle)

        sin_val = tf.math.sin(angle)

        one = tf.constant([1], tf.float32)

        zero = tf.constant([0], tf.float32)

        rot_mat_inv = tf.concat([cos_val, sin_val, zero, -sin_val, cos_val, zero, zero, zero, one], axis=0)

        rot_mat_inv = tf.reshape(rot_mat_inv, [3,3])

        return rot_mat_inv

    angle = float(angle) * tf.random.normal([1],dtype='float32')

    rot_mat_inv = get_rotation_mat_inv(angle)

    return transform(image, rot_mat_inv, image_shape)





def GridMask(image_height, image_width, d1, d2, rotate_angle=1, ratio=0.5):

    h, w = image_height, image_width

    hh = int(np.ceil(np.sqrt(h*h+w*w)))

    hh = hh+1 if hh%2==1 else hh

    d = tf.random.uniform(shape=[], minval=d1, maxval=d2, dtype=tf.int32)

    l = tf.cast(tf.cast(d,tf.float32)*ratio+0.5, tf.int32)



    st_h = tf.random.uniform(shape=[], minval=0, maxval=d, dtype=tf.int32)

    st_w = tf.random.uniform(shape=[], minval=0, maxval=d, dtype=tf.int32)



    y_ranges = tf.range(-1 * d + st_h, -1 * d + st_h + l)

    x_ranges = tf.range(-1 * d + st_w, -1 * d + st_w + l)



    for i in range(0, hh//d+1):

        s1 = i * d + st_h

        s2 = i * d + st_w

        y_ranges = tf.concat([y_ranges, tf.range(s1,s1+l)], axis=0)

        x_ranges = tf.concat([x_ranges, tf.range(s2,s2+l)], axis=0)



    x_clip_mask = tf.logical_or(x_ranges < 0 , x_ranges > hh-1)

    y_clip_mask = tf.logical_or(y_ranges < 0 , y_ranges > hh-1)

    clip_mask = tf.logical_or(x_clip_mask, y_clip_mask)



    x_ranges = tf.boolean_mask(x_ranges, tf.logical_not(clip_mask))

    y_ranges = tf.boolean_mask(y_ranges, tf.logical_not(clip_mask))



    hh_ranges = tf.tile(tf.range(0,hh), [tf.cast(tf.reduce_sum(tf.ones_like(x_ranges)), tf.int32)])

    x_ranges = tf.repeat(x_ranges, hh)

    y_ranges = tf.repeat(y_ranges, hh)



    y_hh_indices = tf.transpose(tf.stack([y_ranges, hh_ranges]))

    x_hh_indices = tf.transpose(tf.stack([hh_ranges, x_ranges]))



    y_mask_sparse = tf.SparseTensor(tf.cast(y_hh_indices, tf.int64),  tf.zeros_like(y_ranges), [hh, hh])

    y_mask = tf.sparse.to_dense(y_mask_sparse, 1, False)



    x_mask_sparse = tf.SparseTensor(tf.cast(x_hh_indices, tf.int64), tf.zeros_like(x_ranges), [hh, hh])

    x_mask = tf.sparse.to_dense(x_mask_sparse, 1, False)



    mask = tf.expand_dims( tf.clip_by_value(x_mask + y_mask, 0, 1), axis=-1)



    mask = random_rotate(mask, rotate_angle, [hh, hh, 1])

    mask = tf.image.crop_to_bounding_box(mask, (hh-h)//2, (hh-w)//2, image_height, image_width)



    return mask



def apply_grid_mask(image, image_shape, PROBABILITY = gridmask_rate):

    AugParams = {

        'd1' : 100,

        'd2': 160,

        'rotate' : 45,

        'ratio' : 0.3

    }

    

        

    mask = GridMask(image_shape[0], image_shape[1], AugParams['d1'], AugParams['d2'], AugParams['rotate'], AugParams['ratio'])

    if image_shape[-1] == 3:

        mask = tf.concat([mask, mask, mask], axis=-1)

        mask = tf.cast(mask,tf.float32)

        #print(mask.shape) # (299,299,3)



# 会报错，放弃

#     imgs = []

#     BATCH_SIZE=len(image)

#     for j in range(BATCH_SIZE):

#         P = tf.cast(tf.random.uniform([], 0, 1) <= PROBABILITY, tf.int32)

#         if P==1:

#             imgs.append(image[j,]*mask)

#         else:

#             imgs.append(image[j,])

#     return tf.cast(imgs,tf.float32)



        

    # 整个batch启用或者不启用

    P = tf.cast(tf.random.uniform([], 0, 1) <= PROBABILITY, tf.int32)

    if P==1:

        return image*mask

    else:

        return image



def gridmask(img_batch, label_batch):

    return apply_grid_mask(img_batch, (img_size,img_size, 3)), label_batch
def get_train_dataset(train_paths,train_labels=None):



    # num_parallel_calls并发处理数据的并发数

    #train_dataset = tf.data.Dataset.from_tensor_slices((train_paths, train_labels.astype(np.float32))).map(decode_image, num_parallel_calls=AUTO)



    #特殊情况，这样更改下

    train_dataset = load_dataset(train_paths, labeled=True)    

    train_dataset = train_dataset.cache().map(data_aug, num_parallel_calls=AUTO).repeat()

    

    if bool_rotation_transform:

        train_dataset =train_dataset.map(rotation_transform)

                     

    train_dataset = train_dataset.shuffle(512).batch(BATCH_SIZE,drop_remainder=True)





    if cutmix_rate:  

        print('启用cutmix')

        train_dataset =train_dataset.map(cutmix, num_parallel_calls=AUTO)

    if mixup_rate:  

        print('启用mixup')

        train_dataset =train_dataset.map(mixup, num_parallel_calls=AUTO)

    if gridmask_rate:

        print('启用gridmask')

        train_dataset =train_dataset.map(gridmask, num_parallel_calls=AUTO)

    if (cutmix_rate or mixup_rate):

        train_dataset =train_dataset.unbatch().shuffle(512).batch(BATCH_SIZE)





    # repeat()代表无限制复制原始数据，这里可以用count指明复制份数，但要注意要比fit中的epochs大才可

    # 直接调用repeat()的话，生成的序列就会无限重复下去

    # prefetch: prefetch next batch while training (autotune prefetch buffer size)

    train_dataset = train_dataset.prefetch(AUTO)



    return train_dataset
try:

    view_train_dataset=get_train_dataset(train_paths,train_labels)

except:

    view_train_dataset=get_train_dataset(train_paths)

    

it = view_train_dataset.__iter__()



#看看train_dataset 是否正常显示

show_x, show_y = it.next()

print(show_x.shape,'\n\n',show_y[0])



plt.figure(figsize=(12, 6))

for i in range(8):

    plt.subplot(2, 4, i+1)

    plt.imshow(show_x[i])
def get_validation_dataset(valid_paths,valid_labels=None):

    #dataset = tf.data.Dataset.from_tensor_slices((valid_paths, valid_labels))

    #dataset = dataset.map(decode_image, num_parallel_calls=AUTO)

    

    #特殊情况，这样更改下

    dataset = load_dataset(valid_paths, labeled=True, ordered=True)

    

    



    dataset = dataset.cache()

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset
#生成测试集

def re_produce_test_dataset(test_paths):

    #test_dataset = tf.data.Dataset.from_tensor_slices(test_paths)

    #test_dataset = test_dataset.map(decode_image, num_parallel_calls=AUTO)

    

    #特殊情况，这样更改下

    test_dataset = load_dataset(test_paths, labeled=False, ordered=True)

    



    if bool_tta:

        test_dataset = test_dataset.cache().map(data_aug, num_parallel_calls=AUTO)



    test_dataset = test_dataset.batch(BATCH_SIZE)

    return test_dataset
#tta时可重复运行这块观察是否多次运行时生成了不同的测试数据

view_dataset = re_produce_test_dataset(test_paths[:8])

it = view_dataset.__iter__()

show_x= it.next()

try:

    print(show_x.shape)

except:

    print(show_x[0].shape)

    show_x=show_x[0]

    

plt.figure(figsize=(12, 6))

for i in range(8):

    plt.subplot(2, 4, i+1)

    plt.imshow(show_x[i])
# if special_monitor=='auc':

#     my_metrics.append(tf.keras.metrics.AUC(name='auc'))





#创建模型

def get_model():

    with strategy.scope():

        base_model =  efn.EfficientNetB0(weights=pre_trained, include_top=False, pooling='avg', input_shape=(img_size, img_size, 3))

        x = base_model.output

        predictions = Dense(nb_classes, activation=dense_activation)(x)

        model = Model(inputs=base_model.input, outputs=predictions)



    if label_smoothing_rate:

        print('启用label_smoothing')

        my_loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing_rate)

    elif bool_focal_loss:



        my_loss = tfa.losses.SigmoidFocalCrossEntropy(reduction=tf.keras.losses.Reduction.AUTO)



    else:

        my_loss='categorical_crossentropy'



    model.compile(optimizer=tf.keras.optimizers.Adam(lr_if_without_scheduler), 

                  loss=my_loss,

                  metrics=my_metrics

                 )



    return model
callbacks=[]

if bool_lr_scheduler:

    callbacks.append(lr_callback)
if cross_validation_folds:

    probabilities =[]

    histories = []



    # early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 3)

    kfold = KFold(cross_validation_folds , shuffle = True)

    

    i=1

    

    #特殊处理

    for trn_ind, val_ind in kfold.split(train_paths):

        # print(trn_ind)

        # print(val_ind)

    

    #for trn_ind, val_ind in kfold.split(train_paths,train_labels):

        print(); print('#'*25)

        print('### FOLD',i)

        print('#'*25)

        

        #每轮都应该重置 ModelCheckpoint

        ch_p1 = ModelCheckpoint(filepath="temp_best.h5", monitor='val_loss', save_weights_only=True,verbose=1,save_best_only=True)



#该功能暂时作废

#         if special_monitor=='auc': 

#             ch_p1 = ModelCheckpoint(filepath="temp_best.h5", monitor='val_auc', mode='max',save_weights_only=True,verbose=1,save_best_only=True)



        temp_callbacks=callbacks.copy()

        temp_callbacks.append(ch_p1)

        

        trn_paths = np.array(train_paths)[trn_ind]

        val_paths=np.array(train_paths)[val_ind]



        #特殊处理

#         trn_labels = train_labels[trn_ind]

#         val_labels=train_labels[val_ind]

        trn_labels=None

        val_labels=None

        

        model = get_model()

        history = model.fit(

            get_train_dataset(trn_paths,trn_labels), 

            #steps_per_epoch = trn_labels.shape[0]//BATCH_SIZE,

            

            #特殊处理下

            steps_per_epoch = 16465//BATCH_SIZE,

            

            epochs = EPOCHS,

            callbacks = temp_callbacks,

            validation_data = (get_validation_dataset(val_paths,val_labels)),

        )

        

        i+=1

        histories.append(history)

        

        #用val_loss最小的权重来预测

        model.load_weights("temp_best.h5")

        #prob = model.predict(re_produce_test_dataset(test_paths), verbose=1)

        

        #特殊处理

        prob = model.predict(re_produce_test_dataset(test_paths).map(lambda image, idnum: image), verbose=1)

        

        probabilities.append(prob)
#画history咯

def display_training_curves(training, title, subplot, validation=None):

    if subplot%10==1: # set up the subplots on the first call

        plt.subplots(figsize=(10,10), facecolor='#F0F0F0')

        plt.tight_layout()

    ax = plt.subplot(subplot)

    ax.set_facecolor('#F8F8F8')

    ax.plot(training)

    if validation is not None:

        ax.plot(validation)

    ax.set_title('model '+ title)

    ax.set_ylabel(title)

    #ax.set_ylim(0.28,1.05)

    ax.set_xlabel('epoch')

    if validation is not None:

        ax.legend(['train', 'valid.'])

    else:

        ax.legend(['train'])
# 写开，防止交叉验证时out of memory导致什么都没保存

if cross_validation_folds:

    y_pred = np.mean(probabilities,axis =0)

    

    

#然后画画- -

    for h in range(len(histories)):

        display_training_curves(histories[h].history['loss'], 'loss', 211, histories[h].history['val_loss'])

        display_training_curves(histories[h].history['accuracy'], 'accuracy', 212, histories[h].history['val_accuracy'])
if not cross_validation_folds:

    

    model = get_model()

    

    #特殊处理

    train_labels=None

    

    history = model.fit(

        get_train_dataset(train_paths,train_labels), 

        #steps_per_epoch=train_labels.shape[0] // BATCH_SIZE,

        

        #特殊处理下

        steps_per_epoch = 16465//BATCH_SIZE,

        

        

        callbacks=callbacks,

        epochs=EPOCHS

    )





    if bool_tta:

        probabilities = []

        for i in range(tta_times+1):

            print('TTA Number: ',i,'\n')

            test_dataset = re_produce_test_dataset(test_paths)

            #probabilities.append(model.predict(test_dataset))

            

            #特殊处理

            probabilities.append(model.predict(re_produce_test_dataset(test_paths).map(lambda image, idnum: image), verbose=1))

        y_pred = np.mean(probabilities,axis =0)



    else:

        #test_dataset = re_produce_test_dataset(test_paths)

        #y_pred = model.predict(test_dataset)

        

        #特殊处理

        y_pred = model.predict(re_produce_test_dataset(test_paths).map(lambda image, idnum: image), verbose=1)

        

if not cross_validation_folds:

    display_training_curves(history.history['loss'], 'loss', 211)

    display_training_curves(history.history['accuracy'], 'accuracy', 212)
#针对不同数据不同后处理

predictions = np.argmax(y_pred, axis=-1)

print(predictions)



print('Generating submission.csv file...')

test_ids_ds = re_produce_test_dataset(test_paths).map(lambda image, idnum: idnum).unbatch()

test_ids = next(iter(test_ids_ds.batch(7382))).numpy().astype('U') # all in one batch

np.savetxt('submission.csv', np.rec.fromarrays([test_ids, predictions]), fmt=['%s', '%d'], delimiter=',', header='id,label', comments='')

!head submission.csv