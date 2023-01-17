import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import glob
import tensorflow as tf
import re, math
import matplotlib.pyplot as plt
os.system('cp ../input/gdcm-conda-install/gdcm.tar .')
os.system('tar -xvzf gdcm.tar')
os.system('conda install --offline ./gdcm/gdcm-2.8.9-py37h71b2a6d_0.tar.bz2')
print("GDCM Loaded!")
import pydicom
import cv2
os.system('rm gdcm.tar')
os.system('rm -r gdcm')
train = pd.read_csv("../input/rsna-str-pulmonary-embolism-detection/train.csv")
# PATHS TO IMAGES
PATH_TRAIN = '../input/rsna-str-pulmonary-embolism-detection/train'
PATH_TEST = '../input/rsna-str-pulmonary-embolism-detection/test'
case_train = glob.glob(PATH_TRAIN + '/*/*')
print('Total number of train cases: ', len(case_train))
case_test = glob.glob(PATH_TEST + '/*/*')
print('Total number of test cases: ', len(case_test))
def transform_to_hu(medical_image, image):
    intercept = medical_image.RescaleIntercept
    slope = medical_image.RescaleSlope
    hu_image = image * slope + intercept
    return hu_image

def window(img, WL=50, WW=350):
    upper, lower = WL+WW//2, WL-WW//2
    X = np.clip(img.copy(), lower, upper)
    X = X - np.min(X)
    X = X / np.max(X)
    X = (X*255.0).astype('uint8')
    return X

def convert_to_rgb(array):
    shape_gray = array.shape
    R_lung_window = window(array, -600, 1500)
    G_pe_window = window(array, 100, 700)
    B_mediastinal_window = window(array, 40, 400)
    return np.stack([R_lung_window, G_pe_window, B_mediastinal_window], axis=2).reshape(shape_gray + (3,))
# def get_imgs_by_case(path):
#     img_path = glob.glob(path + '/*')
    
#     img_set = []
#     z_set = []
#     sop_set = []
    
#     for p in img_path:
#         med_img = pydicom.dcmread(p)
#         img = med_img.pixel_array
#         img = transform_to_hu(med_img, img)
#         img = convert_to_rgb(img)
        
#         img_set = np.append([img_set, img])
#         z_set = np.append([z_set, float(med_img.ImagePositionPatient[-1])])
#         sop_set = np.append([sop_set, p.split('/')[-1].split('.')[0]])
    
#     return img_set, z_set, sop_set
def get_img(path):    
    med_img = pydicom.dcmread(path)
    img = med_img.pixel_array
    img = transform_to_hu(med_img, img)
    img = convert_to_rgb(img)

    pos_z = float(med_img.ImagePositionPatient[-1])
    sop = path.split('/')[-1].split('.')[0]
    
    return img, pos_z, sop
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def serialize_example(feature0, feature1, feature2, feature3):
    feature = {
      'image': _bytes_feature(feature0),
      'image_id': _bytes_feature(feature1),
      'position_z': _float_feature(feature2),
      'target': _bytes_feature(feature3)
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()
selected_cols = ['pe_present_on_image',
                 'negative_exam_for_pe', 'rv_lv_ratio_gte_1', 'rv_lv_ratio_lt_1',
                 'leftsided_pe', 'chronic_pe', 'rightsided_pe',
                 'acute_and_chronic_pe', 'central_pe', 'indeterminate']
BATCH = 51
SIZE = len(case_train)//(BATCH-1)
SIZE
PATH_SET = [[] for i in range(BATCH)]
m = 0
for n, p in enumerate(case_train):
    img_path = glob.glob(p + '/*')
    if (n+1) % SIZE == 0:
        m += 1
    PATH_SET[m] = np.append(PATH_SET[m], img_path)

PATH_SET = np.array(PATH_SET)
print(PATH_SET.shape, PATH_SET[0].shape)
for j in range(BATCH):
    print(); print('Writing TFRecord %i of %i...'%(j,BATCH-1))
    with tf.io.TFRecordWriter('train%.2i_%i.tfrec'%(j, PATH_SET[j].shape[0])) as writer:
        for k in range(PATH_SET[j].shape[0]):
            img, pos_z, name = get_img(PATH_SET[j][k])
            img = cv2.imencode('.jpg', img, (cv2.IMWRITE_JPEG_QUALITY, 94))[1].tostring()
            target = train[train['SOPInstanceUID'] == name][selected_cols].values[0]
            example = serialize_example(
                img, str.encode(name), pos_z, 
                tf.io.serialize_tensor(np.array(target, dtype=np.uint8))
            )
            writer.write(example)
            if k%1000==0: print(k,', ',end='')
np.set_printoptions(threshold=15, linewidth=80)
CLASSES = [0,1]

def batch_to_numpy_images_and_labels(data):
    images, labels = data
    numpy_images = images.numpy()
    numpy_labels = labels.numpy()
    #if numpy_labels.dtype == object: # binary string in this case, these are image ID strings
    #    numpy_labels = [None for _ in enumerate(numpy_images)]
    # If no labels, only image IDs, return None for labels (this is the case for test data)
    return numpy_images, numpy_labels

def title_from_label_and_target(label, correct_label):
    if correct_label is None:
        return CLASSES[label], True
    correct = (label == correct_label)
    return "{} [{}{}{}]".format(CLASSES[label], 'OK' if correct else 'NO', u"\u2192" if not correct else '',
                                CLASSES[correct_label] if not correct else ''), correct

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
    display_batch_of_images(images, predictions)
    display_batch_of_images((images, labels))
    display_batch_of_images((images, labels), predictions)
    """
    # data
    images, labels = batch_to_numpy_images_and_labels(databatch)
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
        title = label
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
def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range
    image = tf.reshape(image, [*IMAGE_SIZE, 3]) # explicit size needed for TPU
    return image

def read_labeled_tfrecord(example):
    LABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
        "image_id": tf.io.FixedLenFeature([], tf.string),  # shape [] means single element
    }
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    image = decode_image(example['image'])
    label = example['image_id']
    return image, label # returns a dataset of (image, label) pairs

def load_dataset(filenames, labeled=True, ordered=False):
    # Read from TFRecords. For optimal performance, reading from multiple files at once and
    # disregarding data order. Order does not matter since we will be shuffling the data anyway.

    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False # disable order, increase speed

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO) # automatically interleaves reads from multiple files
    dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(read_labeled_tfrecord)
    # returns a dataset of (image, label) pairs if labeled=True or (image, id) pairs if labeled=False
    return dataset

def get_training_dataset():
    dataset = load_dataset(TRAINING_FILENAMES, labeled=True)
    dataset = dataset.repeat() # the training dataset must repeat for several epochs
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def count_data_items(filenames):
    # the number of data items is written in the name of the .tfrec files, i.e. flowers00-230.tfrec = 230 data items
    n = [int(re.compile(r"_([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)
IMAGE_SIZE= [512,512]; BATCH_SIZE = 32
AUTO = tf.data.experimental.AUTOTUNE
TRAINING_FILENAMES = tf.io.gfile.glob('train*.tfrec')
print('There are %i train images'%count_data_items(TRAINING_FILENAMES))
training_dataset = get_training_dataset()
training_dataset = training_dataset.unbatch().batch(20)
train_batch = iter(training_dataset)

display_batch_of_images(next(train_batch))
