import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import KFold

import numpy as np, pandas as pd, os

import matplotlib.pyplot as plt, cv2

import tensorflow as tf

import random, re, math, os

from glob import glob

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



def serialize_example(feature0, feature2):

  feature = {

      'image': _bytes_feature(feature0),

      #'path': _bytes_feature(feature1),

      'label': _int64_feature(feature2)

  }

  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))

  return example_proto.SerializeToString()
%%time





def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)



SEED=37

seed_everything(SEED)



dataset = []



for label, kind in enumerate(['Cover', 'JMiPOD', 'JUNIWARD', 'UERD']):

    for path in glob('../input/alaska2-image-steganalysis/Cover/*.jpg'):

        dataset.append({

            'kind': kind,

            'image_name': path.split('/')[-1],

            'label': label

        })



random.shuffle(dataset)

dataset = pd.DataFrame(dataset)

gkf = KFold(n_splits=10)



dataset.loc[:, 'fold'] = 0

for fold_number, (train_index, val_index) in enumerate(gkf.split(X=dataset.index, y=dataset['label'], groups=dataset['image_name'])):

    dataset.loc[dataset.iloc[val_index].index, 'fold'] = fold_number



dataset.sort_values("image_name",inplace=True)

dataset.head()
fold_number = 0

train_df = dataset[dataset['fold'] != fold_number]

val_df = dataset[dataset['fold'] == fold_number]
%%time



#GCS_DS_PATH = KaggleDatasets().get_gcs_path() 

GCS_DS_PATH = '../input/alaska2-image-steganalysis'

train_paths = []

train_labels = []



for i in range(len(train_df['kind'])):

    kind = train_df['kind'].iloc[i]

    im_id = train_df['image_name'].iloc[i]

    label = train_df['label'].iloc[i]

    path = os.path.join(GCS_DS_PATH, kind, im_id)

    

    train_paths.append(path)

    train_labels.append(label)

    

len(train_paths), len(train_labels)

print(train_paths[0:5])

print(train_labels[0:5])
%%time



valid_paths = []

valid_labels = []



for i in range(len(val_df['kind'])):

    kind = val_df['kind'].iloc[i]

    im_id = val_df['image_name'].iloc[i]

    label = val_df['label'].iloc[i]

    path = os.path.join(GCS_DS_PATH, kind, im_id)

    

    valid_paths.append(path)

    valid_labels.append(label)

    

len(valid_paths), len(valid_labels)
# SIZE = 10000

# CT = len(train_paths)//SIZE + int(len(train_paths)%SIZE!=0)

# for j in range(CT):

#     print('Writing TFRecord %i of %i...'%(j,CT))

#     CT2 = min(SIZE,len(train_paths)-j*SIZE)

#     with tf.io.TFRecordWriter('train%.2i-%i.tfrec'%(j,CT2)) as writer:

#         for k in range(CT2):

#             path=train_paths[j*SIZE+k]

#             img = open(path, 'rb').read()

#             label=train_labels[j*SIZE+k]

#             example = serialize_example(img,label)#,str.encode(path)                

#             writer.write(example)

#             if k%1000==0: print(k,', ',end='')



#             img = cv2.imread(train_paths[j*SIZE+k])

#             img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # Fix incorrect colors

#             img = cv2.imencode('.jpg', img, (cv2.IMWRITE_JPEG_QUALITY, 94))[1].tostring()
SIZE = 10000

CT = len(valid_paths)//SIZE + int(len(valid_paths)%SIZE!=0)

for j in range(CT):

    print('Writing TFRecord %i of %i...'%(j,CT))

    CT2 = min(SIZE,len(valid_paths)-j*SIZE)

    with tf.io.TFRecordWriter('valid%.2i-%i.tfrec'%(j,CT2)) as writer:

        for k in range(CT2):

            path=valid_paths[j*SIZE+k]

            img = open(path, 'rb').read()

#             img = cv2.imread(valid_paths[j*SIZE+k])

#             #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # Fix incorrect colors

#             img = cv2.imencode('.jpg', img, (cv2.IMWRITE_JPEG_QUALITY, 94))[1].tostring()

            label=valid_labels[j*SIZE+k]

            example = serialize_example(img,label)#,str.encode(path)                

            writer.write(example)

            if k%1000==0: print(k,', ',end='')