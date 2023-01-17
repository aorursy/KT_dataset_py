import os

import random, re, math, gc

from collections import Counter, defaultdict



from PIL import Image

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import cv2



import tensorflow as tf, tensorflow.keras.backend as K

from kaggle_datasets import KaggleDatasets

AUTO = tf.data.experimental.AUTOTUNE



print(tf.__version__)

print(tf.keras.__version__)
def seed_everything(seed=1234):

    np.random.seed(seed)

    tf.random.set_seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
image_shape = (1024,1024,3)

image_quality = 90

SEED = 1234

seed_everything(SEED)

BATCH_SIZE = 16 



train_fold = 5

test_fold = 6





BASE_PATH = '../input/siim-isic-melanoma-classification'

train_metadata = pd.read_csv(os.path.join(BASE_PATH, 'train.csv'))

test_metadata = pd.read_csv(os.path.join(BASE_PATH, 'test.csv'))

sample_submission = pd.read_csv(os.path.join(BASE_PATH, 'sample_submission.csv'))
train_metadata.tail()
test_metadata.tail()
print('Missing Values in Train_metadata in(%): \n')

missing_values = train_metadata.isnull().sum() / len(train_metadata)

missing_values = missing_values[missing_values>0.0]

print(missing_values)



print('\n\nMissing Values in Test_metadata : \n')

missing_values = test_metadata.isnull().sum() / len(test_metadata)

missing_values = missing_values[missing_values>0.0]

print(missing_values)
print('Unique values in column with frequency : ')



print('\nsex : ', dict(train_metadata.sex.value_counts()))

print('\nage_approx : ', dict(train_metadata.age_approx.value_counts()))

print('\nanatom_site_general_challenge : ', dict(train_metadata.anatom_site_general_challenge.value_counts()))

print('\ndiagnosis : ', dict(train_metadata.diagnosis.value_counts()))

print('\nbenign_malignant : ', dict(train_metadata.benign_malignant.value_counts()))

print('\ntarget : ', dict(train_metadata.target.value_counts()))
def get_stratify_group(row):

    return '{}_{}_{}'.format(row['anatom_site_general_challenge'],row['sex'],row['target'])
train = train_metadata.copy()



train_cat_list_patient_id = train['patient_id'].astype('category').cat.categories

train['patient_id'] = train.patient_id.astype('category').cat.codes



train['sex'] = train['sex'].fillna('unknown')

# cat_list_sex = train['sex'].astype('category').cat.categories

# train['sex'] = train.sex.astype('category').cat.codes



train['age_approx'] = train['age_approx'].fillna(train.age_approx.mean())

train['age_approx'] = train['age_approx'].astype('int')



train['anatom_site_general_challenge'] = train['anatom_site_general_challenge'].fillna("unknown")

# cat_list_anatom_site_general_challenge = train['anatom_site_general_challenge'].astype('category').cat.categories

# train['anatom_site_general_challenge'] = train['anatom_site_general_challenge'].astype('category').cat.codes



train['stratify_group'] = train.fillna("NA").apply(get_stratify_group, axis=1)

train['stratify_group'] = train['stratify_group'].astype('category').cat.codes
test = test_metadata.copy()



test_cat_list_patient_id = test['patient_id'].astype('category').cat.categories

test['patient_id'] = test.patient_id.astype('category').cat.codes



test['anatom_site_general_challenge'] = test['anatom_site_general_challenge'].fillna("unknown")

# test_cat_list_anatom_site_general_challenge = test['anatom_site_general_challenge'].astype('category').cat.categories

# test['anatom_site_general_challenge'] = test['anatom_site_general_challenge'].astype('category').cat.codes



test['target'] = 2



test['stratify_group'] = test.fillna("NA").apply(get_stratify_group, axis=1)

test['stratify_group'] = test['stratify_group'].astype('category').cat.codes
def stratified_group_k_fold(X, y, groups, k, seed=None):

    labels_num = np.max(y) + 1

    y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))

    y_distr = Counter()

    for label, g in zip(y, groups):

        y_counts_per_group[g][label] += 1

        y_distr[label] += 1



    y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))

    groups_per_fold = defaultdict(set)



    def eval_y_counts_per_fold(y_counts, fold):

        y_counts_per_fold[fold] += y_counts

        std_per_label = []

        for label in range(labels_num):

            label_std = np.std([y_counts_per_fold[i][label] / y_distr[label] for i in range(k)])

            std_per_label.append(label_std)

        y_counts_per_fold[fold] -= y_counts

        return np.mean(std_per_label)

    

    groups_and_y_counts = list(y_counts_per_group.items())

    random.Random(seed).shuffle(groups_and_y_counts)



    for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):

        best_fold = None

        min_eval = None

        for i in range(k):

            fold_eval = eval_y_counts_per_fold(y_counts, i)

            if min_eval is None or fold_eval < min_eval:

                min_eval = fold_eval

                best_fold = i

        y_counts_per_fold[best_fold] += y_counts

        groups_per_fold[best_fold].add(g)



    all_groups = set(groups)

    

    for i in range(k):

        train_groups = all_groups - groups_per_fold[i]

        test_groups = groups_per_fold[i]



        train_indices = [i for i, g in enumerate(groups) if g in train_groups]

        test_indices = [i for i, g in enumerate(groups) if g in test_groups]



        yield train_indices, test_indices
%%time



train['fold'] = 0





for fold_ind, (train_ind, val_ind) in enumerate(stratified_group_k_fold(train_metadata, 

                                                                        train.stratify_group.values, 

                                                                        train_metadata.patient_id.values, 

                                                                        k=train_fold, 

                                                                        seed=SEED)):

    train.loc[val_ind,'fold'] = fold_ind



train.fold.value_counts()
sex_code = pd.get_dummies(train.sex, prefix='sex')

anatom_site_general_challenge_code = pd.get_dummies(train.anatom_site_general_challenge, prefix='anatom_site')

age_aprox_normalized = (train.age_approx-train.age_approx.mean())/train.age_approx.std()

train_coded = pd.concat([train.image_name, sex_code, age_aprox_normalized ,anatom_site_general_challenge_code, train.target, train.fold], axis=1)
train_coded.head()
print(len(train_coded.columns))

train_coded.columns
%%time



test['fold'] = 0



for fold_ind, (test_ind, val_ind) in enumerate(stratified_group_k_fold(test_metadata, 

                                                                       test.stratify_group.values, 

                                                                       test_metadata.patient_id.values, 

                                                                       k=test_fold, 

                                                                       seed=SEED)):

    test.loc[val_ind,'fold'] = fold_ind



test.fold.value_counts()
sex_code = pd.get_dummies(test.sex, prefix='sex')

anatom_site_general_challenge_code = pd.get_dummies(test.anatom_site_general_challenge, prefix='anatom_site')

age_aprox_normalized = (test.age_approx-test.age_approx.mean())/test.age_approx.std()

test_coded = pd.concat([test.image_name, sex_code, age_aprox_normalized ,anatom_site_general_challenge_code, test.fold], axis=1)

test_coded['sex_unknown'] = 0
test_coded.head()
print(len(test_coded.columns))

test_coded.columns
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
def get_path(image_name, _set='train'):

    if _set == 'train':    

        return BASE_PATH + '/jpeg/train/' + str(image_name) + '.jpg'

    else:

        return BASE_PATH + '/jpeg/test/' + str(image_name) + '.jpg'
def row_serialize(row, image_shape, _set='train'):

    

    image_path = get_path(row['image_name'],_set=_set)

    

    ## Method -1

    

#     raw_image = tf.keras.preprocessing.image.load_img(image_path,

#                                                       color_mode='rgb',

#                                                       target_size=image_shape

#                                                      ).tobytes()



    ## Method-2 (with 10% compression)

    

    image = tf.io.read_file(image_path)

    image = tf.image.decode_jpeg(image, channels=3)

    image = tf.image.resize(image, image_shape[:2], method="nearest")

    raw_image = tf.image.encode_jpeg(image, quality=image_quality, optimize_size=True)

    del image

    

    ## Method-3 (with 5% compression)

    

#     image = cv2.imread(image_path)

#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

#     image = cv2.resize(image,image_shape[:2])

#     result, encimg = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

#     raw_image = encimg.tostring()

#     del image,result



    feature = {

        # _bytes_feature

        'image': _bytes_feature(raw_image),

        'image_name': _bytes_feature(str.encode(row['image_name'])),

        

        # _int64_feature

        'sex_female': _int64_feature(row['sex_female']),

        'sex_male': _int64_feature(row['sex_male']),

        'sex_unknown': _int64_feature(row['sex_unknown']),

        'anatom_site_head/neck': _int64_feature(row['anatom_site_head/neck']),

        'anatom_site_lower extremity': _int64_feature(row['anatom_site_lower extremity']),

        'anatom_site_oral/genital': _int64_feature(row['anatom_site_oral/genital']),

        'anatom_site_palms/soles': _int64_feature(row['anatom_site_palms/soles']),

        'anatom_site_torso': _int64_feature(row['anatom_site_torso']),

        'anatom_site_unknown': _int64_feature(row['anatom_site_unknown']),

        'anatom_site_upper extremity': _int64_feature(row['anatom_site_upper extremity']),

        

        # _float_feature

        'age_approx': _float_feature(row['age_approx']),

      }

    

    

    if _set=='train':

        feature['target'] = _int64_feature(row['target'])



    

    row_feature = tf.train.Example(features=tf.train.Features(feature=feature))

    return row_feature.SerializeToString()
train_fold_path = '../working/train_fold_tfrecords_{}x{}'.format(image_shape[0],image_shape[1])

try:

    os.mkdir(train_fold_path)

except OSError as error: 

    print(error, "\nIt's OK You are Good to GO")     
%%time

MAX_size = 1024

for fold, group in train_coded.groupby('fold'):

    Max_sub_files = len(group)//MAX_size

    print('\nWriting Fold : ',fold)

    for file_id in range(Max_sub_files+1):

        

        if file_id != Max_sub_files:

            num_image = MAX_size

        else:

            num_image = len(group)%MAX_size

            

        with tf.io.TFRecordWriter(train_fold_path + '/tain_fold_{}_{}-{}.tfrec'.format(fold,file_id,num_image)) as writer:

#             print(file_id*MAX_size,file_id*MAX_size + num_image, num_image)

            for i in range(file_id*MAX_size,file_id*MAX_size + num_image):

                row_feature_serialize = row_serialize(group.iloc[i], image_shape, _set='train')

                writer.write(row_feature_serialize)

                del row_feature_serialize



                if i%100 == 0:

                    print(i,', ',end='')

    
gc.collect()
# test_fold_path = '../working/test_fold_tfrecords_{}x{}'.format(image_shape[0],image_shape[1])

# try:

#     os.mkdir(test_fold_path)

# except OSError as error: 

#     print(error, "\nIt's OK You are Good to GO")
# %%time

# MAX_size = 1024

# for fold, group in test_coded.groupby('fold'):

#     Max_sub_files = len(group)//MAX_size

#     print('\nWriting Fold : ',fold)

#     for file_id in range(Max_sub_files+1):

        

#         if file_id != Max_sub_files:

#             num_image = MAX_size

#         else:

#             num_image = len(group)%MAX_size

            

#         with tf.io.TFRecordWriter(test_fold_path + '/test_fold_{}_{}-{}.tfrec'.format(fold,file_id,num_image)) as writer:

# #             print(file_id*MAX_size,file_id*MAX_size + num_image, num_image)

#             for i in range(file_id*MAX_size,file_id*MAX_size + num_image):

#                 row_feature_serialize = row_serialize(group.iloc[i], image_shape, _set='test')

#                 writer.write(row_feature_serialize)

#                 del row_feature_serialize



#                 if i%100 == 0:

#                     print(i,', ',end='')

    
# gc.collect()
# TEMP CODE (NEED TO REMOVE)



# train_fold = 6

# test_fold = 2
# all_train_tfrecords_path = tf.io.gfile.glob(train_fold_path + '/*.tfrec')

# test_tfrecords_path = tf.io.gfile.glob(test_fold_path + '/*.tfrec')



# val_tfrecords_path = {}

# train_tfrecords_path = {}



# for i in range(train_fold):

#     val_tfrecords_path['fold_{}'.format(i)] = [path for path in all_train_tfrecords_path if f"tain_fold_{i}_" in path] 

    

#     train_tfrecords_path['fold_{}'.format(i)] = list(set(all_train_tfrecords_path) - set(val_tfrecords_path['fold_{}'.format(i)]))



# for t,v in zip(train_tfrecords_path.items(),val_tfrecords_path.items()):

#     print('\n',t[0])

#     for path in t[1]:

#         print('Train :'+path)

#     print()

#     for path in v[1]:

#         print('validation :'+path)
# def count_data_items(filenames):

#     # the number of data items is written in the name of the .tfrec files, 

#     # i.e. tain_fold_1-1852.tfrec = 1852 data items

#     n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]

#     return np.sum(n)
# N_TRAIN_IMGS = {f'fold_{i}': count_data_items(train_tfrecords_path[f'fold_{i}'])

#                 for i in range(train_fold)}



# N_VAL_IMGS = {f'fold_{i}': count_data_items(val_tfrecords_path[f'fold_{i}'])

#                 for i in range(train_fold)}



# N_TEST_IMGS = count_data_items(test_tfrecords_path)



# print(f"number test image is {N_TEST_IMGS}. It is common for all folds.")

# print("-"*75)

# for i in range(train_fold):

#     print("-"*75)

#     print(f"Fold {i}: {N_TRAIN_IMGS[f'fold_{i}']} training and {N_VAL_IMGS[f'fold_{i}']} validation images.")

# print("-"*75)
# def decode_image(image_data):

#     image = tf.image.decode_jpeg(image_data, channels=3)

#     # convert image to floats in [0, 1] range

#     image = tf.cast(image, tf.float32) / 255.0 

#     # explicit size needed for TPU

#     image = tf.reshape(image, image_shape)

#     return image
# def read_train_tfrecord(example):

#     tfrec_format = {

        

#         # _bytes_feature

#         "image": tf.io.FixedLenFeature([], tf.string), 

#         "image_name": tf.io.FixedLenFeature([], tf.string),

        

#         # _int64_feature

#         "sex_female": tf.io.FixedLenFeature([], tf.int64),

#         "sex_male": tf.io.FixedLenFeature([], tf.int64),

#         "sex_unknown": tf.io.FixedLenFeature([], tf.int64),

#         "anatom_site_head/neck": tf.io.FixedLenFeature([], tf.int64),

#         "anatom_site_lower extremity": tf.io.FixedLenFeature([], tf.int64),

#         "anatom_site_oral/genital": tf.io.FixedLenFeature([], tf.int64),

#         "anatom_site_palms/soles": tf.io.FixedLenFeature([], tf.int64),

#         "anatom_site_torso": tf.io.FixedLenFeature([], tf.int64),

#         "anatom_site_unknown": tf.io.FixedLenFeature([], tf.int64),

#         "anatom_site_upper extremity": tf.io.FixedLenFeature([], tf.int64),

#         "target": tf.io.FixedLenFeature([], tf.int64),

        

#         # _float_feature

#         "age_approx": tf.io.FixedLenFeature([], tf.float32),

#     }

    

        

        

#     example = tf.io.parse_single_example(example, tfrec_format)

    

#     # image data

#     image = decode_image(example['image']) 

    

#     data={}

    

#     # _bytes_feature

#     data['image_name']=image_name=tf.cast(example['image_name'], tf.string)

    

#     # integer features

#     data['sex_female']=tf.cast(example['sex_female'], tf.int32)

#     data['sex_male']=tf.cast(example['sex_male'], tf.int32)

#     data['sex_unknown']=tf.cast(example['sex_unknown'], tf.int32)

#     data['anatom_site_head/neck']=tf.cast(example['anatom_site_head/neck'], tf.int32)

#     data['anatom_site_lower extremity']=tf.cast(example['anatom_site_lower extremity'], tf.int32)

#     data['anatom_site_oral/genital']=tf.cast(example['anatom_site_oral/genital'], tf.int32)

#     data['anatom_site_palms/soles']=tf.cast(example['anatom_site_palms/soles'], tf.int32)

#     data['anatom_site_torso']=tf.cast(example['anatom_site_torso'], tf.int32)

#     data['anatom_site_unknown']=tf.cast(example['anatom_site_unknown'], tf.int32)

#     data['anatom_site_upper extremity']=tf.cast(example['anatom_site_upper extremity'], tf.int32)

    

#     # _float_feature

#     data['age_approx']=tf.cast(example['age_approx'], tf.float32)



#     target=tf.cast(example['target'], tf.int32)



#     return image, target, data
# def read_test_tfrecord(example):

#     tfrec_format = {

        

#         # _bytes_feature

#         "image": tf.io.FixedLenFeature([], tf.string), 

#         "image_name": tf.io.FixedLenFeature([], tf.string),

        

#         # _int64_feature

#         "sex_female": tf.io.FixedLenFeature([], tf.int64),

#         "sex_male": tf.io.FixedLenFeature([], tf.int64),

#         "sex_unknown": tf.io.FixedLenFeature([], tf.int64),

#         "anatom_site_head/neck": tf.io.FixedLenFeature([], tf.int64),

#         "anatom_site_lower extremity": tf.io.FixedLenFeature([], tf.int64),

#         "anatom_site_oral/genital": tf.io.FixedLenFeature([], tf.int64),

#         "anatom_site_palms/soles": tf.io.FixedLenFeature([], tf.int64),

#         "anatom_site_torso": tf.io.FixedLenFeature([], tf.int64),

#         "anatom_site_unknown": tf.io.FixedLenFeature([], tf.int64),

#         "anatom_site_upper extremity": tf.io.FixedLenFeature([], tf.int64),

        

#         # _float_feature

#         "age_approx": tf.io.FixedLenFeature([], tf.float32),

#     }

    

        

        

#     example = tf.io.parse_single_example(example, tfrec_format)

    

#     # image data

#     image = decode_image(example['image']) 

    

#     data={}

    

#     # _bytes_feature

#     data['image_name']=image_name=tf.cast(example['image_name'], tf.string)

    

#     # integer features

#     data['sex_female']=tf.cast(example['sex_female'], tf.int32)

#     data['sex_male']=tf.cast(example['sex_male'], tf.int32)

#     data['sex_unknown']=tf.cast(example['sex_unknown'], tf.int32)

#     data['anatom_site_head/neck']=tf.cast(example['anatom_site_head/neck'], tf.int32)

#     data['anatom_site_lower extremity']=tf.cast(example['anatom_site_lower extremity'], tf.int32)

#     data['anatom_site_oral/genital']=tf.cast(example['anatom_site_oral/genital'], tf.int32)

#     data['anatom_site_palms/soles']=tf.cast(example['anatom_site_palms/soles'], tf.int32)

#     data['anatom_site_torso']=tf.cast(example['anatom_site_torso'], tf.int32)

#     data['anatom_site_unknown']=tf.cast(example['anatom_site_unknown'], tf.int32)

#     data['anatom_site_upper extremity']=tf.cast(example['anatom_site_upper extremity'], tf.int32)

    

#     # _float_feature

#     data['age_approx']=tf.cast(example['age_approx'], tf.float32)

    

#     return image, data
# def load_dataset(filenames, _set="train", ordered=False):

#     # Read from TFRecords. For optimal performance, reading from multiple files 

#     # at once and disregarding data order. Order does not matter since we will 

#     # be shuffling the data anyway.



#     ignore_order = tf.data.Options()

#     if not ordered:

#         # disable order, increase speed

#         ignore_order.experimental_deterministic = False



#     # automatically interleaves reads from multiple files

#     dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)

#     # uses data as soon as it streams in, rather than in its original order

#     dataset = dataset.with_options(ignore_order)

#     # returns a dataset of (image, label) pairs if labeled=True 

#     # or (image, id) pairs if labeled=False

#     dataset = dataset.map(read_train_tfrecord if _set == "train" 

#                           else read_test_tfrecord, num_parallel_calls=AUTO)

    

#     return dataset
# %%time



# training_dataset = load_dataset(train_tfrecords_path['fold_0'])



# # training_dataset.take(1)

# # <TakeDataset shapes: ((512, 512, 3), (), 

# #                       {image_name: (), sex_female: (), sex_male: (), 

# #                       sex_unknown: (), anatom_site_head/neck: (), 

# #                       anatom_site_lower extremity: (), anatom_site_oral/genital: (), 

# #                       anatom_site_palms/soles: (), anatom_site_torso: (), 

# #                       anatom_site_unknown: (), anatom_site_upper extremity: (), 

# #                       age_approx: ()}), types: (tf.float32, tf.int32, 

# #                                                 {image_name: tf.string, sex_female: tf.int32, 

# #                                                  sex_male: tf.int32, sex_unknown: tf.int32, 

# #                                                  anatom_site_head/neck: tf.int32, 

# #                                                  anatom_site_lower extremity: tf.int32, 

# #                                                  anatom_site_oral/genital: tf.int32, 

# #                                                  anatom_site_palms/soles: tf.int32, 

# #                                                  anatom_site_torso: tf.int32, 

# #                                                  anatom_site_unknown: tf.int32, 

# #                                                  anatom_site_upper extremity: tf.int32, 

# #                                                  age_approx: tf.float32})>





# target_map = {0:'benign',1:'malignant'}

# print("Example of the training data:\n")



# for image, target, data in training_dataset.take(1):

#     print("The image batch size:", image.numpy().shape)

#     print("target name:", target_map[target.numpy()])

#     print("image Name :", data['image_name'].numpy())

#     print("sex_female :", data['sex_female'].numpy())

#     print("sex_male :", data['sex_male'].numpy())

#     print("sex_unknown :", data['sex_unknown'].numpy())

#     print("age_approx(Rescaled) :", data['age_approx'].numpy())

#     print("anatom_site_head/neck :", data['anatom_site_head/neck'].numpy())

#     print("anatom_site_lower extremity :", data['anatom_site_lower extremity'].numpy())

#     print("anatom_site_oral/genital :", data['anatom_site_oral/genital'].numpy())

#     print("anatom_site_palms/soles :", data['anatom_site_palms/soles'].numpy())

#     print("anatom_site_torso :", data['anatom_site_torso'].numpy())

#     print("anatom_site_unknown :", data['anatom_site_unknown'].numpy())

#     print("anatom_site_upper extremity :", data['anatom_site_upper extremity'].numpy())

    
# %%time



# validation_dataset = load_dataset(train_tfrecords_path['fold_0'])



# target_map = {0:'benign',1:'malignant'}

# print("Example of the validation data:\n")



# for image, target, data in validation_dataset.take(1):

#     print("The image batch size:", image.numpy().shape)

#     print("target name:", target_map[target.numpy()])

#     print("image Name :", data['image_name'].numpy())

#     print("sex_female :", data['sex_female'].numpy())

#     print("sex_male :", data['sex_male'].numpy())

#     print("sex_unknown :", data['sex_unknown'].numpy())

#     print("age_approx(Rescaled) :", data['age_approx'].numpy())

#     print("anatom_site_head/neck :", data['anatom_site_head/neck'].numpy())

#     print("anatom_site_lower extremity :", data['anatom_site_lower extremity'].numpy())

#     print("anatom_site_oral/genital :", data['anatom_site_oral/genital'].numpy())

#     print("anatom_site_palms/soles :", data['anatom_site_palms/soles'].numpy())

#     print("anatom_site_torso :", data['anatom_site_torso'].numpy())

#     print("anatom_site_unknown :", data['anatom_site_unknown'].numpy())

#     print("anatom_site_upper extremity :", data['anatom_site_upper extremity'].numpy())
# def get_test_dataset(ordered=False):

#     dataset = load_dataset(test_tfrecords_path, _set="test", ordered=ordered)

#     dataset = dataset.batch(BATCH_SIZE)

#     # prefetch next batch while training (autotune prefetch buffer size)

#     dataset = dataset.prefetch(AUTO)

    

#     return dataset
# %%time



# test_dataset = get_test_dataset()



# print("Examples of the test data:")

# for image, data in test_dataset.take(2):

#     print("The image batch size:", image.numpy().shape)

#     print("image Name :", data['image_name'].numpy())

#     print("sex_female :", data['sex_female'].numpy())

#     print("sex_male :", data['sex_male'].numpy())

#     print("sex_unknown :", data['sex_unknown'].numpy())

#     print("age_approx(Rescaled) :", data['age_approx'].numpy())

#     print("anatom_site_head/neck :", data['anatom_site_head/neck'].numpy())

#     print("anatom_site_lower extremity :", data['anatom_site_lower extremity'].numpy())

#     print("anatom_site_oral/genital :", data['anatom_site_oral/genital'].numpy())

#     print("anatom_site_palms/soles :", data['anatom_site_palms/soles'].numpy())

#     print("anatom_site_torso :", data['anatom_site_torso'].numpy())

#     print("anatom_site_unknown :", data['anatom_site_unknown'].numpy())

#     print("anatom_site_upper extremity :", data['anatom_site_upper extremity'].numpy())
# # Plot Batch

# fig = plt.figure(figsize=(15,15))



# for i,image in enumerate(next(iter(test_dataset.unbatch().batch(20)))[0].numpy()):

    

#     plt.subplot(4, 5, i+1)

#     plt.imshow(image)