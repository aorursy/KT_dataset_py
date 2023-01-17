import os

import sys

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import bson

import cv2

import matplotlib.pyplot as plt
INPUT_PATH = os.path.join('..', 'input')

CATEGORY_NAMES_DF = pd.read_csv(os.path.join(INPUT_PATH, 'category_names.csv'))

TRAIN_DB = bson.decode_file_iter(open(os.path.join(INPUT_PATH, 'train.bson'), 'rb'))

TEST_DB = bson.decode_file_iter(open(os.path.join(INPUT_PATH, 'test.bson'), 'rb'))
for item in TRAIN_DB:

    break

print(type(item), list(item.keys()))

print(item['_id'], len(item['imgs']), item['category_id'],)
def decode(data):

    arr = np.asarray(bytearray(data), dtype=np.uint8)

    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 



import io

from PIL import Image



def decode_pil(data):

    return Image.open(io.BytesIO(data))



for img_dict in item['imgs']:

    img = decode(img_dict['picture'])

    plt.figure()

    plt.imshow(img)
level_tags = CATEGORY_NAMES_DF.columns[1:]

CATEGORY_NAMES_DF[CATEGORY_NAMES_DF['category_id'] == item['category_id']][level_tags]
# Method to compose a single image from 1 - 4 images

def decode_images(item_imgs):

    nx = 2 if len(item_imgs) > 1 else 1

    ny = 2 if len(item_imgs) > 2 else 1

    composed_img = np.zeros((ny * 180, nx * 180, 3), dtype=np.uint8)

    for i, img_dict in enumerate(item_imgs):

        img = decode(img_dict['picture'])

        h, w, _ = img.shape        

        xstart = (i % nx) * 180

        xend = xstart + w

        ystart = (i // nx) * 180

        yend = ystart + h

        composed_img[ystart:yend, xstart:xend] = img

    return composed_img
max_counter = 15

counter = 0

n = 4

for item in TRAIN_DB:    

    if counter % n == 0:

        plt.figure(figsize=(14, 6))

    

    mask = CATEGORY_NAMES_DF['category_id'] == item['category_id']    

    plt.subplot(1, n, counter % n + 1)

    cat_levels = CATEGORY_NAMES_DF[mask][level_tags].values.tolist()[0]

    cat_levels = [c[:25] for c in cat_levels]

    title = str(item['category_id']) + '\n'

    title += '\n'.join(cat_levels)

    plt.title(title)

    plt.imshow(decode_images(item['imgs']))

    plt.axis('off')

    

    counter += 1

    if counter == max_counter:

        break
for item in TEST_DB:

    break

print(type(item), list(item.keys()))

print(item['_id'], len(item['imgs']))
max_counter = 15

counter = 0

n = 4

for item in TEST_DB:    

    if counter % n == 0:

        plt.figure(figsize=(14, 6))

    

    plt.subplot(1, n, counter % n + 1)

    title = str(item['_id'])

    plt.title(title)

    plt.imshow(decode_images(item['imgs']))

    plt.axis('off')

    

    counter += 1

    if counter == max_counter:

        break
import struct

from tqdm import tqdm_notebook



num_dicts = 7069896 # according to data page

length_size = 4

IDS_MAPPING = {}



with open(os.path.join(INPUT_PATH, 'train.bson'), 'rb') as f, tqdm_notebook(total=num_dicts) as bar:

    item_data = []

    offset = 0

    while True:        

        bar.update()

        f.seek(offset)

        

        item_length_bytes = f.read(length_size)     

        if len(item_length_bytes) == 0:

            break                

        # Decode item length:

        length = struct.unpack("<i", item_length_bytes)[0]

        

        f.seek(offset)

        item_data = f.read(length)

        assert len(item_data) == length, "%i vs %i" % (len(item_data), length)

        

        # Check if we can decode

        item = bson.BSON.decode(item_data)

        

        IDS_MAPPING[item['_id']] = (offset, length)        

        offset += length            

            

def get_item(item_id):

    assert item_id in IDS_MAPPING

    with open(os.path.join(INPUT_PATH, 'train.bson'), 'rb') as f:

        offset, length = IDS_MAPPING[item_id]

        f.seek(offset)

        item_data = f.read(length)

        return bson.BSON.decode(item_data)
item = get_item(1234)



mask = CATEGORY_NAMES_DF['category_id'] == item['category_id']    

cat_levels = CATEGORY_NAMES_DF[mask][level_tags].values.tolist()[0]

cat_levels = [c[:25] for c in cat_levels]

title = str(item['category_id']) + '\n'

title += '\n'.join(cat_levels)

plt.title(title)

plt.imshow(decode_images(item['imgs']))

_ = plt.axis('off')
print("Unique categories: ", len(CATEGORY_NAMES_DF['category_id'].unique()))

print("Unique level 1 categories: ", len(CATEGORY_NAMES_DF['category_level1'].unique()))

print("Unique level 2 categories: ", len(CATEGORY_NAMES_DF['category_level2'].unique()))

print("Unique level 3 categories: ", len(CATEGORY_NAMES_DF['category_level3'].unique()))
gb = CATEGORY_NAMES_DF.groupby('category_level3')

cnt = gb.count()

cnt[cnt['category_id'] > 1]
gb.get_group(cnt[cnt['category_id'] > 1].index.values[0])
import seaborn as sns
plt.figure(figsize=(12,12))

_ = sns.countplot(y=CATEGORY_NAMES_DF['category_level1'])
cat_level2_counts = CATEGORY_NAMES_DF.groupby('category_level2')['category_level2'].count()

print(cat_level2_counts.describe())

print("Level 2 the most frequent category: ", cat_level2_counts.argmax())
cat_level3_counts = CATEGORY_NAMES_DF.groupby('category_level3')['category_level3'].count()

print(cat_level3_counts.describe())

print("Level 3 the most frequent category: ", cat_level3_counts.argmax())
from tqdm import tqdm_notebook



num_dicts = 7069896 # according to data page

prod_to_category = [None] * num_dicts



with tqdm_notebook(total=num_dicts) as bar:        

    TRAIN_DB = bson.decode_file_iter(open(os.path.join(INPUT_PATH, 'train.bson'), 'rb'))



    for i, item in enumerate(TRAIN_DB):

        bar.update()

        prod_to_category[i] = (item['_id'], item['category_id'])
TRAIN_CATEGORIES_DF = pd.DataFrame(prod_to_category, columns=['_id', 'category_id'])

TRAIN_CATEGORIES_DF.head()
print("Unique categories: %i in %i entries" % (len(TRAIN_CATEGORIES_DF['category_id'].unique()), len(TRAIN_CATEGORIES_DF)))
train_categories_gb = TRAIN_CATEGORIES_DF.groupby('category_id')

train_categories_count = train_categories_gb['category_id'].count()

print(train_categories_count.describe())
most_freq_cats = train_categories_count[train_categories_count == train_categories_count.max()]

less_freq_cats = train_categories_count[train_categories_count == train_categories_count.min()]



print("Most frequent category: ", CATEGORY_NAMES_DF[CATEGORY_NAMES_DF['category_id'].isin(most_freq_cats.index)].values)

print("Less frequent category: ", CATEGORY_NAMES_DF[CATEGORY_NAMES_DF['category_id'].isin(less_freq_cats.index)].values)
most_freq_cat = most_freq_cats.index[0]



plt.figure(figsize=(16, 4))

mask = CATEGORY_NAMES_DF['category_id'] == most_freq_cat    

cat_levels = CATEGORY_NAMES_DF[mask][level_tags].values.tolist()[0]

title = str(most_freq_cat) + '\n'

title += '\n'.join(cat_levels)

plt.suptitle(title)



most_freq_cat_ids = train_categories_gb.get_group(most_freq_cat)['_id']

max_counter = 50

counter = 0

n = 10

for item_id in most_freq_cat_ids.values[:max_counter]:    

    if counter > 0 and counter % n == 0:

        plt.figure(figsize=(14, 6))

    

    item = get_item(item_id)

    

    mask = CATEGORY_NAMES_DF['category_id'] == item['category_id']    

    plt.subplot(1, n, counter % n + 1)

    plt.imshow(decode_images(item['imgs']))

    plt.axis('off')

    

    counter += 1

    if counter == max_counter:

        break
for less_freq_cat in less_freq_cats.index:

    less_freq_cat_ids = train_categories_gb.get_group(less_freq_cat)['_id']

    counter = 0

    n = 12

    

    plt.figure(figsize=(16, 4))

    mask = CATEGORY_NAMES_DF['category_id'] == less_freq_cat    

    cat_levels = CATEGORY_NAMES_DF[mask][level_tags].values.tolist()[0]

    title = str(less_freq_cat) + '\n'

    title += '\n'.join(cat_levels)

    plt.suptitle(title)



    for item_id in less_freq_cat_ids.values:    

        if counter > 0 and counter % n == 0:

            plt.figure(figsize=(16, 4))



        item = get_item(item_id)



        mask = CATEGORY_NAMES_DF['category_id'] == item['category_id']    

        plt.subplot(1, n, counter % n + 1)

        plt.imshow(decode_images(item['imgs']))

        plt.axis('off')



        counter += 1        
sorted_train_categories_count = sorted(train_categories_count.values)

index_8000 = np.where(np.array(sorted_train_categories_count) > 8000)[0][0]



plt.figure(figsize=(12, 6))

plt.title("Sorted category counts")

_ = plt.plot(sorted_train_categories_count, '*-')



plt.figure(figsize=(12, 6))

plt.subplot(121)

plt.title("Sorted category counts < %i" % index_8000)

_ = plt.plot(sorted_train_categories_count[:index_8000], '*-')



plt.subplot(122)

plt.title("Sorted category counts > %i" % index_8000)

_ = plt.plot(sorted_train_categories_count[index_8000:], '*-')
plt.figure(figsize=(12, 6))

plt.title('Category_count vs Image_count')

bin_size = 25

plt.hist(train_categories_count, bins=range(0, int(1e4), bin_size))

plt.xlabel('Amount of available images')

_ = plt.ylabel('Number of classes')