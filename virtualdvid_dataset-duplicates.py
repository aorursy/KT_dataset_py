%%capture

#install GapCV

!pip install -q gapcv==1.0rc13
import os

import hashlib

import gc

import numpy as np



from gapcv.vision import Images



import warnings

warnings.filterwarnings('ignore')



from matplotlib import pyplot as plt

%matplotlib inline



print(os.listdir('../input'))

print(os.listdir('./'))
def delete_duplicates(dataset):

    cleaner_control = {}

    for root, dirs, files in os.walk(dataset):

        for file in files:

            _dir = root.split('/')[-1]

            img_path = os.path.join(root, file)

            if os.path.isfile(img_path):

                with open(img_path, 'rb') as f:

                    filehash = hashlib.md5(f.read()).hexdigest()

                if _dir not in cleaner_control:

                    cleaner_control[_dir] = []

                if filehash not in cleaner_control[_dir]:

                    cleaner_control[_dir].append(filehash)

                else:

                    os.remove(img_path)
dataset_inventory = {}

for root, dirs, files in os.walk('../input/oregon-wildlife'):

    for _dir in dirs:

        if _dir not in dataset_inventory:

            dataset_inventory[_dir] = {}

            dataset_inventory[_dir]['count'] = 0

            dataset_inventory[_dir]['duplicates'] = 0



    if not dataset_inventory['oregon_wildlife']:

        dataset_inventory['oregon_wildlife'] = {}

        dataset_inventory['oregon_wildlife']['count'] = 0

        dataset_inventory['oregon_wildlife']['duplicates'] = 0

    for file in files:

        _dir = root.split('/')[-1]

        img_path = os.path.join(root, file)

        if os.path.isfile(img_path) and not img_path.endswith('.h5'):

            dataset_inventory['oregon_wildlife']['count'] += 1

            dataset_inventory[_dir]['count'] += 1

            with open(img_path, 'rb') as f:

                filehash = hashlib.md5(f.read()).hexdigest()

            if filehash not in dataset_inventory[_dir]:

                dataset_inventory[_dir][filehash] = {}

                dataset_inventory[_dir][filehash]['count'] = 1

                dataset_inventory[_dir][filehash]['paths'] = []

            else:

                dataset_inventory['oregon_wildlife']['duplicates'] += 1

                dataset_inventory[_dir]['duplicates'] += 1

                dataset_inventory[_dir][filehash]['count'] += 1

                # os.remove(img_path) # uncomment to delete duplicated image

            dataset_inventory[_dir][filehash]['paths'].append(img_path)
# https://stackoverflow.com/a/50205834

def plot_stacked_bar(data, series_labels, category_labels=None, 

                     show_values=False, value_format="{}", y_label=None, 

                     colors=None, grid=True, reverse=False):

    """Plots a stacked bar chart with the data and labels provided.



    Keyword arguments:

    data            -- 2-dimensional numpy array or nested list

                       containing data for each series in rows

    series_labels   -- list of series labels (these appear in

                       the legend)

    category_labels -- list of category labels (these appear

                       on the x-axis)

    show_values     -- If True then numeric value labels will 

                       be shown on each bar

    value_format    -- Format string for numeric value labels

                       (default is "{}")

    y_label         -- Label for y-axis (str)

    colors          -- List of color labels

    grid            -- If True display grid

    reverse         -- If True reverse the order that the

                       series are displayed (left-to-right

                       or right-to-left)

    """



    ny = len(data[0])

    ind = list(range(ny))



    axes = []

    cum_size = np.zeros(ny)



    data = np.array(data)

    

    plt.figure(figsize=(25, 10))

    plt.rcParams.update({'font.size': 20})



    if reverse:

        data = np.flip(data, axis=1)

        category_labels = reversed(category_labels)



    for i, row_data in enumerate(data):

        axes.append(

            plt.bar(

                ind,

                row_data,

                bottom=cum_size, 

                label=series_labels[i],

                color=colors[i]

            )

        )

        cum_size += row_data



    if category_labels:

        plt.xticks(ind, category_labels)

        plt.xticks(rotation='vertical')



    if y_label:

        plt.ylabel(y_label)



    plt.legend()



    if grid:

        plt.grid()



    if show_values:

        for axis in axes:

            for bar in axis:

                w, h = bar.get_width(), bar.get_height()

                plt.text(

                    bar.get_x() + w/2,

                    bar.get_y() + h/2,

                    value_format.format(h),

                    ha='center',

                    va='center'

                )

                

    # plt.savefig('bar.png')

    plt.show()
total = dataset_inventory['oregon_wildlife']['count']

duplicates = dataset_inventory['oregon_wildlife']['duplicates']



print(f'labels: {len(dataset_inventory) - 1}')

print(f'Total images: {total}')

print(f'Total duplicates: {duplicates}')

print(f'% duplicates: {(duplicates/total)*100:.2f}%')
series_labels = ['count', 'duplicates']



data = [[], []]

category_labels = []

for key, value in dataset_inventory.items():

    if key != 'oregon_wildlife':

        print(f"label: {key} % duplicates: {(value['duplicates']/value['count'])*100:.2f}%")

        data[0].append(value['count'] - value['duplicates'])

        data[1].append(value['duplicates'])

        if key not in category_labels:

            category_labels.append(key)
plot_stacked_bar(

    data, 

    series_labels, 

    category_labels=category_labels, 

    show_values=True, 

    value_format="{:.1f}",

    colors=['tab:green', 'tab:orange'],

    y_label="Quantity (units)"

)



del dataset_inventory

gc.collect()
wildlife_filter = ('black_bear', 'bald_eagle', 'cougar', 'elk', 'gray_wolf')



## create two list to use as paramaters in GapCV

images_list = []

classes_list = []

for folder in os.scandir('../input/oregon-wildlife/oregon_wildlife/oregon_wildlife'):

    if folder.name in wildlife_filter:

        for image in os.scandir(folder.path):

            images_list.append(image.path)

            classes_list.append(image.path.split('/')[-2])
dataset_name = 'wildlife128'
images = Images(

    dataset_name,

    images_list,

    classes_list,

    config=[

        'resize=(50,50)',

        'verbose',

        'duplicate', # <- save the file with duplicates 

        'store'

    ]

)
print('content:', os.listdir('./'))

print(f'time to preprocess the data set: {images.elapsed}')

print(f'total images in dataset: {images.count}')

print(f"dataset size: {os.path.getsize('wildlife128.h5')/(1024*1024):.2f} MB")

del images

gc.collect()

os.remove('wildlife128.h5')
images = Images(

    dataset_name,

    images_list,

    classes_list,

    config=[

        'resize=(50,50)',

        'verbose',

        'store'

    ]

)
print('content:', os.listdir('./'))

print(f'time to preprocess the data set: {images.elapsed}')

print(f'total images in dataset: {images.count}')

print(f"dataset size: {os.path.getsize('wildlife128.h5')/(1024*1024):.2f} MB")

del images

gc.collect()

os.remove('wildlife128.h5')