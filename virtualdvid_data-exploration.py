%%capture

!pip install -q gapcv
!pip show gapcv
import os

import cv2

import numpy as np



from gapcv.vision import Images



import warnings

warnings.filterwarnings("ignore")



from matplotlib import pyplot as plt

%matplotlib inline



print(os.listdir("../input"))

print(os.listdir("./"))
def plot_sample(imgs_set, labels_set, img_size=(12,12), columns=4, rows=4, random=False):

    """

    Plot a sample of images

    """

    

    fig=plt.figure(figsize=img_size)

    

    for i in range(1, columns*rows + 1):

        if random:

            img_x = np.random.randint(0, len(imgs_set))

        else:

            img_x = i-1

        img = imgs_set[img_x]

        ax = fig.add_subplot(rows, columns, i)

        ax.set_title(str(labels_set[img_x]))

        plt.axis('off')

        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    plt.show()
data_set = '../input/oregon_wildlife/oregon_wildlife'



labels = os.listdir(data_set)

print("Number of Labels:", len(labels))



total = 0

for lb in os.scandir(data_set):

    print('folder: {} images: {}'.format(lb.name, len(os.listdir(lb))))

    total += len(os.listdir(lb))

print('Total images:', total)
dataset_name = 'wildlife128'

wildlife_filter = ['black_bear', 'bald_eagle', 'cougar', 'elk', 'gray_wolf']



if not os.path.isfile('{}.h5'.format(dataset_name)):

    ## create two list to use as paramaters in GapCV

    print('{} preprocessing started'.format(dataset_name))

    images_list = []

    classes_list = []

    for folder in os.scandir('../input/oregon_wildlife/oregon_wildlife'):

        if folder.name in wildlife_filter:

            for image in os.scandir(folder.path):

                images_list.append(image.path)

                classes_list.append(image.path.split('/')[-2])



    ## GapCV

    images = Images(

        dataset_name,

        images_list,

        classes_list,

        config=[

            'resize=(128,128)',

            'store', 'stream'

        ]

    )
print('content:', os.listdir("./"))

print('time to load data set:', images.elapsed)

print('number of images in data set:', images.count)

print('classes:', images.classes)

print('data type:', images.dtype)
images.split = 0.2

X_test, Y_test = images.test

images.minibatch = 128

gap_generator = images.minibatch

X_train, Y_train = next(gap_generator)
plot_sample(X_train, Y_train, random=True)
plot_sample(X_test, Y_test, random=True)
del images

images = Images(

    config=['stream'],

    augment=[

        'flip=horizontal',

        'edge',

        'zoom=0.3',

        'denoise'

    ]

)

images.load(dataset_name)

print('{} dataset ready for streaming'.format(dataset_name))
images.split = 0.2

X_test, Y_test = images.test

images.minibatch = 16

gap_generator = images.minibatch

X_train, Y_train = next(gap_generator)
plot_sample(X_train, Y_train)