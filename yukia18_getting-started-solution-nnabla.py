import os

import random



import numpy as np

import pandas as pd

import cv2



import matplotlib.pyplot as plt

import seaborn as sns



import albumentations as A



from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



import nnabla as nn

import nnabla.functions as F

import nnabla.parametric_functions as PF

import nnabla.solvers as S

from nnabla.ext_utils import get_extension_context

from nnabla.utils.data_source import DataSource

from nnabla.utils.data_iterator import data_iterator





%matplotlib inline
def seed_everything(seed=42):

    os.environ['PYTHONHASHSEED'] = str(seed)

    os.environ['NNABLA_CUDNN_DETERMINISTIC'] = '1'

    random.seed(seed)

    np.random.seed(seed)

    



def enable_cudnn_extension(device_id=0, type_config='float'):

    assert type_config in ['float', 'half', 'mixed_half']

    cxt = get_extension_context('cudnn', device_id=device_id, type_config=type_config)

    nn.set_default_context(cxt)

    return cxt
DIRPATH = '../input/ailab-ml-training-0/'

TRAIN_IMAGE_DIR = 'train_images/train_images/'

TEST_IMAGE_DIR = 'test_images/test_images/'



ID = 'fname'

TARGET = 'label'



VALID_SIZE = 0.2

EPOCHS = 5

BATCH_SIZE = 64

LR = 1e-3

WEIGHT_DECAY = 1e-5



SEED = 42

seed_everything(SEED)



DEVICE_ID = 0

TYPE_CONFIG = 'float'



enable_cudnn_extension(device_id=DEVICE_ID, type_config=TYPE_CONFIG)
os.listdir(DIRPATH)
train_df = pd.read_csv(os.path.join(DIRPATH, 'train.csv'))
train_df.head()
sample_index = [0, 10, 100]



fig, ax = plt.subplots(1, len(sample_index))

fig.set_size_inches(4 * len(sample_index), 4)



for i, idx in enumerate(sample_index):

    fname, label = train_df.loc[idx, [ID, TARGET]]

    img = cv2.imread(os.path.join(DIRPATH, TRAIN_IMAGE_DIR, fname))

    ax[i].imshow(img)

    ax[i].set_title(f'{fname} - label: {label}')



plt.show()
class MNISTDataSource(DataSource):

    def __init__(

        self,

        fname_list,

        label_list,

        image_dir,

        transform=None,

        shuffle=False,

        rng=None,

    ):

        super().__init__(shuffle=shuffle, rng=rng)

        

        self.fname_list = fname_list

        self.label_list = label_list

        self.image_dir = image_dir

        self.transform = transform

        

        self._size = len(fname_list)

        self._variables = ('x', 'y')

        self.rng = rng if rng is not None else np.random.RandomState(313)

        self.reset()

    

    def reset(self):

        if self._shuffle:

            self._indexes = self.rng.permutation(self._size)

        else:

            self._indexes = np.arange(self._size)

        super(MNISTDataSource, self).reset()

    

    def _get_data(self, position):

        idx = self._indexes[position]

        

        fname = self.fname_list[idx]

        label = self.label_list[idx]

        

        image = cv2.imread(os.path.join(self.image_dir, fname))

        if self.transform is not None:

            image = self.transform(image=image)['image']

        

        return image, label
def simple_classifier(x, test=False, scope='simple_classifier'):

    with nn.parameter_scope(scope):

        # (N, 3, 28, 28) --> (N, 32, 14, 14)

        x = PF.convolution(x, 32, (3, 3), stride=(1, 1), pad=(1, 1), name='conv1', with_bias=True)

        x = F.relu(x)

        x = F.max_pooling(x, (2, 2))

        # (N, 32, 14, 14) --> (N, 64, 7, 7)

        x = PF.convolution(x, 64, (3, 3), stride=(1, 1), pad=(1, 1), name='conv2', with_bias=True)

        x = F.relu(x)

        x = F.max_pooling(x, (2, 2))

        # (N, 64, 7, 7) --> (N, 128, 7, 7)

        x = PF.convolution(x, 128, (3, 3), stride=(1, 1), pad=(1, 1), name='conv3', with_bias=True)

        x = F.relu(x)

        # (N, 128 * 7 * 7) --> (N, 10)

        x = F.reshape(x, (x.shape[0], -1))

        x = PF.affine(x, 10, name='affine1', with_bias=True)

    return x
fname_list = train_df[ID].to_list()

label_list = train_df[TARGET].to_list()



train_fname_list, valid_fname_list, train_label_list, valid_label_list = train_test_split(

    fname_list, label_list, test_size=VALID_SIZE, random_state=SEED, shuffle=True

)
len(fname_list), len(train_fname_list), len(valid_fname_list)
image_dir = os.path.join(DIRPATH, TRAIN_IMAGE_DIR)



transform = A.Compose([

    A.Rotate(limit=10, interpolation=1, p=1.0),

])



train_data_source = MNISTDataSource(

    train_fname_list, train_label_list, image_dir,

    transform=transform, shuffle=True, rng=None,

)

valid_data_source = MNISTDataSource(

    valid_fname_list, valid_label_list, image_dir, 

    transform=transform, shuffle=False, rng=None

)



train_data_iterator = data_iterator(

    train_data_source, BATCH_SIZE, rng=None, with_memory_cache=False, with_file_cache=False,

)

valid_data_iterator = data_iterator(

    valid_data_source, BATCH_SIZE, rng=None, with_memory_cache=False, with_file_cache=False,

)
x, y = train_data_iterator.next()

image = nn.Variable(x.shape)

label = nn.Variable((y.shape[0], 1))

label_hat = simple_classifier(image, test=False, scope='simple_classifier')

label_hat.persistent = True

loss = F.mean(F.softmax_cross_entropy(label_hat, label, axis=1))



x, y = valid_data_iterator.next()

val_image = nn.Variable(x.shape)

val_label = nn.Variable((y.shape[0], 1))

val_label_hat = simple_classifier(val_image, test=True, scope='simple_classifier')

val_label_hat.persistent = True

val_loss = F.mean(F.softmax_cross_entropy(val_label_hat, val_label, axis=1))



solver = S.Adam(alpha=LR)

with nn.parameter_scope('simple_classifier'):

    solver.set_parameters(nn.get_parameters())
for epoch in range(EPOCHS):

    

    # training

    

    train_loss_list = []

    train_accuracy_list = []

    

    while epoch == train_data_iterator.epoch:

        x, y = train_data_iterator.next()

        image.d = x

        label.d = y.reshape(y.shape[0], 1)

        solver.zero_grad()

        loss.forward()

        loss.backward(clear_buffer=True)

        solver.weight_decay(WEIGHT_DECAY)

        solver.update()

        

        train_loss_list.append(loss.d)

        accuracy = accuracy_score(np.argmax(label_hat.d, axis=1), y)

        train_accuracy_list.append(accuracy)

    

    # validation

    

    valid_loss_list = []

    valid_accuracy_list = []

    

    while epoch == valid_data_iterator.epoch:

        x, y = valid_data_iterator.next()

        val_image.d = x

        val_label.d = y.reshape(y.shape[0], 1)

        val_loss.forward(clear_buffer=True)

        

        valid_loss_list.append(val_loss.d)

        accuracy = accuracy_score(np.argmax(val_label_hat.d, axis=1), y)

        valid_accuracy_list.append(accuracy)

    

    # verbose

    

    print('epoch: {}/{} - loss: {:.5f} - accuracy: {:.3f} - val_loss: {:.5f} - val_accuracy: {:.3f}'.format(

        epoch,

        EPOCHS, 

        np.mean(train_loss_list),

        np.mean(train_accuracy_list),

        np.mean(valid_loss_list),

        np.mean(valid_accuracy_list)

    ))
submission_df = pd.read_csv(os.path.join(DIRPATH, 'sample_submission.csv'))
submission_df.head()
fname_list = submission_df[ID].to_list()

label_list = submission_df[TARGET].to_list()



image_dir = os.path.join(DIRPATH, TEST_IMAGE_DIR)



transform = None



test_data_source = MNISTDataSource(

    fname_list, label_list, image_dir, 

    transform=transform, shuffle=False, rng=None

)



test_data_iterator = data_iterator(

    test_data_source, 1, rng=None, with_memory_cache=False, with_file_cache=False,

)
predictions = []

while 1 > test_data_iterator.epoch:

    x, y = test_data_iterator.next()

    val_image.d = x

    val_label.d = y.reshape(y.shape[0], 1)

    val_label_hat.forward(clear_buffer=True)

    pred = np.argmax(val_label_hat.d, axis=1)

    predictions.append(pred[0])
submission_df[TARGET] = predictions
sample_index = [0, 10, 100]



fig, ax = plt.subplots(1, len(sample_index))

fig.set_size_inches(4 * len(sample_index), 4)



for i, idx in enumerate(sample_index):

    fname, label = submission_df.loc[idx, [ID, TARGET]]

    img = cv2.imread(os.path.join(DIRPATH, TEST_IMAGE_DIR, fname))

    ax[i].imshow(img)

    ax[i].set_title(f'{fname} - label: {label}')



plt.show()
submission_df.head()
submission_df.to_csv('submission.csv', index=False)
from IPython.display import FileLink

FileLink('submission.csv')