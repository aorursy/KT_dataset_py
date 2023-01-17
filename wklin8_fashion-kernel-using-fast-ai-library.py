import torch
import pandas as pd
import numpy as np
import fastai

from fastai.conv_learner import *

from pathlib import Path
!ls ../input
!ls ../input/resnet50

cache_dir = os.path.expanduser(os.path.join('~', '.torch'))
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
models_dir = os.path.join(cache_dir, 'models')
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

!cp ../input/resnet50/resnet50.pth /tmp/.torch/models/resnet50-19c8e357.pth

# comment the following line if you want to using cached weights
!rm -rf ../working/fashionmnist

!ls ../working
PATH = Path('../input/fashionmnist')

train_raw = pd.read_csv(PATH/"fashion-mnist_train.csv")
test_raw = pd.read_csv(PATH/"fashion-mnist_test.csv")

def add_color_channel(matrix):
    matrix = np.stack((matrix, ) *3, axis = -1)
    return matrix
labels_dict={
'0': 'T-shirt/top',
'1': 'Trouser',
'2': 'Pullover',
'3': 'Dress',
'4': 'Coat',
'5': 'Sandal',
'6': 'Shirt',
'7': 'Sneaker',
'8': 'Bag',
'9': 'Ankle boot'
}

arch=resnet50

def display_img(df, idx):
    l = str(df.iloc[idx].values[0])
    plt.imshow(df.iloc[idx][1:].values.reshape(28, 28))
    plt.title(labels_dict[l])

display_img(train_raw, 42)
sz = 28 #that depends on the architecture of the image net
tfms = tfms_from_model(resnet50, sz)

total_rows = train_raw.shape[0]
train_ratio = 0.2

val_idxs = list(range(int(total_rows * train_ratio), total_rows))
# or
val_idxs = get_cv_idxs(total_rows, val_pct=train_ratio)

xs, y = train_raw[train_raw.columns[1:]], train_raw[train_raw.columns[:1]]

((val_xs, trn_xs), (val_y, trn_y)) = split_by_idx(val_idxs, xs, y)

img_size = 28 #sqrt (784)

trn_xs = trn_xs.values
val_xs = val_xs.values
test_xs = test_raw[test_raw.columns[1:]].values

trn_xs = trn_xs.reshape(-1, img_size, img_size) / 255.
trn_xs = add_color_channel(trn_xs)

val_xs = val_xs.reshape(-1, img_size, img_size) / 255.
val_xs = add_color_channel(val_xs)

test_xs = test_xs.reshape(-1, img_size, img_size) / 255.
test_xs = add_color_channel(test_xs)

train_set = (trn_xs, trn_y.values.flatten())
valid_set = (val_xs, val_y.values.flatten())

classes = np.unique(trn_y)

print(train_set[0].shape, train_set[1].shape, valid_set[0].shape, valid_set[1].shape)

data = ImageClassifierData.from_arrays(path="../working/fashionmnist", 
                                       trn=train_set,
                                       val=valid_set,
                                       classes=classes,
                                       test=test_xs,
                                       tfms=tfms,
                                      )

learn = ConvLearner.pretrained(arch, data, precompute=True)
lrf=learn.lr_find()

## need to call lr_find() first
learn.sched.plot()
lr = 1e-2
learn.fit(lr, 2)
learn.precompute = False
lr = 1e-3
learn.fit(lr, 3, cycle_len=1, cycle_mult=2)
# run this to increase the prediction, but it can be slow

# learn.unfreeze()
# lrs=np.array([1e-4,1e-3,1e-2])
# learn.fit(lrs, 3, cycle_len=1, cycle_mult=2)
y = learn.predict(is_test=True)

# or use TTA, it can have a slightly better accuracy
# y, _ = learn.TTA(is_test=True)
# y = np.mean(y, axis=0)

y = np.argmax(y, axis=1)
!rm -rf ../working/fashionmnist