import numpy as np 
import pandas as pd
from fastai.conv_learner import *
import os

# Input data files are available in the "../input/" directory.
# Any results you write to the current directory are saved as output.

PATH = "data/dogscats/"
# Create FastAi friendly directory structure for training and test sets
#
# The Kaggle data looks like this:
#
# ../input/train/: 12500 cats.*.jpg images
#                  12500 dogs.*.jpg images
#  ../input/test/ : 12500 nnnnn.jpg images
#
# To keep it simple with the Fastai code below,
# And also to add a Validation Set to the game (randomly chosen from the training set),
# I'm creating the following data structure using symbolic links:
#
# data/dogscats/train/cats: 11500 cats.*.jpg images
# data/dogscats/train/dogs: 11500 dogs.*.jpg images
# data/dogscats/valid/cats: 1000 cats.*.jpg images
# data/dogscats/valid/dogs: 1000 dogs.*.jpg images
# data/dogscats/test1: 12500 nnnnn.jpg images
#
! rm -rf data
os.makedirs(f'{PATH}train/cats')
os.makedirs(f'{PATH}train/dogs')
os.makedirs(f'{PATH}valid/cats')
os.makedirs(f'{PATH}valid/dogs')
os.makedirs(f'{PATH}test1')
# Symbolic links for test images
for file in os.listdir('../input/test'):
    os.symlink('/kaggle/input/test/' + file, '/kaggle/working/data/dogscats/test1/' + file)
# Symbolic links for cats and dogs images from training set
for file in os.listdir('../input/train'):
    if 'cat' in file:
        os.symlink('/kaggle/input/train/' + file, '/kaggle/working/data/dogscats/train/cats/' + file)
    elif 'dog' in file:
        os.symlink('/kaggle/input/train/' + file, '/kaggle/working/data/dogscats/train/dogs/' + file)
# Need to move 1000 images from training set to validation set for cats and dogs
import random
for r in random.sample(range(12499), 1000):
    os.rename(f'{PATH}train/cats/cat.{r}.jpg', f'{PATH}valid/cats/cat.{r}.jpg')
for r in random.sample(range(12499), 1000):
    os.rename(f'{PATH}train/dogs/dog.{r}.jpg', f'{PATH}valid/dogs/dog.{r}.jpg')
# Image size, batch size and pretrained model architecture
sz=224
bs=20
arch=resnet50
tfms = tfms_from_model(arch, sz, aug_tfms=transforms_side_on, max_zoom=1.1)    # transformers of images (train and valid)
data = ImageClassifierData.from_paths(PATH, tfms=tfms, bs=bs,                  # Read in images and their labels
                                      trn_name='train', 
                                      test_name='test1', 
                                      val_name='valid')
learn = ConvLearner.pretrained(arch, data, precompute=False)                   # Prepare neural network layers
learn.fit(0.01, 3, cycle_len=1)                                                # 3 training iterations with a fixed learning rate
learn.unfreeze()                              # Unfreeze all layers
learn.bn_freeze(True)                         # Freeze batch normalization parameters (standard deviation & mean)
learn.fit(lrs=[1e-5,1e-4,1e-2], n_cycle=1,    # Stochastic gradient descent gradually changing learning rates
          cycle_len=1)
log_preds,y = learn.TTA()                     # Predict with Test Time Augmentation (to validation images)
probs = np.mean(np.exp(log_preds),0)          # Probabilities, accurary and logistic loss
accuracy_np(probs, y), metrics.log_loss(y, probs)
# Get predictions from test set
prediction = learn.predict(is_test=True)      # Get predictions from test set
pred_test = np.argmax(prediction, axis=1)     # From log scale to binary values
label_probs = np.exp(prediction[:,1])         # From log scale to probabilities
# Create submission file: 2 colmuns with header id, label
submission = pd.DataFrame({'id':os.listdir(f'{PATH}test1'), 'label':label_probs})
submission['id'] = submission['id'].map(lambda x: x.split('.')[0])
submission['id'] = submission['id'].astype(int)
submission = submission.sort_values('id')
submission.to_csv('../working/submission.csv', index=False)
# Clean up to prevent too many output files error message at commit
! rm -rf data