# Put these at the top of every notebook, to get automatic reloading and inline plotting
%reload_ext autoreload
%autoreload 2
%matplotlib inline
# This file contains all the main external libs we'll use
from fastai.imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *
import os
# Let's see what the directories are like
print(os.listdir("../input/"))

# After some listdir fun we've determined the proper path
PATH = '../input/dice-d4-d6-d8-d10-d12-d20-images/dice-d4-d6-d8-d10-d12-d20/dice'

# Let's make the resnet101 model available to FastAI
# Credit to Shivam for figuring this out: 
# https://www.kaggle.com/shivamsaboo17/amazon-from-space-using-fastai/notebook AND http://forums.fast.ai/t/how-can-i-load-a-pretrained-model-on-kaggle-using-fastai/13941/7
from os.path import expanduser, join, exists
from os import makedirs
cache_dir = expanduser(join('~', '.torch'))
if not exists(cache_dir):
    makedirs(cache_dir)
models_dir = join(cache_dir, 'models')
if not exists(models_dir):
    makedirs(models_dir)

# copy time!
!cp ../input/resnet101/resnet101.pth /tmp/.torch/models/resnet101-5d3b4d8f.pth
arch=resnet101
workers=8 # This number should match your number of processor cores for best results
sz=240 # image size 240x240
bs=64 # batch size
learnrate = 5e-3 #0.005
dropout = [0.3,0.6] # I found this to be the sweet spot for this data set to reduce overfitting with high accuracy
# I used the following notebook structure to help determine a good rate: 
# https://github.com/ucffool/fastai-custom-learning-notebooks/blob/master/Testing%20Dropout%20Rates%20(small%20images).ipynb
# Since this data set already incorporates basic rotations in the set due to the method used, no additional transforms used (the model actually overfits and gets worse)
# tfms = tfms_from_model(arch, sz, max_zoom=1.1)

# TESTING added lighting changes | July 13, 2018 Version 21
tfms = tfms_from_model(arch, sz, aug_tfms = [RandomLighting(b=0.5, c=0.5, tfm_y=TfmType.NO)], max_zoom=1.1)
data = ImageClassifierData.from_paths(PATH, tfms=tfms, bs=bs, num_workers=workers)
# adjust the path for writing to something writeable (this avoids an error about read-only directories)
import pathlib
data.path = pathlib.Path('.')
# Make sure precompute is FALSE or it will not complete when you COMMIT (even if it runs when editing)
learn = ConvLearner.pretrained(arch, data, precompute=False, ps=dropout)
# Finding the learning rate
lrf=learn.lr_find()
# Plotting the learning rate
learn.sched.plot()
learn.fit(learnrate, 1)
%time learn.fit(learnrate, 2, cycle_len=1)
lr = learnrate # just in case the next code block is skipped
learn.unfreeze()
# BatchNorm is recommended when using anything bigger than resnet34, but on my local test the results were worse so I'll comment it out for now
# learn.bn_freeze(True) 
lr=np.array([learnrate/100,learnrate/10,learnrate])
%time learn.fit(lr, 3, cycle_len=1, cycle_mult=2)
# %time learn.fit(lr, 3, cycle_len=1)
learn.save("240_resnet101_all")
learn.load("240_resnet101_all")
%time log_preds,y = learn.TTA()
probs = np.mean(np.exp(log_preds),0)
accuracy_np(probs,y)
preds = np.argmax(probs, axis=1)
probs = probs[:,1]

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, preds)
plot_confusion_matrix(cm, data.classes)
