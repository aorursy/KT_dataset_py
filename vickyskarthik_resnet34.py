# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
%reload_ext autoreload

%autoreload 2

%matplotlib inline


from fastai import *

from fastai.vision import *

import os

import pandas as pd

from fastai.metrics import error_rate
print(os.listdir("../input/plant-seedlings-classification"))
train_dir = '../input/plant-seedlings-classification/train/'

test_dir = '../input/plant-seedlings-classification/test/'
print(os.listdir(train_dir)[:5])

print(os.listdir(test_dir)[:5])
tfms = get_transforms()

data = ImageDataBunch.from_folder( 

    path = train_dir,

    test="../test",

    valid_pct = 0.2,

    bs = 16,

    size = 336,

    ds_tfms = tfms,

    num_workers = 0

).normalize(imagenet_stats)

data

print(data.classes)

data.show_batch()
learn = cnn_learner(data, models.resnet152, metrics=accuracy, model_dir="/tmp/models")
learn.fit_one_cycle(3)
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_multi_top_losses(10, figsize=(6,6))
interp.plot_confusion_matrix()
class_score, y = learn.get_preds(DatasetType.Test)

class_score = np.argmax(class_score, axis=1)
predicted_classes = [data.classes[i] for i in class_score]

predicted_classes[:10]
submission  = pd.DataFrame({

    "file": os.listdir(test_dir),

    "species": predicted_classes

})

submission.to_csv("submission_resnet152.csv", index=False)

submission[:10]