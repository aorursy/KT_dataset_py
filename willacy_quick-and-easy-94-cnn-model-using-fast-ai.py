# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# Any results you write to the current directory are saved as output.
# Import the fast.ai library

from fastai.vision import *
path = Path('/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/test/')

path.mkdir(parents=True, exist_ok=True)
path.ls()
np.random.seed(42)

data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,

        ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)
data.classes
data.show_batch(rows=3, figsize=(7,8))
data.classes, data.c, len(data.train_ds), len(data.valid_ds)
learn = cnn_learner(data, models.resnet34, metrics=error_rate, model_dir=Path('/kaggle/working'))
learn.fit_one_cycle(4)
learn.save('stage-1')

learn.unfreeze()

learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(3, max_lr=slice(3e-5,6e-4))
learn.save('stage-2')
learn.load('stage-2');
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_confusion_matrix()
learn.path = Path('/kaggle/working')

learn.export()
path = Path('/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/train')
PNEUMONIA = ['person63_bacteria_306.jpeg',

'person26_bacteria_122.jpeg',

'person890_bacteria_2814.jpeg',

'person519_virus_1038.jpeg',

'person968_virus_1642.jpeg']



NORMAL = ['IM-0757-0001.jpeg',

'IM-0540-0001.jpeg',

'IM-0683-0001.jpeg',

'NORMAL2-IM-1288-0001.jpeg',

'NORMAL2-IM-0482-0001.jpeg',]
learn = load_learner(Path('/kaggle/working'))
print('These 5 X-rays have Pneumonia')

for l,i in enumerate(PNEUMONIA):

    img = open_image(path/'PNEUMONIA'/i)

    pred_class,pred_idx,outputs = learn.predict(img)

    if str(pred_class) == 'PNEUMONIA':

        print(f'Prediction for image {l+1} is Correct')
print('These 5 X-rays are normal and do not show any signs of Pneumonia')

for l, i in enumerate(NORMAL):

    img = open_image(path/'NORMAL'/i)

    pred_class,pred_idx,outputs = learn.predict(img)

    if str(pred_class) == 'NORMAL':

        print(f'Prediction for {l+6} is Correct')