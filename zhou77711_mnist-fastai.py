# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
%reload_ext autoreload

%autoreload 2

%matplotlib inline
import fastai

from fastai.vision import *

import fastai.utils.collect_env; fastai.utils.collect_env.show_install(1)

print('Working on "%s"' % Path('.').absolute())
path = Path('../input')
train = pd.read_csv(path/'train.csv', header='infer')
test = pd.read_csv(path/'test.csv', header='infer')
class CustomImageList(ImageList):

    def open(self, fn):

        """Replace the original open method"""

        fn = fn.reshape(28,28)

        fn = np.stack((fn,)*3, axis=-1)

        return Image(pil2tensor(fn,dtype=np.float32))

    

    @classmethod

    def from_df_custom(cls, df, path:PathOrStr, **kwargs) ->'ItemList':

        res = super().from_df(df, path=path, cols=0, **kwargs)

        if 'label' in df.columns:

            df = df.drop('label', axis=1)

        df = np.array(df,dtype=np.float32)/255.

        df = df/255.

        mean = df.mean()

        std = df.std()

        res.items = (df-mean)/std

        return res
test_data = CustomImageList.from_df_custom(test,path=path)

test_data
tfms = get_transforms(do_flip=False)

data = (CustomImageList.from_df_custom(train, path=path)

        .split_by_rand_pct(.2, seed=2019)

        .label_from_df(cols='label')

        .add_test(test_data, label=0)

        .transform(tfms)

        .databunch(bs=128, num_workers=0)

        .normalize(imagenet_stats))

data
data.show_batch(2, figsize=(6,6))
learner = cnn_learner(data, models.resnet50,metrics=accuracy,model_dir='/kaggle/working/models')
learner.summary()
learner.lr_find()
learner.recorder.plot()
learner.fit_one_cycle(7, 1e-2)
learner.unfreeze()
learner.lr_find()

learner.recorder.plot()
learner.fit_one_cycle(8, slice(1e-5,1e-4))
interp = ClassificationInterpretation.from_learner(learner)

interp.plot_top_losses(9, figsize=(7,7))
learner.save('stage1')
learner.recorder.plot_losses()
learner.lr_find()

learner.recorder.plot()
learner.unfreeze()

learner.fit_one_cycle(8, slice(1e-5,1e-4))
learner.recorder.plot_losses()
learner.save('MnistStage2')
learner.show_results()
pred, y, losses = learner.get_preds(ds_type=DatasetType.Test, with_loss=True)
labels = torch.argmax(pred, dim=1)
submission_df = pd.DataFrame({'ImageId': range(1, len(y) + 1), 'Label': labels}, columns=['ImageId', 'Label'])

submission_df.head()
submission_df.to_csv('MnistSubmission.csv', index=False)
!head MnistSubmission.csv