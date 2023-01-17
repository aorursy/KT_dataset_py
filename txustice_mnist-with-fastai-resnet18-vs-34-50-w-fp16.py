%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai import *

from fastai.vision import *

from fastai.metrics import *

from fastai.data_block import *

import numpy as np

import pandas as pd



path = Path("../input/digit-recognizer")
class CustomImageList(ImageList):

    def open(self, fn):

        img = fn.reshape(28,28)

        img = np.stack((img,)*3, axis=-1)

        return Image(pil2tensor(img, dtype=np.float32))

    

    @classmethod

    def from_csv_custom(cls, path:PathOrStr, csv_name:str, imgIdx:int=1, header:str='infer', **kwargs)->'ItemList': 

        df = pd.read_csv(Path(path)/csv_name, header=header)

        res = super().from_df(df, path=path, cols=0, **kwargs)

        

        res.items = df.iloc[:,imgIdx:].apply(lambda x: x.values / 255.0, axis=1).values

        

        return res

    

    @classmethod

    def from_df_custom(cls, path:PathOrStr, df:DataFrame, imgIdx:int=1, header:str='infer', **kwargs)->'ItemList': 

        res = super().from_df(df, path=path, cols=0, **kwargs)

        

        res.items = df.iloc[:,imgIdx:].apply(lambda x: x.values / 255.0, axis=1).values

        

        return res
test = CustomImageList.from_csv_custom(path=path, csv_name="test.csv", imgIdx=0)

test
data = (CustomImageList.from_csv_custom(path=path, csv_name="train.csv", imgIdx=1)

        .split_by_rand_pct(0.2)

        .label_from_df(cols='label')

        .add_test(test, label=0)

        .transform(get_transforms(do_flip=False))

        .databunch(bs=128)

        .normalize(imagenet_stats))
data.show_batch(rows=3, figsize=(5,5))
learn = cnn_learner(data, models.resnet18, metrics=accuracy, model_dir="/kaggle/working/models")

learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(1, max_lr=1e-02)
learn.save('mnist-resnet18-1')
learn.load('mnist-resnet18-1')
learn.unfreeze()

learn.lr_find()

learn.recorder.plot(suggestion=True)
max_lr = 1e-6

learn.fit_one_cycle(10, max_lr=max_lr)
learn.save('mnist-resnet18-2')
learn = cnn_learner(data, models.resnet34, metrics=accuracy, model_dir="/kaggle/working/models").to_fp16()

learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(1, max_lr=1e-02)
learn.save('mnist-resnet34-fp16-1')
learn.unfreeze()

learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(6, max_lr=slice(1e-05, 1e-04))
learn.save('mnist-resnet34-fp16-2')
learn = cnn_learner(data, models.resnet50, metrics=accuracy, model_dir="/kaggle/working/models").to_fp16()

learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(1, max_lr=1e-02)
learn.save('mnist-resnet50-fp16-1')
learn.unfreeze()

learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(6, max_lr=slice(1e-06, 1e-05))
learn = cnn_learner(data, models.resnet34, metrics=accuracy, model_dir="/kaggle/working/models").to_fp16().load('mnist-resnet34-fp16-2')
predictions, *_ = learn.get_preds(DatasetType.Test)

labels = np.argmax(predictions, 1)

submission_df = pd.DataFrame({'ImageId': list(range(1,len(labels)+1)), 'Label': labels})

submission_df.to_csv(f'submission.csv', index=False)