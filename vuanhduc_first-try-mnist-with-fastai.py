# https://ipython.readthedocs.io/en/stable/config/extensions/autoreload.html

# enable 'autoreload' extension

%reload_ext autoreload



# Reload all modules (except those excluded by %aimport) every time before executing the Python code typed

%autoreload 2



#To enable the inline image from matplotlib to Notebook

%matplotlib inline
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from fastai.vision import *

from fastai.metrics import error_rate
!ls ../input

!mkdir -p ../working/.fastai/models/checkpoints



!cp ../input/fastai-pretrained-models/* ../working/.fastai/models/checkpoints
path_models=Path('/kaggle/working/.fastai/models'); path_models.ls()

class CustomImageItemList(ImageList):

    def open(self, fn):

        img = fn.reshape(28, 28)

        img = np.stack((img,)*3, axis=-1) # convert to 3 channels

        return Image(pil2tensor(img, dtype=np.float32))



    @classmethod

    def from_csv_custom(cls, path:PathOrStr, csv_name:str, imgIdx:int=1, header:str='infer', **kwargs) -> 'ItemList':

        df = pd.read_csv(Path(path)/csv_name, header=header)

        res = super().from_df(df, path=path, cols=0, **kwargs)

        # convert pixels to an ndarray

        res.items = df.iloc[:,imgIdx:].apply(lambda x: x.values / 783.0, axis=1).values

        return res
path_input = Path('../input/digit-recognizer'); path_input.ls()
bs=64
test = CustomImageItemList.from_csv_custom(path=path_input, csv_name='test.csv', imgIdx=0)

data = (CustomImageItemList.from_csv_custom(path=path_input, csv_name='train.csv')

                       .split_by_rand_pct(.2)

                       .label_from_df(cols='label')

                       .add_test(test, label=0)

                       .databunch(bs=bs, num_workers=0)

                       .normalize(imagenet_stats))
data.show_batch(rows=3, figsize=(12,9))

# Config.DEFAULT_CONFIG = { 

#     'data_path': '/tmp/fastai/data', 

#     'model_path': '/kaggle/working/.fastai/models' 

# } 

# Config.create('/tmp/myconfig.yml') 

# Config.DEFAULT_CONFIG_PATH = '/tmp/myconfig.yml'

os.environ['TORCH_HOME'] = "/kaggle/working/.fastai/models"
learner = cnn_learner(data, models.resnet50, model_dir=path_models)
learner.fit_one_cycle(1)

learner.lr_find()

learner.recorder.plot()
learner.unfreeze()

learner.fit_one_cycle(4, max_lr= slice(1e-3, 1e-2))
preds, y, losses = learner.get_preds(ds_type=DatasetType.Test, with_loss=True)
y = torch.argmax(preds, dim=1)

submission_df = pd.DataFrame({'ImageId': range(1, len(y) + 1), 'Label': y}, columns=['ImageId', 'Label'])

submission_df.head()
submission_df.to_csv('submission.csv', index=False)

!head submission.csv
from IPython.display import FileLink

FileLink('submission.csv')