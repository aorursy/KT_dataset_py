%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai.vision import *

from fastai.metrics import error_rate
monkey_images_path = '../input/10-monkey-species/'

tfms = get_transforms()

data = ImageDataBunch.from_folder(monkey_images_path, train='training', valid='validation', ds_tfms=tfms, size=128)
print(data.classes)

len(data.classes),data.c
data.show_batch()
!mkdir -p /tmp/.torch/models/

!cp ../input/resnet34/resnet34.pth /tmp/.torch/models/resnet34-333f7ec4.pth

!cp ../input/resnet50/resnet50.pth  /tmp/.torch/models/resnet50-19c8e357.pth
path_model='/kaggle/working/'

path_input='/kaggle/input/'

learn_resnet34 = cnn_learner(data, models.resnet34, metrics=error_rate, model_dir=f'{path_model}')
learn_resnet34.fit_one_cycle(4)
learn_resnet34.lr_find()

learn_resnet34.recorder.plot()
learn.unfreeze()

learn.fit_one_cycle(2, max_lr=slice(1e-03,1e-04))
learn_resnet50 = cnn_learner(data, models.resnet50, metrics=error_rate, model_dir=f'{path_model}')
learn_resnet50.fit_one_cycle(8)
interp_resnet50 = ClassificationInterpretation.from_learner(learn_resnet50)

interp_resnet50.most_confused(min_val=2)
interp_resnet38 = ClassificationInterpretation.from_learner(learn_resnet34)

interp_resnet38.most_confused(min_val=2)
interp_resnet38.plot_confusion_matrix(figsize=(12,12), dpi=60)
interp_resnet50.plot_confusion_matrix(figsize=(12,12), dpi=60)