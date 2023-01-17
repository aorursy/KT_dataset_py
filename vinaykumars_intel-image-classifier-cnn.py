!curl -s https://course.fast.ai/setup/colab | bash
%reload_ext autoreload

%autoreload 2

%matplotlib inline
import os

from fastai import *

from fastai.vision import *

from fastai.metrics import error_rate, accuracy
bs = 64

# bs = 16   # uncomment this line if you run out of memory even after clicking Kernel->Restart
base_path = Path(os.path.join("..", "input", "intel-image-classification"))

train_path = base_path/'seg_train'/'seg_train'

val_path = base_path/'seg_test'/'seg_test'

pred_path = base_path/'seg_pred'/'seg_pred'
np.random.seed(2)

tfms = get_transforms()

data = ImageDataBunch.from_folder(path = base_path,

                                 train = 'seg_train',

                                 valid = 'seg_test',

                                 test = 'seg_pred',

                                 ds_tfms = tfms,

                                 bs = bs,

                                 size = 224)

data.normalize(imagenet_stats)
data.show_batch(rows = 3)
learn = cnn_learner(data, models.resnet34, metrics = [error_rate, accuracy])

learn.model_dir = "/kaggle/working/"

learn.fit_one_cycle(3)
learn.save('resnet34-stage-1')
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(9)
interp.most_confused(min_val=2)
interp.plot_confusion_matrix()
learn.unfreeze()
learn.fit_one_cycle(1)
learn.load('resnet34-stage-1')

learn.lr_find()
learn.recorder.plot()
learn.unfreeze()

learn.fit_one_cycle(3, max_lr = slice(1e-6,1e-4))