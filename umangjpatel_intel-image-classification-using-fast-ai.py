%reload_ext autoreload

%autoreload 2

%matplotlib inline
import os

from fastai import *

from fastai.vision import *
base_path = Path(os.path.join("..", "input", "intel-image-classification"))

print(base_path.ls(), end="\n\n")



train_path = base_path/'seg_train'/'seg_train'

print(train_path.ls(), end="\n\n")

val_path = base_path/'seg_test'/'seg_test'

print(val_path.ls(),end="\n\n")

pred_path = base_path/'seg_pred'/'seg_pred'

print("No. of test images : {}".format(len(pred_path.ls())))
data = ImageDataBunch.from_folder(path = base_path,

                                 train = 'seg_train',

                                 valid = 'seg_test',

                                 test = 'seg_pred',

                                 seed = 42,

                                 ds_tfms = get_transforms(),

                                 bs = 32,

                                 size = 224)

data.normalize(imagenet_stats)
data.show_batch(rows = 3, figsize = (12,10))
learner = cnn_learner(data, models.resnet34, metrics=[accuracy, error_rate])

learner.model_dir = "/kaggle/working/"
learner.fit_one_cycle(5)
learner.save('resnet34-stage-1')
interp = ClassificationInterpretation.from_learner(learner)
interp.plot_top_losses(5, figsize=(12,10), heatmap = True)
interp.plot_confusion_matrix(figsize=(12,10), dpi=60)
interp.most_confused(min_val = 5)
learner.lr_find()

learner.recorder.plot(suggestion=True)
learner.unfreeze()

learner.fit_one_cycle(5, max_lr = slice(1e-6,1e-4))
learner.save('resnet34-stage-2')
interp = ClassificationInterpretation.from_learner(learner)
interp.plot_confusion_matrix(figsize = (12,10), dpi = 60)
interp.plot_top_losses(5, figsize = (15,11))
interp.most_confused(5)