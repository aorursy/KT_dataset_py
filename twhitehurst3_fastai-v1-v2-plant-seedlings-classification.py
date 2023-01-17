from fastai import *

from fastai.vision import * 
path = '../input/v2-plant-seedlings-dataset'

data = ImageDataBunch.from_folder(path,

                                 ds_tfms=get_transforms(do_flip=False),

                                 size = 224,

                                 bs=28,

                                 valid_pct = 0.2).normalize(imagenet_stats)

data.show_batch(rows=3,figsize=(7,6))
learner = cnn_learner(data,models.resnet34,metrics=accuracy,model_dir='/tmp/model/')
learner.fit_one_cycle(3)
learner.save('/kaggle/working/stage-1')

learner.unfreeze()
learner.lr_find()

learner.recorder.plot()
learner.fit_one_cycle(5, max_lr=slice(1e-6,1e-4))
interp = ClassificationInterpretation.from_learner(learner)

losses,idxs = interp.top_losses()

interp.plot_confusion_matrix(figsize=(5,5), dpi=120)

interp.plot_top_losses(9, figsize=(15,11))