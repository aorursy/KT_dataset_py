from fastai import *

from fastai.vision import *

from fastai.callbacks.hooks import *
path = Path('../input/dataset/DATASET')

path.ls()
data = ImageDataBunch.from_folder(

    path,

    train = "TRAIN",

    valid = "TEST",

    ds_tfms=get_transforms(do_flip=False),

    size = 128,

    bs=32,

    valid_pct=0.2,

    num_workers=0

).normalize(imagenet_stats)

print(f'Classes: \n {data.classes}')

data.show_batch(rows=10,figsize=(10,10))
learn = create_cnn(data, models.resnet50,metrics=accuracy,model_dir='/tmp/model/')

learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(5)
learn.recorder.plot_losses()
inter = ClassificationInterpretation.from_learner(learn)

inter.plot_top_losses(9,figsize=(20,20))
inter.plot_confusion_matrix(figsize=(10,10))
learn.unfreeze()

learn.fit_one_cycle(2)
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(5, max_lr=slice(1e-6,1e-4))
learn.recorder.plot_losses()
inter = ClassificationInterpretation.from_learner(learn)

inter.plot_top_losses(9,figsize=(20,20))
inter.plot_confusion_matrix(figsize=(10,10),dpi=75)

learn.save('waste-clf-fastai-V1')