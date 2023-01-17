from fastai import *

from fastai.vision import *

from fastai.callbacks.hooks import *
path = Path('../input/heroes/heroes/')

path.ls()
data = ImageDataBunch.from_folder(

    path,

    train = '.',

    valid_pct = 0.1,

    ds_tfms=get_transforms(max_warp=0,flip_vert=True,do_flip=True),

    size = 128,

    bs=16

).normalize(imagenet_stats)

print(f'Classes: \n {data.classes}')

data.show_batch(rows=8,figsize=(10,10))
learn = create_cnn(data,models.resnet50,metrics=accuracy,model_dir='/tmp/model/')

learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(5)
learn.recorder.plot_losses()
learn.save('overwatch-stage-1')
learn.unfreeze()

learn.fit_one_cycle(2)
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(10,max_lr=slice(1e-6,1e-4))
learn.recorder.plot_losses()
inter = ClassificationInterpretation.from_learner(learn)

inter.plot_top_losses(10,figsize=(20,20))
inter.plot_confusion_matrix(figsize=(10,10))