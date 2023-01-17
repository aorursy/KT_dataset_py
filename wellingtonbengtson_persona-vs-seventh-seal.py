from fastai import *

from fastai.vision import *
classes = ['seventh_seal','persona']
folder = 'seventh_seal'

file = 'seventhseal.csv'
path = Path('data/movies')

dest = path/folder

dest.mkdir(parents=True, exist_ok=True)
!cp ../input/* {path}/
download_images(path/file, dest, max_pics=200)
folder = 'persona'

file = 'persona.csv'
path = Path('data/movies')

dest = path/folder

dest.mkdir(parents=True, exist_ok=True)
download_images(path/file, dest, max_pics=200)
for c in classes:

    print(c)

    verify_images(path/c, delete=True, max_size=500)
np.random.seed(42)

data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,

         ds_tfms=get_transforms(), size=64, num_workers=0).normalize(imagenet_stats)
data.classes
data.show_batch(rows=3, figsize=(3,4))
learn = create_cnn(data, models.resnet34, metrics=error_rate)
learn.fit_one_cycle(4)
learn.save('stage-1')
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(2, max_lr=slice(1e-4,1e-3))
learn.save('stage-2')
learn.load('stage-2');
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()