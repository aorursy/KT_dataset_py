%reload_ext autoreload

%autoreload 2

%matplotlib inline



from fastai.vision import *

from fastai.metrics import error_rate
# bs: batch size

bs = 64
# download dataset

path = untar_data(URLs.PETS)

# lihat path

path
# lihat apa saja yang ada di path

path.ls()
path_anno = path/'annotations'

path_img = path/'images'
fnames = get_image_files(path_img)

fnames[:5]
# set random seed agar dapat diulangi kedepannya

np.random.seed(2)

pat = r'/([^/]+)_\d+.jpg$'
data = ImageDataBunch.from_name_re(path_img, 

                                   fnames, 

                                   pat, 

                                   ds_tfms=get_transforms(), 

                                   size=224, 

                                   bs=bs).normalize(imagenet_stats)
data.show_batch(rows=3, figsize=(7, 6))
print(data.c) # banyaknya kelas

print(data.classes) # nama nama kelas
learn = cnn_learner(data, models.resnet34, metrics=error_rate, model_dir='/tmp/models')
learn.model
learn.fit_one_cycle(4) # 4 epoch
# save model yang telah dilatih

learn.save('stage-1')
interp = ClassificationInterpretation.from_learner(learn)



losses, idxs = interp.top_losses()



len(data.valid_ds)==len(losses)==len(idxs)
# plot gambar apa saja yang memiliki loss yang besar

interp.plot_top_losses(9, figsize=(15, 11))
# plot confusion matrixnya

interp.plot_confusion_matrix(figsize=(12, 12), dpi=60)
# lihat klasifikasi apa yang paling banyak yang salah

interp.most_confused(min_val=2)
learn.unfreeze()
learn.fit_one_cycle(1)
# load model

learn.load('stage-1')

learn.lr_find()
# lihat hasil lr finder

learn.recorder.plot()
learn.unfreeze()

learn.fit_one_cycle(2, max_lr=slice(1e-6, 1e-4))
data = ImageDataBunch.from_name_re(path_img, 

                                   fnames, 

                                   pat, 

                                   ds_tfms=get_transforms(), 

                                   size=299, 

                                   bs=bs//2).normalize(imagenet_stats)
learn = cnn_learner(data, models.resnet50, metrics=[error_rate, accuracy], model_dir='/tmp/models')
# panggil lr finder

learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(8)
learn.save('stage-1-50')

learn.unfreeze()
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(3, max_lr=slice(1e-5, 1e-4))
interp = ClassificationInterpretation.from_learner(learn)

interp.most_confused(min_val=2)