from fastai.vision import *

from fastai.metrics import error_rate
bs = 64
help(untar_data)
path = untar_data(URLs.PETS)

path_anno = path/'annotations'

path_img = path/'images'

fnames = get_image_files(path_img)
print("Total number of images",len(fnames))
np.random.seed(2)

pat = r'/([^/]+)_\d+.jpg$'
fnames[:5]
data = ImageDataBunch.from_name_re(path_img,fnames,pat,ds_tfms=get_transforms(),size = 224,bs=bs)

data.normalize(imagenet_stats)
data.show_batch(rows=3,figsize=(7,6))
print(data.classes)
len(data.classes)
data.c
learn  = cnn_learner(data,models.resnet34,metrics=error_rate)

learn.model
learn.fit_one_cycle(4)
learn.save('stage-1')