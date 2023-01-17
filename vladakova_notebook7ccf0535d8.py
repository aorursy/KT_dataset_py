!ls ../input/mycars/cars
from fastai.vision import *
path = '../input/mycars/cars'
data = ImageDataBunch.from_folder(path, train=path, valid_pct=0.2,
                                  ds_tfms=get_transforms(), size=224, num_workers=0).normalize(imagenet_stats)
learn = cnn_learner(data, models.resnet18, metrics=accuracy)
learn.fit(1)
