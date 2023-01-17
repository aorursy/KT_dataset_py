from fastai.vision import *
from fastai.metrics import *
path = "../input/100-bird-species"
data = ImageDataBunch.from_folder(path, size=224, bs=8, train='train',
                                  test='test', valid='valid', ds_tfms = get_transforms())
data.show_batch(3)
learner = cnn_learner(data, models.resnet18, metrics = accuracy)
learner.fit_one_cycle(10)
learner.fit_one_cycle(3, 1e-5, moms=(.99,.89))
imgpath = "../input/100-bird-species/test/AMERICAN COOT/2.jpg"
# Notice that its an American Coot

img = open_image(imgpath)
pred_class, pred_idx, y = learner.predict(img)

print(pred_class)
img.show(figsize=(5,5))
imgpath_2 = "../input/100-bird-species/test/BLACK SWAN/2.jpg"
# Notice, again, that its an American Coot

img_2 = open_image(imgpath_2)

pred_class, pred_idx, y = learner.predict(img_2)

print(pred_class)
img_2.show(figsize=(5,5))