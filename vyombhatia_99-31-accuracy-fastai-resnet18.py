from fastai.vision import *
from fastai.metrics import *
path = "../input/fruits/fruits-360"
data = ImageDataBunch.from_folder(path, ds_tfms = get_transforms(),
                                  train='Training', valid='Test',
                                  size=360, bs=16)
data.show_batch(4)
learn = cnn_learner(data, models.resnet18, metrics = accuracy, callback_fns = ShowGraph)
learn.fit_one_cycle(2)
img = open_image("../input/fruits/fruits-360/test-multiple_fruits/Bananas(lady_finger)4.jpg")

img.show(figsize=(6,6))
imgclass, y, idx = learn.predict(img)
print(imgclass)
