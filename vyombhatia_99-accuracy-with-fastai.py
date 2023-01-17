import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from fastai.vision import *
from fastai.metrics import accuracy
path = "../input/nepali-digit-28-by-28-images/nepali_digit_28_by_28_images"

data = ImageDataBunch.from_folder(path, ds_tfms=get_transforms(), 
                                 valid_pct=0.3, size=256,
                                 bs=16)
data.show_batch(row=4)
print(data.classes)
learn = cnn_learner(data, models.resnet34, metrics=accuracy, model_dir="/tmp/model/")
learn.model
learn.fit_one_cycle(6)
learn.lr_find()
learn.recorder.plot()
learn.save('stage-1')
learn.unfreeze()
learn.fit_one_cycle(3, max_lr=slice(1e-5, 1e-4))
image = open_image("../input/nepali-digit-28-by-28-images/nepali_digit_28_by_28_images/test_set/digit_7/020_04.jpg")
import matplotlib.pyplot as plt
plt.figure(figsize=(6,6))
image.show()
pred_class, pred_idx, pred_outputs = learn.predict(image)
print(pred_class)