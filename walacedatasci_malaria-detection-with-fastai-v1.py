%reload_ext autoreload

%autoreload 2

%matplotlib inline
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from fastai import *

from fastai.vision import *

from fastai.callbacks.hooks import *



import os

print(os.listdir("../input/cell_images/cell_images/"))
img_dir='../input/cell_images/cell_images/'
path=Path(img_dir)

path
data = ImageDataBunch.from_folder(path, train=".", 

                                  valid_pct=0.2,

                                  ds_tfms=get_transforms(flip_vert=True, max_warp=0),

                                  size=224,bs=64, 

                                  num_workers=0).normalize(imagenet_stats)
print(f'Classes: \n {data.classes}')
data.show_batch(rows=3, figsize=(7,6))
learn = create_cnn(data, models.resnet34, metrics=accuracy, model_dir="/tmp/model/")

learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(6,1e-2)
learn.save('stage-1')
learn.unfreeze()
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(4, max_lr=slice(5e-6,5e-5 ))
learn.save('stage-2')
learn.recorder.plot_losses()
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(9, figsize=(15,11))
interp.plot_confusion_matrix(figsize=(8,8), dpi=60)
interp.most_confused(min_val=2)
pred_data= ImageDataBunch.from_folder(path, train=".", 

                                  valid_pct=0.2,

                                  ds_tfms=get_transforms(flip_vert=True, max_warp=0),

                                  size=224,bs=64, 

                                  num_workers=0).normalize(imagenet_stats)
predictor=create_cnn(data, models.resnet34, metrics=accuracy, model_dir="/tmp/model/").load('stage-2')
pred_data.single_from_classes(path, pred_data.classes)
x,y = data.valid_ds[3]

x.show()

data.valid_ds.y[3]
pred_class,pred_idx,outputs = predictor.predict(x)

pred_class
def heatMap(x,y,data, learner, size=(0,224,224,0)):

    """HeatMap"""

    

    # Evaluation mode

    m=learner.model.eval()

    

    # Denormalize the image

    xb,_ = data.one_item(x)

    xb_im = Image(data.denorm(xb)[0])

    xb = xb.cuda()

    

    # hook the activations

    with hook_output(m[0]) as hook_a: 

        with hook_output(m[0], grad=True) as hook_g:

            preds = m(xb)

            preds[0,int(y)].backward()



    # Activations    

    acts=hook_a.stored[0].cpu()

    

    # Avg of the activations

    avg_acts=acts.mean(0)

    

    # Show HeatMap

    _,ax = plt.subplots()

    xb_im.show(ax)

    ax.imshow(avg_acts, alpha=0.5, extent=size,

              interpolation='bilinear', cmap='magma')

    
heatMap(x,y,pred_data,learn)