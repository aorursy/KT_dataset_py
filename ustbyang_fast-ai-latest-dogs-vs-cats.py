%reload_ext autoreload

%autoreload 2

%matplotlib inline
import fastai

from fastai.imports import *

from fastai.vision import *

from fastai.metrics import *

from fastai.gen_doc.nbdoc import *

print('fast.ai version:{}'.format(fastai.__version__))
import os

WORK_DIR = os.getcwd()

WORK_DIR
path = Path('../input/')

path.ls()
path_img = path/'train'

fnames = get_image_files(path_img)

fnames[:5]
np.random.seed(2)

pat = r"([a-z]+).\d+.jpg$"
import re

prog = re.compile(pat)

result = prog.search('../input/train/dog.7504.jpg')

result[0]
show_doc(get_transforms)
batch_size = 64

tfms = get_transforms()

data = ImageDataBunch.from_name_re(path, fnames, pat, ds_tfms=tfms, test='test', size=224, bs=batch_size)

data = data.normalize(imagenet_stats)

data
data.show_batch(rows=4, figsize=(7, 6))
print(data.classes)

len(data.classes)
model = models.resnet152

learn = cnn_learner(data, model, metrics=accuracy, model_dir=WORK_DIR)

learn.model
learn.fit_one_cycle(2)
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_top_losses(9, figsize=(15, 11))
interp.plot_confusion_matrix(figsize=(4,4))
learn.recorder.plot_losses()
learn.recorder.plot_lr(show_moms=True)
learn.save('stage-1')
learn.unfreeze()

learn.lr_find()
learn.recorder.plot()
learn.load('stage-1')

learn.unfreeze()

learn.fit_one_cycle(4, max_lr=slice(1e-6, 1e-4))
learn.recorder.plot_losses()
learn.save('stage-2')
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_top_losses(9, figsize=(15, 11))
interp.plot_confusion_matrix(figsize=(4,4))
try:

    # TTA cause error @fast.ai v1.0.39

    preds, _ = learn.TTA(ds_type=DatasetType.Test)

except:

    preds, _ = learn.get_preds(DatasetType.Test)

else:

    print('Predict with TTA done.')
preds
# from here we know that 'cats' is label 0 and 'dogs' is label 1.

print(data.classes)

dict_label_order = {label:order for order,label in enumerate(data.classes)}

print(dict_label_order)
n_dogs = dict_label_order['dog']

n_dogs
prob_dogs = preds[:,n_dogs].numpy()

prob_dogs
plt.hist(prob_dogs)
ids = [int(file.stem) for file in data.test_ds.x.items]

ids[:10]
import pandas as pd

submission = pd.DataFrame({'id':ids,'label':prob_dogs})

submission = submission.sort_values(by=['id'])
submission.head()
submission.label.describe()
submission.to_csv('submission.csv', index=False)