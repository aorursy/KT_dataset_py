import fastai
from fastai.imports import *
from fastai.vision import *
from fastai.metrics import *
from fastai.gen_doc.nbdoc import *
print('fast.ai version:{}'.format(fastai.__version__))
model = models.resnet152#resnet34
WORK_DIR = os.getcwd()
IMAGE_DIR = Path('../input/')
image_size=224
batch_size=32
fnames = get_image_files(IMAGE_DIR/'train')
fnames[:5]
show_doc(get_transforms)
show_doc(ImageDataBunch.from_name_re)
pattern_get_class = re.compile(r'/([^/]+)\.\d+\.jpg$')
data = ImageDataBunch.from_name_re(path = IMAGE_DIR,
                                   fnames = fnames, 
                                   pat = pattern_get_class, 
                                   ds_tfms=get_transforms(), 
                                   test ='test',
                                   size=image_size, 
                                   bs=batch_size,
                                   num_workers=0).normalize(imagenet_stats)
data
data.show_batch(rows=3)
show_doc(create_cnn)
learn = create_cnn(data, model, metrics=accuracy, model_dir=WORK_DIR)
show_doc(learn.fit_one_cycle)
learn.fit_one_cycle(1)
learn.recorder.plot_losses()
learn.recorder.plot_lr(show_moms=True)
# Left plot is Learning-Rate vs itteration, Right one is momentams vs itteration.
learn.save('stage-1')
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()
learn.load('stage-1')
learn.unfreeze()
learn.fit_one_cycle(4, max_lr=slice(1e-6,1e-4))
learn.recorder.plot_losses()
learn.save('stage-2')
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(9)
interp.plot_confusion_matrix()
# Check if DatasetType loaded
# It's depend on difference between fast.ai version v1.0.39/v1.0.36.
if 'DatasetType' not in sys.modules:
    from fastai import DatasetType
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