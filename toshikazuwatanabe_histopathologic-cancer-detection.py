!ls ../input
import fastai

from fastai.imports import *

from fastai.vision import *

from fastai.metrics import *

from fastai.gen_doc.nbdoc import *

print('fast.ai version:{}'.format(fastai.__version__))
WORK_DIR = os.getcwd()
show_doc(get_transforms)
show_doc(ImageDataBunch.from_csv)
param_transforms = {

    'flip_vert':True,

    'max_rotate':90,

    'max_zoom':1,

    'max_warp':None,

    'max_lighting':None,

}

params = {'path': Path('../input/histopathologic-cancer-detection'),

          'csv_labels':'train_labels.csv',

          'ds_tfms':get_transforms(**param_transforms),

          'suffix':'.tif',

          'folder':'train',

          'test':'test',

          'size':32,

          'bs':128,

          'num_workers':0,

         }

data = ImageDataBunch.from_csv(**params).normalize(imagenet_stats)

data
data.show_batch(rows=3)
learn = create_cnn(data, models.resnet101, metrics=accuracy, path='.', model_dir='.')
!ls ../input/dataset
learn.load('../input/dataset/histopathologic_cancer_detection')
learn.unfreeze()

# learn.fit_one_cycle(16)

learn.fit_one_cycle(16,slice(1e-6,1e-4))
learn.save('histopathologic_cancer_detection')
learn.recorder.plot_losses()

plt.ylim([0,0.5])
# Check if DatasetType loaded

# It's depend on difference between fast.ai version v1.0.39/v1.0.36.

try:

    DatasetType

except:

    from fastai import DatasetType
try:

    # TTA cause error @fast.ai v1.0.39

    preds, _ = learn.TTA(ds_type=DatasetType.Test)

    print('Predict w/ test time augument done.')

except:

    preds, _ = learn.get_preds(DatasetType.Test)
preds
ids = [file.stem for file in data.test_ds.x.items]

ids[:5]
data.classes
df=pd.read_csv('../input/histopathologic-cancer-detection/sample_submission.csv').set_index('id')

df.head()
df.loc[ids,'label'] = preds[:,1].numpy()

df = df.reset_index()

df.head()
df.to_csv('submission.csv', index=False)