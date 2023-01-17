from fastai import *

from fastai.vision import *
# Always check and update your fastai version

import fastai

fastai.__version__
path = Path('../input')
data = (ImageList.from_csv(path, 'labels.csv', folder='train')

        .random_split_by_pct()

        .label_from_df()

        .add_test_folder()

        .transform(get_transforms(), size=112)

        .databunch()

       )
data.show_batch(3)
learn = create_cnn(data, models.resnet18, path="", metrics=accuracy)
learn.fit_one_cycle(3,1e-2)
preds,_ = learn.TTA(ds_type=DatasetType.Test)

preds.shape
subm = pd.DataFrame(preds.numpy())

subm.columns = data.classes
fnames = os.listdir(path/'test')

subm.insert(0, 'id', fnames)
subm.head()
subm.to_csv('subm.csv', index=None)