from fastai.vision import *
path = '../input' # print(os.listdir(path))
np.random.seed(42)

data = ImageDataBunch.from_csv(path, folder='train', csv_labels='labels.csv', valid_pct=0.2, ds_tfms=get_transforms(), size=224, test='test', suffix='.jpg', bs=64).normalize(imagenet_stats)
data.show_batch(rows=3, figsize=(10,12))
data.classes
data.c, len(data.train_ds), len(data.valid_ds)
learn = cnn_learner(data, models.resnet50, metrics=accuracy)
learn.fit_one_cycle(8)
learn.export('/kaggle/working/dog-breed.pkl')
from IPython.display import FileLink

FileLink('dog-breed.pkl')