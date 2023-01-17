from fastai.vision import *

from fastai import *
folder = 'aloevera'

file = 'urls_aloevera.txt'

path = Path('data/succulents')

dest = path/folder

dest.mkdir(parents=True, exist_ok=True)

classes = ['aloevera','jade','burro']

download_images("../input/urls_aloevera.txt", dest, max_pics=200)
folder = 'jade'

file = 'urls_jade.txt'

path = Path('data/succulents')

dest = path/folder

dest.mkdir(parents=True, exist_ok=True)

download_images("../input/urls_jade.txt", dest, max_pics=200)
folder = 'burro'

file = 'urls_burro.txt'

path = Path('data/succulents')

dest = path/folder

dest.mkdir(parents=True, exist_ok=True)

download_images("../input/urls_burro.txt", dest, max_pics=200)
os.listdir("../input/")

path.ls()
for c in classes:

    print(c)

    verify_images(path/c, delete=True, max_size=500)
np.random.seed(42)

data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,

        ds_tfms=get_transforms(), size=224, num_workers=0).normalize(imagenet_stats)

# If you already cleaned your data, run this cell instead of the one before

#np.random.seed(42)

#data = ImageDataBunch.from_csv(".", folder=".", valid_pct=0.2, csv_labels='cleaned.csv',        

#                               ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)
data.classes
data.show_batch(rows=2, figsize=(5,5))
data.classes, data.c, len(data.train_ds), len(data.valid_ds)
learn = create_cnn(data, models.resnet34, metrics=error_rate)
learn.fit_one_cycle(5)
learn.save('stage-1')
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_confusion_matrix()

interp.plot_top_losses(9, figsize=(7,8))
learn.unfreeze()
learn.fit_one_cycle(1)
learn.load('stage-1')
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(2, max_lr=slice(3e-5,3e-4))
learn.save('stage-2')
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_confusion_matrix()
img = open_image(path/'aloevera'/'00000050.jpg')

img.show(figsize=(4, 4))
pred_class,pred_idx,outputs = learn.predict(img)

pred_class
img = open_image(path/'burro'/'00000015.jpg')

img.show(figsize=(4, 4))
pred_class,pred_idx,outputs = learn.predict(img)

pred_class
img = open_image(path/'jade'/'00000001.jpg')

img.show(figsize=(4, 4))
pred_class,pred_idx,outputs = learn.predict(img)

pred_class
#learn.model = learn.model.cpu()

learn.export()
path.ls()
learn = load_learner(path)
defaults.device = torch.device('cpu')
img = open_image(path/'jade'/'00000002.jpg')

img.show(figsize=(4, 4))
pred_class,pred_idx,outputs = learn.predict(img)

pred_class
import shutil

shutil.rmtree("./data/succulents")