from fastai import *

from fastai.vision import *

import fastai

fastai.__version__
path = Path('../input')

path.ls()
labs = pd.read_csv(path/'train.csv')

labs.head()
labs.has_cactus.count()
labs['has_cactus'].value_counts().plot(kind='bar')
data = (ImageList.from_csv(path, 'train.csv', folder='train/train')

        .random_split_by_pct()

        .label_from_df()

        .add_test_folder('test/test')

        .transform(get_transforms(max_rotate=0, max_lighting=0.1, max_zoom=1, flip_vert=True), size=32)

        .databunch(bs=256))
data.show_batch(5, figsize=(6,6))
# https://www.kaggle.com/guntherthepenguin/fastai-v1-densenet169

from sklearn.metrics import roc_auc_score



def auc_score(y_pred,y_true,tens=True):

    score=roc_auc_score(y_true,torch.sigmoid(y_pred)[:,1])

    if tens:

        score=tensor(score)

    else:

        score=score

    return score
learn = create_cnn(data, models.resnet50, path=".", metrics=[accuracy, auc_score])
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(5, 1e-2)
! pip install objgraph xdot -q

import objgraph

objgraph.show_refs([learn])
learn.save('model-resnet')

print(Path('./models/model-resnet.pth').stat().st_size//(1024*1024), 'MB')
# Export the model for inference

learn.export()
!pip install git+https://github.com/FrancescAlted/ipython_memwatcher -q
from ipython_memwatcher import MemWatcher

mw = MemWatcher()

mw.start_watching_memory()
learn = load_learner("")
img = open_image(path/'train/train'/labs.id.iloc[0])
learn.predict(img)
print(mw.measurements)

mw.stop_watching_memory()
!pip install git+https://github.com/Tom2718/pytorch-OpCounter -q --upgrade
from thop import profile
model = learn.model

flops, params = profile(model, input_size=(1, 3, 32,32), 

                        custom_ops={Flatten: None})



print('FLOPs:', flops//1e6, 'M')

print('Params:', params//1e6, 'M')
learn = create_cnn(data, models.squeezenet1_1, path=".", metrics=[accuracy, auc_score])
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(10, 5e-2)
learn.unfreeze()

learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(7, slice(5e-4, 5e-3))
learn.save('model-squeezenet')

print(Path('./models/model-squeezenet.pth').stat().st_size//(1024*1024), 'MB')
learn.export()

mw.start_watching_memory()
learn = load_learner("")
img = open_image(path/'train/train'/labs.id.iloc[0])
learn.predict(img)
print(mw.measurements)

mw.stop_watching_memory()
model = learn.model

flops, params = profile(model, input_size=(1, 3, 32,32), custom_ops={Flatten: None})



print('FLOPs:', flops//1e6, 'M')

print('Params:', params//1e6, 'M')
learn = create_cnn(data, models.resnet18, path=".", metrics=[accuracy, auc_score])
learn.fit_one_cycle(10, 5e-2)
learn.save('model-resnet18')

print(Path('./models/model-resnet18.pth').stat().st_size//(1024*1024), 'MB')
learn.export()

mw.start_watching_memory()
learn = load_learner("")
img = open_image(path/'train/train'/labs.id.iloc[0])
learn.predict(img)
print(mw.measurements)

mw.stop_watching_memory()
model = learn.model

flops, params = profile(model, input_size=(1, 3, 32,32), custom_ops={Flatten: None})



print('FLOPs:', flops//1e6, 'M')

print('Params:', params//1e6, 'M')
plt.plot([14,48,114], [4,38,88])

plt.xlabel('Model Size (MB)')

plt.ylabel('FLOPs (M)')

plt.title('Model Size vs FLOPs')