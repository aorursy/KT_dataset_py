# Put these at the top of every notebook, to get automatic reloading and inline plotting
%reload_ext autoreload
%autoreload 2
%matplotlib inline
# This file contains all the main external libs we'll use
from fastai.imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *
#PATH = "data/dogscats/"
TMP_PATH = "/tmp/tmp"
MODEL_PATH = "/tmp/model/"
PATH = "../input/lego brick images/LEGO brick images/"
sz=224
torch.cuda.is_available()
torch.backends.cudnn.enabled
os.listdir(PATH)
os.listdir(f'{PATH}valid')
files = os.listdir(f'{PATH}valid/3673 Peg 2M')[:5]
files
img = plt.imread(f'{PATH}valid/3004 Brick 1x2/{files[1]}')
plt.imshow(img);
# Uncomment the below if you need to reset your precomputed activations
# shutil.rmtree(f'{PATH}tmp', ignore_errors=True)
arch=resnet34
data = ImageClassifierData.from_paths(PATH, tfms=tfms_from_model(arch, sz))
learn = ConvLearner.pretrained(arch, data, precompute=True, tmp_name=TMP_PATH, models_name=MODEL_PATH)
learn.fit(0.02, 3)
learn = ConvLearner.pretrained(arch, data, precompute=True, tmp_name=TMP_PATH, models_name=MODEL_PATH)
lrf=learn.lr_find()
learn.sched.plot_lr()
learn.sched.plot()
tfms = tfms_from_model(resnet34, sz, aug_tfms=transforms_top_down, max_zoom=1.1)
def get_augs():
    data = ImageClassifierData.from_paths(PATH, bs=2, tfms=tfms, num_workers=1)
    x,_ = next(iter(data.aug_dl))
    return data.trn_ds.denorm(x)[1]
ims = np.stack([get_augs() for i in range(6)])
plots(ims, rows=2)
data = ImageClassifierData.from_paths(PATH, tfms=tfms)
learn = ConvLearner.pretrained(arch, data, precompute=True, tmp_name=TMP_PATH, models_name=MODEL_PATH)
learn.precompute=False
learn.unfreeze()
lr=np.array([1e-4,1e-3,2e-1])
# On a GTX 1080 Ti this takes about 8 minutes
learn.fit(lr, 4, cycle_len=1, cycle_mult=2)
learn.sched.plot_lr()
learn.sched.plot_loss()
learn.save('224_all')
learn.load('224_all')
log_preds,y = learn.TTA()
probs = np.mean(np.exp(log_preds),0)
accuracy_np(probs, y)