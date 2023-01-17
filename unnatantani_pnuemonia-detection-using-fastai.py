%reload_ext autoreload
%autoreload 2
%matplotlib inline
from fastai.vision.all import *
from fastai.imports import *
from fastai.vision.data import *
from fastai import *
import numpy as np
import fastai
path = Path("/kaggle/input/chest-xray-pneumonia/chest_xray")
path.ls()

np.random.seed(42)
#data = ImageDataLoaders.from_folder(path, train=".", valid_pct=0.2, size=256, num_workers=4).normalize(imagenet_stats)
data = ImageDataLoaders.from_folder(path, train="train", valid="val", item_tfms=RandomResizedCrop(512, min_scale=0.75),
                                    bs=16,batch_tfms=[*aug_transforms(size=224, max_warp=0), Normalize.from_stats(*imagenet_stats)],num_workers=0)

data.show_batch(nrows=3, figsize=(7,8))
learn = cnn_learner(data, resnet50, metrics=error_rate)
learn.fit_one_cycle(4)
learn.export("/kaggle/working/Stage -1.pkl")
learn.unfreeze()
learn.fit_one_cycle(3, lr_max=slice(2e-6,2e-7))
learn.freeze()
learn.fit_one_cycle(2)
learn.model = learn.model.cpu()
learn.export('/kaggle/working/pneumonia.pkl')
defaults.device = torch.device('cpu')
t = Image.open('/kaggle/input/chest-xray-pneumonia/chest_xray/test/NORMAL/IM-0001-0001.jpeg')
t #Normal Xray
t = Image.open('/kaggle/input/chest-xray-pneumonia/chest_xray/test/PNEUMONIA/person100_bacteria_475.jpeg')
t #Pneu Case
l = load_learner('/kaggle/working/pneumonia.pkl', cpu=True)
defaults.device  = torch.device('cpu')
defaults.device
fnames_normal=get_image_files('/kaggle/input/chest-xray-pneumonia/chest_xray/test/NORMAL/')
fnames_pneu=get_image_files('/kaggle/input/chest-xray-pneumonia/chest_xray/test/PNEUMONIA/')

# fnames_normal
pred_class,pred_idx,outputs = l.predict(fnames_normal[0])
print("Actual: Normal, Predicted = {}".format(pred_class))
pred_class,pred_idx,outputs = l.predict(fnames_pneu[0])
print("Actual: Pneumonia, Predicted = {}".format(pred_class))
