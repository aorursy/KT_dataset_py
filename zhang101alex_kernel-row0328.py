!pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# %reload_ext autoreload

# %autoreload 2

%matplotlib inline

import skimage

from skimage import io,data,data_dir

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pycocotools import coco, cocoeval, _mask

from pycocotools import mask as maskUtils 

from fastai import *

from fastai.vision import *

from fastai.callbacks.hooks import *

from fastai.metrics import error_rate

import os

import os.path 

import shutil 

print(os.listdir("./"))

print(os.listdir("../input/"))

print(os.listdir("../"))





# Any results you write to the current directory are saved as output.
ROOT_PATH="../input/rowsall/" 

print(os.listdir(ROOT_PATH))
pathimg_train=pathlib.Path(ROOT_PATH+'allrows/img')

#pathimg_val=pathlib.Path(ROOT_PATH+'valminus')

pathimg_test=pathlib.Path(ROOT_PATH+'allrows/testimg')

pathmask_train=pathlib.Path(ROOT_PATH+'allrows/train.json')

#pathmask_val=pathlib.Path(ROOT_PATH+'annotations/instances_valminusminival2014.json')

#pathmask_test=pathlib.Path(ROOT_PATH+'allrows/test.json')

pathimg_train,pathmask_train
IMG_COUNT_LIMIT=len(os.listdir(pathimg_train))

MASK_PATH='../input/rowsall/allrows/masks'

IMG_PATH='../input/rowsall/allrows/img'
fnames = get_image_files(pathimg_train)

#fnames[:3]
# lbl_names = get_image_files(pathmask_train)

# #lbl_names[:3]

CATEGORY_NAMES=[0,'row']

ANNOTATION_FILE=ROOT_PATH+'allrows/train.json'#其实也是pathmask_train的str格式，为方便与资料网站对应
coco = coco.COCO(ANNOTATION_FILE)

#coco

catIds = coco.getCatIds(catNms=CATEGORY_NAMES);

imgIds = coco.getImgIds(catIds=catIds);

imgDict = coco.loadImgs(imgIds)

len(imgIds) , len(catIds)
imgDF = pd.DataFrame.from_dict(imgDict)

#imgDF[:3]



def createImageForMask(file_path):

    file_name = str(file_path).split("/")[-1]

    out_data= imgDF[imgDF['file_name']==file_name]

    index= int(out_data['id'])

    sampleImgIds = coco.getImgIds(imgIds = [index])

    sampleImgDict = coco.loadImgs(index)[0]

    annIds = coco.getAnnIds(imgIds=sampleImgDict['id'], catIds=catIds, iscrowd=None)

    anns = coco.loadAnns(annIds)

    mask = coco.annToMask(anns[0])

    for i in range(len(anns)):

        mask = mask | coco.annToMask(anns[i])

    img=Image(pil2tensor(mask, dtype=np.float32))

    img.save(MASK_PATH+'/'+file_name)

    return MASK_PATH+'/'+file_name



ID=10

sampleImgIds = coco.getImgIds(imgIds = [ID])

sampleImgDict = coco.loadImgs(sampleImgIds[np.random.randint(0,len(sampleImgIds))])[0]

sampleImgDict



coco_url=ROOT_PATH+'allrows/img/'+sampleImgDict['file_name']

coco_url
ID=10



sampleImgIds = coco.getImgIds(imgIds = [ID])

sampleImgDict = coco.loadImgs(sampleImgIds[np.random.randint(0,len(sampleImgIds))])[0]

I = skimage.io.imread(coco_url)

plt.imshow(I); plt.axis('off')

annIds = coco.getAnnIds(imgIds=sampleImgDict['id'], catIds=catIds, iscrowd=0)

anns = coco.loadAnns(annIds)

coco.showAnns(anns)
mask = coco.annToMask(anns[0])

for i in range(len(anns)):

    mask = mask | coco.annToMask(anns[i])

plt.imshow(mask) ; plt.axis('off')



pixVals = set()

for pixRow in mask:

    for pix in pixRow:

        pixVals.add(pix)

print(pixVals)
src_size = np.array(mask.shape)

src_size
SZ=src_size//3
def get_y_fn(file_path):

    file_name = str(file_path).split("/")[-1]

    return MASK_PATH+'/'+file_name
class MySegmentationLabelList(SegmentationLabelList):

      def open(self, fn): return open_mask(fn, div=True)





class MySegmentationItemList(SegmentationItemList):

    _label_cls = MySegmentationLabelList  #



# tfms = get_transforms(max_rotate=25)

# len(tfms)



BS=2

TARGET_CLASSES=array([0,'row'])

src = (MySegmentationItemList.from_folder(IMG_PATH)

        .random_split_by_pct(.15)

        .label_from_func(get_y_fn , classes=TARGET_CLASSES))


tfms = get_transforms()

np.random.seed(23)

data = (src.transform(tfms, size=SZ, tfm_y=True)

        .databunch(bs=BS)   #, num_workers=NUM_WORKERS

        .normalize(imagenet_stats))

data.show_batch(rows=2, figsize=(9,12))
def dice(pred, targs):

    pred = (pred>0).float()

    return 2.0 * (pred*targs).sum() / ((pred+targs).sum() + 1.0)



def IoU(input:Tensor, targs:Tensor) -> Rank0Tensor:

    "IoU coefficient metric for binary target."

    n = targs.shape[0]

    input = input.argmax(dim=1).view(n,-1)

    targs = targs.view(n,-1)

    intersect = (input*targs).sum().float()

    union = (input+targs).sum().float()

    return intersect / (union-intersect+1.0)
learn = unet_learner(data, models.resnet34,wd=1e-2)  # metrics=metrics



learn.opt_fn=optim.Adam

learn.metrics=[partial(accuracy_thresh, thresh=0.5),IoU]  #dice,
lr=3e-5
learn.fit_one_cycle(10, slice(lr))
learn.model_dir="/kaggle/working/"
print(os.listdir("/"))
print(os.listdir("/kaggle/"))
learn.save('stage-01')
learn.unfreeze()
learn.load('stage-01')
lr_find(learn)

learn.recorder.plot()
lrs = slice((1e-6)/3,lr/5)
!cp /kaggle/input/rowsall/allrows/img/models/stage-1.pth  /kaggle/working/stage-01.pth
learn.load('stage-01');
learn.unfreeze()
lrs = slice((1e-6)/3,lr/5)
learn.fit_one_cycle(12, lrs)
learn.save('stage-2')
print(os.listdir("/kaggle/working/"))
!cp /kaggle/working/stage-2.pth /kaggle/input/rowsall/allrows/img/models/stage-2.pth
size = src_size

bs=1
data = (src.transform(get_transforms(), size=size, tfm_y=True)

        .databunch(bs=bs)

        .normalize(imagenet_stats))
learn.load('stage-2');
lr_find(learn)

learn.recorder.plot()