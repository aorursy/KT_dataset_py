# Python 3 environment defined by the kaggle/python docker image: https://github.com/kaggle/docker-python



from datetime import datetime

from time import time



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from tqdm import tqdm



import matplotlib

import matplotlib.pyplot as plt

from matplotlib.lines import Line2D



import warnings



import fastai

from fastai.vision import *

from fastai.callbacks import *

from fastai.callbacks.hooks import *

from fastai.utils.mem import *
from fastai.utils.show_install import show_install; show_install()
!nvidia-smi
def fmt_now():

    return datetime.today().strftime('%Y%m%d-%H%M%S')
path = Path("/kaggle/input/understanding_cloud_organization")

path.ls()
path_img = path/"train_images"



fnames_train = get_image_files(path_img)

fnames_train[:3]

print(len(fnames_train))
path_test = path/"test_images"



fnames_test = get_image_files(path_test)

fnames_test[:3]

print(len(fnames_test))
img_f = fnames_train[1]

img = open_image(img_f)

img.show(figsize=(10, 10))
def split_img_label(img_lbl):

    """Return image and label from file name like '0011165.jpg_Flower'"""

    s = img_lbl.split("_")

    assert len(s) == 2

    return s[0], s[1]
train = pd.read_csv(f'{path}/train.csv')



# split Image_Label

train["Image"] = train["Image_Label"].apply(lambda img_lbl: split_img_label(img_lbl)[0])

train["Label"] = train["Image_Label"].apply(lambda img_lbl: split_img_label(img_lbl)[1])

del train["Image_Label"]



train.head()
train_with_mask = train.dropna(subset=["EncodedPixels"])

ax = train_with_mask["Label"].value_counts().plot(kind="pie", autopct='%1.1f%%', title="Shares of each classes", figsize=(10, 6))
class_counts = train.dropna(subset=["EncodedPixels"]).groupby("Image")["Label"].nunique()

ax = class_counts.plot(kind="hist", title="Number of classes per image")
# pivot to have one row per image and masks as columns

train = train.pivot(index='Image', columns='Label', values='EncodedPixels')

assert len(train) == len(fnames_train) # sanity check

train.head()
def show_img_fn(fname, figsize=(10, 10)):

    img = open_image(fname)

    img.show(figsize=figsize)    
def show_img_info(fname):

    show_img_fn(path_img/fname)

    display(train.loc[[fname]])   
unusual_imgs = ["1588d4c.jpg", "c0306e5.jpg", "c26c635.jpg", "fa645da.jpg", "41f92e5.jpg", "e5f2f24.jpg"]
for fname in unusual_imgs:

    img = open_image(path_img/fname)

    img.show(figsize=(5, 5), title=fname)     
train_img_dims = (1400, 2100)  # Train and test images are 1400x2100 pixels
def rle_to_mask(rle, shape):

    mask_img = open_mask_rle(rle, shape)

    mask = mask_img.px.permute(0, 2, 1)

    return mask
def mask_to_rle(mask):

    """Convert binary `mask` to RLE string"""

    return rle_encode(mask.numpy().T)
def test_mask_rle():

    """test case for mask RLE encode/decode"""

    mask_rle = train.iloc[0]["Fish"]    

    mask = rle_to_mask(mask_rle, train_img_dims)

    mask_rle_enc = mask_to_rle(mask)

    assert mask_rle_enc == mask_rle

    

    print(mask.shape)

    Image(mask).show()

    

test_mask_rle()
# TODO remove use_partial_data()

item_list = (SegmentationItemList.

             from_df(df=train.reset_index(), path=path_img, cols="Image")

             .use_partial_data(sample_pct=0.1)  # use only a subset of data to speedup tests

             .split_by_rand_pct(0.2))
class MultiLabelImageSegment(ImageSegment):

    """Store overlapping masks in separate image channels"""



    def show(self, ax:plt.Axes=None, figsize:tuple=(3,3), title:Optional[str]=None, hide_axis:bool=True,

        cmap:str='tab20', alpha:float=0.5, class_names=None, **kwargs):

        "Show the masks on `ax`."

             

        # put all masks into a single channel

        flat_masks = self.px[0:1, :, :].clone()

        for idx in range(1, self.shape[0]): # shape CxHxW

            mask = self.px[idx:idx+1, :, :] # slice tensor to a single mask channel

            # use powers of two for class codes to keep them distinguishable after sum 

            flat_masks += mask * 2**idx

        

        # use same color normalization in image and legend

        norm = matplotlib.colors.Normalize(vmin=0, vmax=2**self.shape[0]-1)

        ax = show_image(Image(flat_masks), ax=ax, hide_axis=hide_axis, cmap=cmap, norm=norm,

                        figsize=figsize, interpolation='nearest', alpha=alpha, **kwargs)

        

        # custom legend, see https://matplotlib.org/3.1.1/gallery/text_labels_and_annotations/custom_legends.html

        cm = matplotlib.cm.get_cmap(cmap)

        legend_elements = []

        for idx in range(self.shape[0]):

            c = 2**idx

            label = class_names[idx] if class_names is not None else f"class {idx}"

            line = Line2D([0], [0], color=cm(norm(c)), label=label, lw=4)

            legend_elements.append(line)

        ax.legend(handles=legend_elements)

        

        # debug info

        # ax.text(10, 10, f"px={self.px.size()}", {"color": "white"})

        

        if title: ax.set_title(title)



    def reconstruct(self, t:Tensor): 

        return MultiClassImageSegment(t)
# source: https://forums.fast.ai/t/unet-how-to-get-4-channel-output/54674/4

def bce_logits_floatify(input, target, reduction='mean'):

    return F.binary_cross_entropy_with_logits(input, target.float(), reduction=reduction)
class MultiLabelSegmentationLabelList(SegmentationLabelList):

    """Return a single image segment with all classes"""

    # adapted from https://forums.fast.ai/t/how-to-load-multiple-classes-of-rle-strings-from-csv-severstal-steel-competition/51445/2

    

    def __init__(self, items:Iterator, src_img_size=None, classes:Collection=None, **kwargs):

        super().__init__(items=items, classes=classes, **kwargs)

        self.loss_func = bce_logits_floatify

        self.src_img_size = src_img_size

        # add attributes to copy by new() 

        self.copy_new += ["src_img_size"]

    

    def open(self, rles):        

        # load mask at full resolution

        masks = torch.zeros((len(self.classes), *self.src_img_size)) # shape CxHxW

        for i, rle in enumerate(rles):

            if isinstance(rle, str):  # filter out NaNs

                masks[i] = rle_to_mask(rle, self.src_img_size)

        return MultiLabelImageSegment(masks)

    

    def analyze_pred(self, pred, thresh:float=0.0):

        # binarize masks

        return (pred > thresh).float()

    

    def reconstruct(self, t:Tensor): 

        return MultiLabelImageSegment(t)
class_names = ["Fish", "Flower", "Gravel", "Sugar"]
def get_masks_rle(img):

    """Get RLE-encoded masks for this image"""

    img = img.split("/")[-1]  # get filename only

    return train.loc[img, class_names].to_list()
# reduce image size

# img_size = tuple(v // 16 for v in train_img_dims)

img_size = (84, 132)  # use multiple of 4

img_size
classes = [0, 1, 2, 3] # no need for a "void" class: if a pixel isn't in any mask, it is not labelled

item_list = item_list.label_from_func(func=get_masks_rle, label_cls=MultiLabelSegmentationLabelList, 

                                      classes=classes, src_img_size=train_img_dims)
# add unlabelled test images

# set empty RLE string as label to produce empty multi-label masks and allow reconstruct() and show()

item_list = item_list.add_test_folder(path_test, label="")
batch_size = 8



# TODO add data augmentation

tfms = ([], [])

# tfms = get_transforms()



item_list = item_list.transform(tfms, tfm_y=True, size=img_size)
data = (item_list

        .databunch(bs=batch_size)

        .normalize(imagenet_stats) # use same stats as pretrained model

       )  

assert data.test_ds is not None
data.show_batch(2, figsize=(15, 10), class_names=class_names)
# adapted from: https://www.kaggle.com/iafoss/unet34-dice-0-87

# can use sigmoid on the input too, in this case the threshold would be 0.5

def dice_metric(pred, targs, threshold=0):

    pred = (pred > threshold).float()

    targs = targs.float()  # make sure target is float too

    return 2.0 * (pred*targs).sum() / ((pred+targs).sum() + 1.0)
metrics = [dice_metric]



callback_fns = [

    # update a graph of learner stats and metrics after each epoch

    ShowGraph,



    # save model at every metric improvement

    partial(SaveModelCallback, every='improvement', monitor='dice_metric', name=f"{fmt_now()}_unet_resnet18_stage1_best"),

    

    # stop training if metric no longer improve

    partial(EarlyStoppingCallback, monitor='dice_metric', min_delta=0.01, patience=2),

]



learn = unet_learner(data, models.resnet18, metrics=metrics, wd=1e-2, callback_fns=callback_fns)

learn.model_dir = "/kaggle/working/"  # point to writable directory
learn.loss_func
learn.summary()
learn.lr_find()

learn.recorder.plot()
defaults.device = torch.device('cuda')

learn.fit_one_cycle(5, max_lr=1e-4)
learn.recorder.plot_metrics()
learn.save(f"{fmt_now()}_unet_resnet18_stage1", return_path=True)
!ls -lth {learn.model_dir}
# learn = learn.load(Path(learn.model_dir)/"20190924-095959_unet_resnet18_stage1_best")
learn.unfreeze()
learn.lr_find()

learn.recorder.plot()
# slice(start,end) syntax: the first group's learning rate is start, the last is end, and the remaining are evenly geometrically spaced

learn.fit_one_cycle(15, max_lr=slice(1e-6, 1e-5))
learn.save(f"{fmt_now()}_unet_resnet18_stage2", return_path=True)
learn.show_results(imgsize=8, class_names=class_names)
preds, _ = learn.get_preds(ds_type=DatasetType.Test, with_loss=False)
preds.shape
learn.show_results(ds_type=DatasetType.Test, imgsize=8, class_names=class_names)
for i in range(3):

    pimg = MultiLabelImageSegment(preds[i] > 0)

    pimg.show(figsize=(6, 6), class_names=class_names)   
def resize_pred_masks(preds, shape=(4, 350, 525)):

    """Resize predicted masks and return them as a generator"""

    for p in range(preds.shape[0]):

        mask = MultiLabelImageSegment(preds[p])

        yield mask.resize(shape)
pred_masks = resize_pred_masks(preds)
test_fnames = [p.name for p in data.test_dl.items]

len(test_fnames)
def write_submission_file(filename, test_fnames, preds, threshold=0):

    with open(filename, mode='w') as f:

        f.write("Image_Label,EncodedPixels\n")



        for img_name, masks in zip(tqdm(test_fnames), resize_pred_masks(preds)):

            binary_masks = masks.px > threshold # TODO use activation instead

            

            for class_idx, class_name in enumerate(class_names):

                rle = mask_to_rle(binary_masks[class_idx])

                f.write(f"{img_name}_{class_name},{rle}\n")



    print(f"Wrote '{f.name}'.")
submission_file = f"{fmt_now()}_submission.csv"
write_submission_file(submission_file, test_fnames, preds)