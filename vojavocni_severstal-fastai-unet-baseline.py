import numpy as np 

import pandas as pd



from fastai.vision import *

import seaborn as sns

import cv2
from pathlib import Path



shape = (1600, 256)



data_folder = Path('/kaggle/input/severstal-steel-defect-detection/')



train_data = pd.read_csv(data_folder / 'train.csv')

train_data = pd.concat([train_data, train_data.ImageId_ClassId.str.split('_', expand=True).rename(columns={0: 'ImageId', 1: 'ClassId'})], axis=1)



train_data.head()
train_df = train_data[['ImageId', 'ClassId', 'EncodedPixels']]

mask_map = train_df.set_index(['ImageId', 'ClassId'])



mask_map.head()
imgs_with_masks = train_data.ImageId.drop_duplicates().isin(train_data.dropna().ImageId.drop_duplicates()).sum()

total_images = train_data.ImageId.drop_duplicates().shape[0]



print(f'{imgs_with_masks}/{total_images} --- {imgs_with_masks / total_images * 100} %')

sns.distplot(train_data.dropna()[['ImageId', 'ClassId']].groupby('ImageId').count())
def mask2rle(img):

    '''

    img: numpy array, 1 - mask, 0 - background

    Returns run length as string formated

    '''

    pixels= img.T.flatten()

    pixels = np.concatenate([[0], pixels, [0]])

    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    

    runs[1::2] -= runs[::2]

    return ' '.join(str(x) for x in runs)

 

def rle2mask(mask_rle, shape=(1600,256)):

    '''

    mask_rle: run-length as string formated (start length)

    shape: (width,height) of array to return 

    Returns numpy array, 1 - mask, 0 - background



    '''

    s = mask_rle.split()

    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]

    starts -= 1

    ends = starts + lengths

    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)

    for lo, hi in zip(starts, ends):

        img[lo:hi] = 1

        

    return img.reshape(shape).T



def decode_image_mask(img_id, shape=(1600,256)):

    masks = []

    for cl in range(1,5):

        cl_str = str(cl)



        cl_rle = mask_map.loc[(img_id, cl_str)].values[0]



        if not isinstance(cl_rle, str):

            cl_mask = np.zeros(shape).T

        else:

            cl_mask = rle2mask(cl_rle, shape)



        cl_mask = cl_mask[np.newaxis,:,:] * cl

        masks.append(cl_mask)



    masks = np.concatenate(masks, axis=0)

    masks = np.sum(masks, axis=0)



    return masks



def encode_image_mask(mask_tensor, shape=(1600,256)):

    mask_array = np.array(mask_tensor)

    mask_array = mask_array.argmax(axis=0)  

        

    rles = []

    for i in range(1,5):

        mask = (mask_array == i).astype(np.uint8)

        mask = cv2.resize(mask, shape, cv2.INTER_NEAREST)

        

        rle = mask2rle(mask)

        rle = np.nan if rle == '' else rle

        rles.append(rle)

    

    return rles
import fastai



def transform(self, tfms:Optional[Tuple[TfmList,TfmList]]=(None,None), **kwargs):

    if not tfms: tfms=(None,None)

    assert is_listy(tfms) and len(tfms) == 2

    self.train.transform(tfms[0], **kwargs)

    self.valid.transform(tfms[1], **kwargs)

    kwargs['tfm_y'] = False # Test data has no labels

    if self.test: self.test.transform(tfms[1], **kwargs)

    return self



fastai.data_block.ItemLists.transform = transform



class ServerstalSegmentationLabelList(SegmentationLabelList):

    

    def open(self, image_id:str):

        mask = decode_image_mask(image_id, shape)[np.newaxis,:,:]

        mask = torch.tensor(mask, dtype=torch.float)

        

        return ImageSegment(mask)

    

    
from sklearn.model_selection import train_test_split



_, valid_ids = train_test_split(train_df.ImageId.drop_duplicates(), test_size=0.2)



train_df['is_valid'] = train_df.ImageId.isin(valid_ids)
bs = 16

size = (128,800)

data = (

    SegmentationItemList.from_df(train_df, data_folder, cols='ImageId', folder='train_images')

    .split_from_df('is_valid')

    .label_from_df('ImageId', label_cls=ServerstalSegmentationLabelList, classes=[0,1,2,3,4])

    .add_test_folder(data_folder / 'test_images', label=None)

    .transform(get_transforms(flip_vert=True), size=size, tfm_y=True)

    .databunch(bs=bs)

    .normalize(imagenet_stats)

)



data.show_batch(5)
import math

import torch

from torch.optim import Adam





class RAdam(Adam):

    

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,

                 weight_decay=0, amsgrad=False):

        super(RAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, 

                                   weight_decay=weight_decay)

        



    def step(self, closure=None):

        """Performs a single optimization step.



        Arguments:

            closure (callable, optional): A closure that reevaluates the model

                and returns the loss.

        """

        loss = None

        if closure is not None:

            loss = closure()



        for group in self.param_groups:

            for p in group['params']:

                if p.grad is None:

                    continue

                grad = p.grad.data

                if grad.is_sparse:

                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')



                state = self.state[p]



                beta1, beta2 = group['betas']

                

                # State initialization

                if len(state) == 0:

                    state['step'] = 0

                    # Exponential moving average of gradient values

                    state['exp_avg'] = torch.zeros_like(p.data)

                    # Exponential moving average of squared gradient values

                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                    # Max SMA

                    state['max_sma'] = 2/(1 - beta2) - 1



                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                

                state['step'] += 1



                if group['weight_decay'] != 0:

                    grad.add_(group['weight_decay'], p.data)

                

                max_sma = state['max_sma']

                

                # Decay the first and second moment running average coefficient

                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                

                bias_correction1 = 1 - beta1 ** state['step']

                

                beta2_t = beta2 ** state['step']

                approx_sma = max_sma - 2 * state['step'] * beta2_t / beta2_t

                

                if approx_sma > 4:

                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                    bias_correction2 = 1 - beta2 ** state['step']

                    var_rectification = math.sqrt(

                                                    ((approx_sma - 4) * (approx_sma - 2) * max_sma) 

                                                    / ((max_sma - 4) * (max_sma - 2) * approx_sma)

                                                 )

                    step_size = group['lr'] * var_rectification * math.sqrt(bias_correction2) / bias_correction1



                    p.data.addcdiv_(-step_size, exp_avg, denom)

                else:

                    p.data.add_(-group['lr'] * exp_avg / bias_correction1)



        return loss
base_model = models.resnet34



learn = unet_learner(data, base_model, model_dir='/kaggle/working', opt_func=RAdam, metrics=[dice])

learn.path = Path('/kaggle/working')
lr_find(learn)



learn.recorder.plot()
learn.fit_one_cycle(9, 1e-3)
# learn.unfreeze()



# lr_find(learn)



# learn.recorder.plot()
# learn.fit_one_cycle(8, slice(4e-6/20, 4e-6))
learn.export()
# load_learner('/kaggle/working')