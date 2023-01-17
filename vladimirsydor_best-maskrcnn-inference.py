import os

import numpy as np

import pandas as pd

import torch

import cv2

import collections

import albumentations as albu

import torch

import torchvision

import seaborn as sns



from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split

from glob import glob

from os import path

from PIL import Image

from tqdm import tqdm

from matplotlib import pyplot as plt



%matplotlib inline
ROOT_PATH_TRAIN = '/kaggle/input/imaterialist-fashion-2019-FGVC6/train'

DF_PATH_TRAIN = '/kaggle/input/imaterialist-fashion-2019-FGVC6/train.csv'

PATH_TO_MODEL_WEIGHTS = '/kaggle/input/best-maskrcnn/best_model.pth'



IMAGE_SIZE = (512, 512)

IAMGE_PREDICTION_SIZE = (256, 256)



DEVICE='cuda'
def rle_decode(mask_rle, shape):

    '''

    mask_rle: run-length as string formated: [start0] [length0] [start1] [length1]... in 1d array

    shape: (height,width) of array to return

    Returns numpy array according to the shape, 1 - mask, 0 - background

    '''

    shape = (shape[1], shape[0])

    s = mask_rle.split()

    # gets starts & lengths 1d arrays

    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]

    starts -= 1

    # gets ends 1d array

    ends = starts + lengths

    # creates blank mask image 1d array

    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)

    # sets mark pixles

    for lo, hi in zip(starts, ends):

        img[lo:hi] = 1

    # reshape as a 2d mask image

    return img.reshape(shape).T
def rle_to_string(runs):

    return ' '.join(str(x) for x in runs)



def rle_encode(mask):

    pixels = mask.T.flatten()

    # We need to allow for cases where there is a '1' at either end of the sequence.

    # We do this by padding with a zero at each end when needed.

    use_padding = False

    if pixels[0] or pixels[-1]:

        use_padding = True

        pixel_padded = np.zeros([len(pixels) + 2], dtype=pixels.dtype)

        pixel_padded[1:-1] = pixels

        pixels = pixel_padded

    rle = np.where(pixels[1:] != pixels[:-1])[0] + 2

    if use_padding:

        rle = rle - 1

    rle[1::2] = rle[1::2] - rle[:-1:2]

    return rle_to_string(rle)
def visualise_masks(image_name, real_df, predicted_df=None, r_p=ROOT_PATH_TRAIN, im_class=None):

    # get image

    img_path = os.path.join(r_p, image_name)

    img = cv2.imread(img_path)

    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, IMAGE_SIZE)

    

    # get original mask

    if im_class is not None:

        original_raws = real_df[(real_df['ImageId'] == image_name) & (real_df['ClassId'] == im_class)]

    else:

        original_raws = real_df[real_df['ImageId'] == image_name]

                

    o_h, o_w = int(original_raws['Height'].mean()), int(original_raws['Width'].mean())

    

    o_mask = np.zeros(IMAGE_SIZE)        

        

    for annotation in original_raws['EncodedPixels']:

        o_mask += cv2.resize(rle_decode(annotation, (o_h, o_w)), IMAGE_SIZE)

        

    o_mask = (o_mask > 0.5).astype(np.uint8)

    

    if predicted_df is not None:

        # get predicted mask

        if im_class is not None:

            predicted_raws = predicted_df[(predicted_df['ImageId'] == image_name) & (predicted_df['ClassId'] == im_class)]

        else:

            predicted_raws = predicted_df[predicted_df['ImageId'] == image_name]

                

        p_mask = np.zeros(IMAGE_SIZE)

        

        for annotation in predicted_raws['EncodedPixels']:

            p_mask += rle_decode(annotation, IMAGE_SIZE)

        

        p_mask = (p_mask > 0.5).astype(np.uint8)

    

    fig=plt.figure(figsize=(20, 20))

    fig.add_subplot(1, 3, 1)

    plt.title('image')

    plt.imshow(img)

    fig.add_subplot(1, 3, 2)

    plt.title('original_mask')

    plt.imshow(o_mask)

    

    if predicted_df is not None:

        fig.add_subplot(1, 3, 3)

        plt.title('predicted_mask')

        plt.imshow(p_mask)

        plt.show()
train_df = pd.read_csv(DF_PATH_TRAIN)

train_df.head()
train_df['CategoryId'] = train_df.ClassId.apply(lambda x: str(x).split("_")[0])
plt.figure(figsize=(20, 7))

sns.countplot(train_df['CategoryId']);
train_df[train_df['ImageId'] == '00000663ed1ff0c4e0132b9b9ac53f6e.jpg']
visualise_masks('00000663ed1ff0c4e0132b9b9ac53f6e.jpg', train_df, im_class='31')
def get_unique_class_id_df(inital_df):

    temp_df = inital_df.groupby(['ImageId','ClassId'])['EncodedPixels'].agg(lambda x: ' '.join(list(x))).reset_index()

    size_df = inital_df.groupby(['ImageId','ClassId'])['Height', 'Width'].mean().reset_index()

    temp_df = temp_df.merge(size_df, on=['ImageId','ClassId'], how='left')

    

    return temp_df
train_df = get_unique_class_id_df(train_df)
train_df[train_df['ImageId'] == '00000663ed1ff0c4e0132b9b9ac53f6e.jpg']
visualise_masks('00000663ed1ff0c4e0132b9b9ac53f6e.jpg', train_df, im_class='31')
def create_one_represent_class(df_param):

    v_c_df = df_param['CategoryId'].value_counts().reset_index()

    one_represent = v_c_df.loc[v_c_df['CategoryId'] == 1, 'index'].tolist()

    df_param.loc[df_param['CategoryId'].isin(one_represent), 'CategoryId'] = 'one_represent'

    return df_param



def custom_train_test_split(df_param):

    

    df_param['CategoryId'] = df_param.ClassId.apply(lambda x: str(x).split("_")[0])

    

    img_categ = train_df.groupby('ImageId')['CategoryId'].apply(list).reset_index()

    img_categ['CategoryId'] = img_categ['CategoryId'].apply(lambda x: ' '.join(sorted(x)))

    

    img_categ = create_one_represent_class(img_categ)

    

    img_train, img_val  = train_test_split(img_categ, test_size=0.2, random_state=42, stratify=img_categ['CategoryId'])

    

    df_param = df_param.drop(columns='CategoryId')

    

    df_train = df_param[df_param['ImageId'].isin(img_train['ImageId'])].reset_index(drop=True)

    df_val = df_param[df_param['ImageId'].isin(img_val['ImageId'])].reset_index(drop=True)

    

    return df_train, df_val
train_df = pd.read_csv(DF_PATH_TRAIN)

train_df, val_df = custom_train_test_split(train_df)



train_df = get_unique_class_id_df(train_df)

val_df = get_unique_class_id_df(val_df)
plt.figure(figsize=(20, 7))

plt.title('Train')

sns.countplot(train_df['ClassId'].apply(lambda x: str(x).split("_")[0]));
plt.figure(figsize=(20, 7))

plt.title('Validation')

sns.countplot(val_df['ClassId'].apply(lambda x: str(x).split("_")[0]));
class MaskRCNNDataset(torch.utils.data.Dataset):

    def __init__(self, image_dir, df, height, width, augmentation=None, preprocessing=None):

        

        self.preprocessing = preprocessing

        self.augmentation = augmentation

        

        self.image_dir = image_dir

        self.df = df

        

        self.height = height

        self.width = width

        

        self.image_info = []

        

        self.df['CategoryId'] = self.df.ClassId.apply(lambda x: str(x).split("_")[0])

        self.num_classes = self.df['CategoryId'].nunique()

        

        temp_df = self.df.groupby('ImageId')['EncodedPixels', 'CategoryId'].agg(lambda x: list(x)).reset_index()

        size_df = self.df.groupby('ImageId')['Height', 'Width'].mean().reset_index()

        temp_df = temp_df.merge(size_df, on='ImageId', how='left')

        

        for index, row in tqdm(temp_df.iterrows(), total=len(temp_df)):

            image_id = row['ImageId']

            image_path = os.path.join(self.image_dir, image_id)

            cur_dict = {}

            cur_dict["image_id"] = image_id

            cur_dict["image_path"] = image_path

            cur_dict["width"] = self.width

            cur_dict["height"] = self.height

            cur_dict["labels"] = row["CategoryId"]

            cur_dict["orig_height"] = row["Height"]

            cur_dict["orig_width"] = row["Width"]

            cur_dict["annotations"] = row["EncodedPixels"]

            

            self.image_info.append(cur_dict)

            

        self.img2tensor = torchvision.transforms.ToTensor()



    def __getitem__(self, idx):

        

        img_path = self.image_info[idx]["image_path"]

        

        img = Image.open(img_path).convert("RGB")

        img = img.resize((self.width, self.height), resample=Image.BILINEAR)

        

        img = self.img2tensor(img)

            

        return img, os.path.basename(img_path)



    def __len__(self):

        return len(self.image_info)
num_classes = 46 + 1

device = torch.device(DEVICE)



model_ft = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

in_features = model_ft.roi_heads.box_predictor.cls_score.in_features

model_ft.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

in_features_mask = model_ft.roi_heads.mask_predictor.conv5_mask.in_channels

hidden_layer = 256

model_ft.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
model_ft.load_state_dict(torch.load(PATH_TO_MODEL_WEIGHTS, 

                                 map_location=torch.device(DEVICE)))
model_ft = model_ft.to(DEVICE)

model_ft.eval();
test_dataset = MaskRCNNDataset(

    ROOT_PATH_TRAIN,

    val_df,

    IAMGE_PREDICTION_SIZE[0],

    IAMGE_PREDICTION_SIZE[0], 

)
def create_final_df(result_dict):

    final_df = {'ImageId':[], 'EncodedPixels':[], 'ClassId':[]}

    for im_name, im_dict in result_dict.items():

        final_df['ImageId'] += [im_name]*len(im_dict)

        for cls_id, enc_p in im_dict.items():

            final_df['EncodedPixels'].append(enc_p)

            final_df['ClassId'].append(cls_id)

            

    return pd.DataFrame(final_df)



class InfernceModel(object):

    def __init__(self, nn_model, device, output_size=(512,512), threshold=0.5, size_min_mask=250):

        self.nn_model = nn_model

        self.output_size = output_size

        self.threshold = threshold

        self.size_min_mask = size_min_mask

        self.device = device

        

    def __call__(self, inf_dataset):

        result = {}

        with torch.no_grad():

            for batch in tqdm(inf_dataset):

                name = batch[1]

                batch = self.nn_model([batch[0].to(self.device)])[0]

                result[name] = self.post_process(batch['masks'].cpu().numpy(), 

                                                 batch['labels'].cpu().numpy(), 

                                                 batch['scores'].cpu().numpy())

                        

        final_df = create_final_df(result) 

        final_df['Height'] = self.output_size[0]

        final_df['Width'] = self.output_size[1]

        final_df['ClassId'] = final_df['ClassId'].astype(str)

                    

        return final_df

        

    def post_process(self, masks, labels, scores):

        labels -= 1

        

        rle_mask = {}

        if scores.max() < 0.5:

            cur_idx = scores.argmax()

            

            item_mask = (masks[cur_idx][0] > self.threshold).astype(np.uint8)

            item_mask = cv2.resize(item_mask, self.output_size)

            

            if item_mask.sum() >= self.size_min_mask:

                rle_mask[str(labels[cur_idx])] = rle_encode(item_mask)

            

        else:

            

            good_masks = masks[scores > 0.5]

            good_labels = labels[scores > 0.5]

        

            for m,l in zip(masks, labels):



                item_mask = (m[0] > self.threshold).astype(np.uint8)

                item_mask = cv2.resize(item_mask, self.output_size)



                if item_mask.sum() < self.size_min_mask:

                    continue

                else:

                    rle_mask[str(l)] = rle_encode(item_mask)

            

        return rle_mask
inf_model = InfernceModel(nn_model=model_ft, device=DEVICE)
predicted_df = inf_model(test_dataset)
predicted_df.head()
plt.figure(figsize=(20, 7))

plt.title('Prediction')

sns.countplot(predicted_df['ClassId']);
visualise_masks('000b3a87508b0fa185fbd53ecbe2e4c6.jpg', val_df, predicted_df)
visualise_masks('fffc631acce2e28e1628de685d40c980.jpg', val_df, predicted_df)
visualise_masks('ffec8295f37df6ea12eecbb60d2c23d4.jpg', val_df, predicted_df)
visualise_masks('001039acb67251508b1b32fd37a49f43.jpg', val_df, predicted_df, im_class='31')
visualise_masks('005ccdb239e2d6cfe62506dd6eb5693e.jpg', val_df, predicted_df, im_class='10')
visualise_masks('000cd2e13d1bdd28f480304d7bb9e1ca.jpg', val_df, predicted_df, im_class='23')
visualise_masks('05e6ef19957d43524d972de6d2f41b57.jpg', val_df, predicted_df, im_class='45')
# mainly from here https://www.kaggle.com/kyazuki/calculate-evaluation-score



def IoU(A,B):

    AorB = np.logical_or(A,B).astype('int')

    AandB = np.logical_and(A,B).astype('int')

    IoU = AandB.sum() / AorB.sum()

    return IoU



def IoU_threshold(data):

    # Note: This rle_to_mask should be called before loop below for speed-up! We currently implement here to reduse memory usage.

    mask_gt = rle_decode(data['EncodedPixels_true'], (int(data['Height_true']), int(data['Width_true'])))

    mask_pred = rle_decode(data['EncodedPixels_pred'], (int(data['Height_pred']), int(data['Width_pred'])))\

    

    if (int(data['Height_true']), int(data['Width_true'])) != IMAGE_SIZE:

        mask_gt = cv2.resize(mask_gt, IMAGE_SIZE)

    if (int(data['Height_pred']), int(data['Width_pred'])) != IMAGE_SIZE:

        mask_pred = cv2.resize(mask_pred, IMAGE_SIZE)

    

    mask_gt = mask_gt > 0.5

    mask_pred = mask_pred > 0.5

    

    return IoU(mask_gt, mask_pred)



def best_metric(true_df, pred_df):

    eval_df = pd.merge(true_df, pred_df, how='outer', on=['ImageId', 'ClassId'], suffixes=['_true', '_pred'])



    # IoU for True Positive

    idx_ = eval_df['EncodedPixels_true'].notnull() & eval_df['EncodedPixels_pred'].notnull()

    IoU = eval_df[idx_].apply(IoU_threshold, axis=1)



    # False Positive

    fp = (eval_df['EncodedPixels_true'].isnull() & eval_df['EncodedPixels_pred'].notnull()).sum()



    # False Negative

    fn = (eval_df['EncodedPixels_true'].notnull() & eval_df['EncodedPixels_pred'].isnull()).sum()



    threshold_IoU = np.arange(0.5, 1.0, 0.05)

    scores = []

    for th in threshold_IoU:

        # True Positive

        tp = (IoU > th).sum()

        iou_fp = (IoU <= th).sum()



        # False Positive (not Ground Truth) + False Positive (under IoU threshold)

        fp = fp + iou_fp



        # Calculate evaluation score

        score = tp / (tp + fp + fn)

        scores.append(score)



    mean_score = sum(scores) / len(threshold_IoU)

    return mean_score
images_to_count_metric = val_df['ImageId'].unique()[:100]



best_metric(val_df[val_df['ImageId'].isin(images_to_count_metric)], val_df[val_df['ImageId'].isin(images_to_count_metric)])
best_metric(val_df, predicted_df)