import random
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import torchvision
from matplotlib import pyplot as plt
from torch.utils.data import Dataset,DataLoader
import cv2
import functools
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from sklearn.model_selection import KFold

import vtk
from vtk.util import numpy_support
from tqdm.auto import tqdm

from torchvision import models

folder_path = "../input/rsna-str-pulmonary-embolism-detection"
train_path = folder_path + "/train/"
test_path = folder_path + "/test/"

test_data  = pd.read_csv(folder_path + "/test.csv")
submission = pd.read_csv(folder_path + "/sample_submission.csv")

cols_ID = ["StudyInstanceUID","SeriesInstanceUID","SOPInstanceUID"]
test_data["ImagePath"] = test_path+ test_data[cols_ID[0]]+"/"+test_data[cols_ID[1]]+"/"+test_data[cols_ID[2]]+".dcm"
SEED  = 42

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASSEED']  = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(SEED)

config = {
    "model_name": "resnet18",
    "learning_rate":  0.001,
    "train_batch_size": 32,
    "valid_batch_size": 32,
    "test_batch_size": 64,
    "epochs": 10,
    "nfolds": 1,
    "number_of_samples": 1790594,
    "weight": '../input/rsna-train-image-data-use-vtk/',
}

formatted_settings = {
    'input_size': [3, 384, 384],
    'input_range': [0, 1],
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225],
}

target_columns = ['pe_present_on_image', 'negative_exam_for_pe', 'rv_lv_ratio_gte_1', 
                  'rv_lv_ratio_lt_1','leftsided_pe', 'chronic_pe','rightsided_pe', 
                  'acute_and_chronic_pe', 'central_pe', 'indeterminate']

study_level_columns = [ 'negative_exam_for_pe', 'rv_lv_ratio_gte_1', 
                  'rv_lv_ratio_lt_1','leftsided_pe', 'chronic_pe','rightsided_pe', 
                  'acute_and_chronic_pe', 'central_pe', 'indeterminate']

classes = len(target_columns)
def get_test_transforms():
    return A.Compose([
         A.Resize(formatted_settings['input_size'][2], formatted_settings['input_size'][1], p=1.0),
    ])

def preprocess_input(x, mean=None, std=None, input_range=None, **kwargs):
    # BGR => RGB
    if x.shape[0] == 3:
        x = x[..., ::-1].copy()

    if input_range is not None:
        if x.max() > 1 and input_range[1] == 1:
            x = x / 255.0

    if mean is not None:
        mean = np.array(mean)
        x = x - mean

    if std is not None:
        std = np.array(std)
        x = x / std
    return x

def to_tensor(x, **kwargs):
    """
    RGB => BGR
    """
    if x.shape[2] == 3:
        x = x.transpose(2, 0, 1).astype('float32')
    return x

def get_preprocessing(preprocessing_fn):
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)
reader = vtk.vtkDICOMImageReader()

def get_img(path):
    reader.SetFileName(path)
    reader.Update()
    _extent = reader.GetDataExtent()
    ConstPixelDims = [_extent[1]-_extent[0]+1, _extent[3]-_extent[2]+1, _extent[5]-_extent[4]+1]

    ConstPixelSpacing = reader.GetPixelSpacing()
    imageData = reader.GetOutput()
    pointData = imageData.GetPointData()
    arrayData = pointData.GetArray(0)
    ArrayDicom = numpy_support.vtk_to_numpy(arrayData)
    ArrayDicom = ArrayDicom.reshape(ConstPixelDims, order='F')
    return ArrayDicom

def window(img, WL=50, WW=350):
    upper, lower = WL+WW//2, WL-WW//2
    X = np.clip(img.copy(), lower, upper)
    return X.astype('uint8')

def convert_to_color(array):
    image_lung = window(array, WL=-600, WW=1500)
    image_mediastinal = window(array, WL=40, WW=400)
    image_pe_specific = window(array, WL=100, WW=700)
    image = np.concatenate([image_mediastinal, image_pe_specific, image_lung], axis=2)
    return image
img = get_img(test_data["ImagePath"][0])
img = convert_to_color(img)

plt.figure(figsize=[12,6])
plt.subplot(131)
plt.imshow(img[:,:,0],cmap='gray')
plt.subplot(132)
plt.imshow(img[:,:,1],cmap='gray')
plt.subplot(133)
plt.imshow(img[:,:,2],cmap='gray')
class RsnaDataset(Dataset):
    def __init__(self,df,transforms=None, preprocessing=None):
        super().__init__()
        self.image_paths = df['ImagePath'].unique()
        self.df = df
        self.study_ids= df[cols_ID[1]].values
        self.sop_ids = df[cols_ID[2]].values
        self.transforms = transforms
        self.preprocessing = preprocessing
    
    def __getitem__(self,index):
        image_path = self.image_paths[index]
        image = get_img(image_path)
        image = convert_to_color(image)
        
        study_id = self.study_ids[index]
        sop_id = self.sop_ids[index]
        
        if self.transforms:
            image = self.transforms(image=image)['image']
        
        if self.preprocessing:
            image = self.preprocessing(image=image)['image']
        

        image = torch.tensor(image,dtype=torch.float)        
        
        return {"image":image,
                "study_id":study_id,
                "sop_id":sop_id}   
    
    def __len__(self):
        return self.image_paths.shape[0]
def get_model(config):
    if config["model_name"] == 'resnext50_32x4d':
        model = models.resnext50_32x4d(pretrained=False)
        model.fc = torch.nn.Linear(in_features=2048, out_features=classes, bias=True)
    else:
        model = models.resnet18(pretrained=False)
        model.fc = torch.nn.Linear(in_features=model.fc.in_features, out_features=classes, bias=True)

    model = model.cuda()
    return model

model = get_model(config)
def inference():
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    all_prediction = np.zeros((test_data.shape[0],len(target_columns)))
    study_ids = list()
    sop_ids = list()
    for i in range(config["nfolds"]):
        model.load_state_dict(torch.load(f"{config['weight']}/model{i}.bin"))
        predictions = list()
        model.to(device)
        test_ds = RsnaDataset(test_data, get_test_transforms(), get_preprocessing(functools.partial(preprocess_input, **formatted_settings)))
        test_dl = DataLoader(test_ds,
                        batch_size=config['test_batch_size'],
                        shuffle=False)
        
        tqdm_loader = tqdm(test_dl)
        
        with torch.no_grad():
            for inputs in tqdm_loader:
                images = inputs["image"].to(device, dtype=torch.float)
                outputs = model(images) 
                predictions.extend(torch.sigmoid(outputs).cpu().detach().numpy())
                if i == 0:
                    study_ids.extend(inputs["study_id"])
                    sop_ids.extend(inputs["sop_id"])

        all_prediction += np.array(predictions)/config['nfolds']
        
    return all_prediction, study_ids, sop_ids
if len(os.listdir("../input/rsna-str-pulmonary-embolism-detection/test/")) > 700:
    predictions,study_ids,sop_ids = inference()

    df = pd.DataFrame(predictions)
    df.columns = target_columns
    df["StudyInstanceUID"] = study_ids
    df["SOPInstanceUID"] = sop_ids

    temp1 = df.groupby("StudyInstanceUID")[target_columns].mean().reset_index()
    temp1.drop("pe_present_on_image",inplace=True,axis=1)
    temp1 = pd.melt(temp1, id_vars=["StudyInstanceUID"], value_vars=study_level_columns)
    temp1["StudyInstanceUID"] = temp1["StudyInstanceUID"].astype(str) + "_" +temp1["variable"].astype(str)
    temp1.drop(["variable"],axis=1,inplace=True)
    temp1.columns = ["id", "label"]

    temp2 = df.drop(study_level_columns +["StudyInstanceUID"],axis=1)
    temp2 = pd.melt(temp2, id_vars=["SOPInstanceUID"], value_vars=['pe_present_on_image'])
    temp2.drop(["variable"],axis=1,inplace=True)
    temp2.columns = ["id", "label"]

    temp_sub = temp2.append(temp1)
    submission = pd.merge(submission[['id']], temp_sub, on='id', how='left')
    print(len(submission[submission['label'].isna()]))
    submission = submission.fillna(0.5)
    submission.to_csv("submission.csv",index=False)
else:
    submission.id = submission.id.astype(str)
    submission.to_csv("submission.csv",index=False)
print(submission.shape)
submission.head()