!pip install timm
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# general imports
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
%matplotlib inline
import seaborn as sns
sns.set()

import cv2
import os  
from PIL import Image 
from pprint import pprint

# torch and torchvision
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms

# load pretrained models 
import timm

# catalyst for training and metrics
from catalyst.utils import split_dataframe_train_test
from catalyst.dl.callbacks import AccuracyCallback
from catalyst.dl import SupervisedRunner
def config():
    cfg = {
        # raw csv data
        'train_csv_path': '/kaggle/input/aptos2019-blindness-detection/train.csv',
        'test_csv_path': '/kaggle/input/aptos2019-blindness-detection/test.csv',
        # images path
        'img_root': '/kaggle/input/aptos2019-blindness-detection/train_images/',
        'test_img_root': '/kaggle/input/aptos2019-blindness-detection/test_images/',
        # backend architecture, features are extracted from this
        'arch': 'resnext50_32x4d',
        # training parameters 
        'random_state': 1,
        'num_classes': 5,
        'test_size': 0.2,
        'input_size': 512,
        'freeze': True,
        'lr': 3e-4,
        'logdir': '/kaggle/working/logs/',
        'device': None,
        'batch_size': 8,
        'test_batch_size': 2,
        'num_epochs': 7,
        # logging 
        'verbose': True,
        'check': False,  # set this true to run for 3 epochs only
        # data labels
        'class_names': ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']

    }
    return cfg

cfg = config()
cfg['device'] = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
print("Parameters for Training:")
pprint(cfg)
train_df = pd.read_csv(cfg['train_csv_path'])
train_df.sample(5)
fig,ax = plt.subplots(figsize=(12,10))
sns.distplot(train_df['diagnosis'], bins=5, kde=False)
def balance_data(csv_path: str, test_size: float = 0.2, random_state: int = 123):
    df = pd.read_csv(csv_path)
    # first class has large number of samples as compares to others
    # one way to balance is by sampling smaller amount of data
    class_0 = df[df['diagnosis'] == 0]
    class_0 = class_0.sample(400)
    class_0_train, class_0_test = split_dataframe_train_test(
        class_0, test_size=test_size, random_state=random_state)
    df_train = class_0_train
    df_test = class_0_test

    class_1 = df[df['diagnosis'] == 1]
    class_1_train, class_1_test = split_dataframe_train_test(
        class_1, test_size=test_size, random_state=random_state)
    df_train = df_train.append(class_1_train)
    df_test = df_test.append(class_1_test)

    # sub sampling data for Moderate category
    class_2 = df[df['diagnosis'] == 2]
    class_2 = class_2.sample(400)
    class_2_train, class_2_test = split_dataframe_train_test(
        class_2, test_size=test_size, random_state=random_state)
    df_train = df_train.append(class_2_train)
    df_test = df_test.append(class_2_test)

    class_3 = df[df['diagnosis'] == 3]
    class_3_train, class_3_test = split_dataframe_train_test(
        class_3, test_size=test_size, random_state=random_state)
    df_train = df_train.append(class_3_train)
    df_test = df_test.append(class_3_test)

    class_4 = df[df['diagnosis'] == 4]
    class_4_train, class_4_test = split_dataframe_train_test(
        class_4, test_size=test_size, random_state=random_state)
    df_train = df_train.append(class_4_train)
    df_test = df_test.append(class_4_test)

    return df_train, df_test
train_df, test_df = balance_data(cfg['train_csv_path'])
print("Training Samples:")
print("No DR:", len(train_df[train_df['diagnosis']==0]))
print("Mild:", len(train_df[train_df['diagnosis']==1]))
print("Moderate:", len(train_df[train_df['diagnosis']==2]))
print("Severe:", len(train_df[train_df['diagnosis']==3]))
print("Proliferative DR:", len(train_df[train_df['diagnosis']==4]))
print("\nTest Samples:")
print("No DR:", len(test_df[test_df['diagnosis']==0]))
print("Mild:", len(test_df[test_df['diagnosis']==1]))
print("Moderate:", len(test_df[test_df['diagnosis']==2]))
print("Severe:", len(test_df[test_df['diagnosis']==3]))
print("Proliferative DR:", len(test_df[test_df['diagnosis']==4]))
fig,ax = plt.subplots(figsize=(12,10))
sns.distplot(train_df['diagnosis'], bins=5, kde=False);
fig,ax = plt.subplots(figsize=(12,10))
sns.distplot(test_df['diagnosis'], bins=5, kde=False)
def read_sample(root:str,filename:str):
    img = cv2.imread(os.path.join(root, filename+'.png'))
    return img
def plot_samples(df:pd.DataFrame, idx:int=0):
    filename = df.iloc[idx]['id_code']
    label = df.iloc[idx]['diagnosis']
    img = read_sample(cfg['img_root'],filename)
    print(f"Image:{img.shape}")
    fig = plt.subplots(nrows=1, ncols=1, figsize=(10,10))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"Diagnosis:{label}")
    plt.axis('off')
plot_samples(train_df, 789); plot_samples(train_df, 432)
class AptosDataset(Dataset):
    """Retrieves each data item for use with dataloaders"""
    def __init__(self,
                 img_root: str,
                 df: pd.DataFrame,
                 img_transforms: transforms = None,
                 is_train: bool = True
                 ):

        self.df = df
        self.img_root = img_root
        self.img_transforms = img_transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        filename = row['id_code']
        target = int(row['diagnosis'])
        img = Image.open(os.path.join(
            self.img_root, filename+'.png')).convert('RGB')
        img = np.asarray(img)
        if self.img_transforms is not None:
            augmented = self.img_transforms(image=img)
            img = augmented['image']
        return img, np.asarray(target)
import albumentations as albu
from albumentations.pytorch import ToTensor



def pre_transforms(image_size=512):
    # Convert the image to a square of size image_size x image_size
    # (keeping aspect ratio)
    result = [
        albu.LongestMaxSize(max_size=image_size),
        albu.PadIfNeeded(image_size, image_size, border_mode=2)
    ]
    
    return result

def hard_transforms():
    result = [
        # Random shifts, stretches and turns with a 50% probability
        albu.ShiftScaleRotate( 
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=15,
            border_mode=0,
            p=0.5
        ),
        # add random brightness and contrast, 30% prob
        albu.RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2, p=0.3
        ),
        # Random gamma changes with a 30% probability
        albu.RandomGamma(gamma_limit=(85, 115), p=0.3),
        # Randomly changes the hue, saturation, and color value of the input image 
        albu.HueSaturationValue(p=0.3),
        albu.JpegCompression(quality_lower=80),
    ]
    
    return result

def post_transforms():
    # we use ImageNet image normalization
    # and convert it to torch.Tensor
    return [albu.Normalize(), ToTensor()]

def compose(transforms_to_compose):
    # combine all augmentations into one single pipeline
    result = albu.Compose([
      item for sublist in transforms_to_compose for item in sublist
    ])
    return result

def get_transforms():
    
    train_transforms = compose([
                        pre_transforms(), 
                        hard_transforms(), 
                        post_transforms()
    ])
    
    val_transforms = compose([pre_transforms(), post_transforms()])
    
    return train_transforms, val_transforms
train_transforms, test_transforms = get_transforms()
train_dataset = AptosDataset(
        img_root=cfg['img_root'],
        df=train_df,
        img_transforms=train_transforms,
        is_train=True,
    )

test_dataset = AptosDataset(
        img_root=cfg['img_root'],
        df=test_df,
        img_transforms=test_transforms,
        is_train=False,
    )
print(f"Training set size:{len(train_dataset)}, Test set size:{len(test_dataset)}")
train_loader = DataLoader(train_dataset, cfg['batch_size'], shuffle=True, num_workers=1)
test_loader = DataLoader(test_dataset, cfg['test_batch_size'], shuffle=False, num_workers=1)

loaders = {
        'train': train_loader,
        'valid': test_loader
}
class AptosModelV2(nn.Module):
    def __init__(self, 
                 arch:str='resnet101', 
                 z_dims:int=2048, 
                 nb_classes:int=5,
                 drop:float=0.5):
        super(AptosModelV2, self).__init__()
        self.model = timm.create_model(arch, pretrained=True,drop_rate=drop)
        self.model.reset_classifier(num_classes = nb_classes)
        
    def forward(self, x):
        return self.model(x)
    
model = AptosModelV2(arch=cfg['arch'])
model.train();
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
# tried launching tensorboard but doesn't work in browser
# you can launch on local machine or colab
%load_ext tensorboard
%tensorboard --logdir cfg['logdir']
runner = SupervisedRunner(device=cfg['device'])
runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    loaders=loaders,
    
    callbacks=[
        AccuracyCallback(
            num_classes=cfg['num_classes'],
            threshold=0.5,
            activation="Sigmoid"
        ),
    ],
    logdir=cfg['logdir'],
    num_epochs=cfg['num_epochs'],
    verbose=cfg['verbose'],
    # set this true to run for 3 epochs only
    check=cfg['check'],
)
from catalyst.dl import utils

utils.plot_metrics(
    logdir=cfg['logdir'], 
    metrics=["loss", "accuracy01"])
def run_evaluation():
    # given model and valid dataset 
    # iterate over dataset and compute prediction
    y_true = []
    y_pred = []
    test_size = len(test_dataset)
    model.eval()
    for i in range(test_size):
        img_tensor = test_dataset[i][0].unsqueeze(0,)
        with torch.no_grad():
            pred = torch.sigmoid(model(img_tensor.to(cfg['device']))).squeeze().cpu()
            _,output = torch.topk(pred,1)
            output = output.numpy()[0]
        label = test_dataset[i][1].item()
        y_true.append(label)
        y_pred.append(output)
    
    return y_true, y_pred
test_true, test_pred = run_evaluation()
from sklearn.metrics import classification_report

print(classification_report(test_true, test_pred, target_names=cfg['class_names']))
def run_on_held_out(csv_path, img_root, img_transforms):
    # given model and valid dataset 
    # iterate over dataset and compute prediction
    
    df = pd.read_csv(csv_path)
    test_size = len(df)
    print(f"Size: {test_size}")
    y_pred = {}
    model.eval()
    for idx,row in df.iterrows():
        filename = row['id_code']
        # load and transform input imate
        img = Image.open(os.path.join(
            img_root, filename+'.png')).convert('RGB')
        img = np.asarray(img)
        augmented = img_transforms(image=img)
        img_tensor = augmented['image']
        img_tensor = img_tensor.unsqueeze(0,)
        
        # run prediction
        with torch.no_grad():
            pred = torch.sigmoid(model(img_tensor.to(cfg['device']))).squeeze().cpu()
            _,output = torch.topk(pred,1)
            output = output.numpy()[0]
        # store results
        y_pred[filename] = output
    
    return y_pred
submission_dict = run_on_held_out(cfg['test_csv_path'], cfg['test_img_root'], test_transforms)
submission_df = pd.DataFrame.from_dict(submission_dict, orient='index', columns=['diagnosis'])
submission_df.index.name = 'id_code'
submission_df.to_csv('submission.csv')
