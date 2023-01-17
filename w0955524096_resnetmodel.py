""" Show all the data path here """
""" you need to define your datapath """
import os
for dirname, _, filenames in os.walk('/kaggle/input/tester'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from PIL import Image

import os
import random
from datetime import datetime

import torch
from torch.utils.data import Dataset, DataLoader,SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import torchvision.models as models
from torchvision import transforms
from torchsummary import summary
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device
LABEL_DIR = '/kaggle/input/tester/label2.csv'

""" image dataset path you should define it """
DATA_DIR = '/kaggle/input/labeler'
IMAGE_DIR = os.path.join(DATA_DIR, 'DaanForestPark')

EPOCHES = 50
FILTER_NUMS = 8
FILTER_NUMS2 = 16
CHANNEL_NUMS = 3
KERNEL_SIZE = 13
STRIDE = 1
BATCH_SIZE = 32
NUM_WORKERS = 4
LR = 1e-2
# ToPILImage() -> Resize() -> ToTensor()
transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
#         transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])        
        ])
class MyDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        _images, _labels = [], []
        # total amount of dataset 
        _number = 0
        # Reading the categorical file
        label_df = pd.read_csv(label_dir)
        
        # Iterate all files including .jpg inages  
        for subdir, dirs, files in tqdm(os.walk(image_dir)):
            for filename in files:
                corr_label = label_df[label_df['dirpath']==subdir[len(DATA_DIR)+1:]]['label'].values
                if corr_label.size!= 0 and filename.endswith(('jpg')):
                    _images.append(subdir + os.sep + filename)
                    _labels.append(corr_label)
                    _number+=1
        
        # Randomly arrange data pairs
        mapIndexPosition = list(zip(_images, _labels))
        random.shuffle(mapIndexPosition)
        _images, _labels = zip(*mapIndexPosition)

        self._image = iter(_images)
        self._labels = iter(_labels)
        self._number = _number
        self._category = label_df['label'].nunique()
        self.transform = transform
        
    def __len__(self):
        return self._number

    def __getitem__(self, index):    
        img = next(self._image)
        lab = next(self._labels)
        
        img = self._loadimage(img)
        if self.transform:
            img = self.transform(img)        
        return img, lab
     
    def _categorical(self, label):
        return np.arange(self._category) == label[:,None]
    
    def _loadimage(self, file):
        return Image.open(file).convert('RGB')
    
    def get_categorical_nums(self):
        return self._category
    
train_dataset = MyDataset(IMAGE_DIR, LABEL_DIR, transform=transform)

valid_size = .2
num_train = len(train_dataset)
indices = list(range(num_train))
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, sampler=train_sampler, drop_last=True)
valid_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, sampler=valid_sampler, drop_last=True)
# model = SimpleCNN(train_dataset.get_categorical_nums()).to(device)
model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, train_dataset.get_categorical_nums())
# channels, H, W
model = model_ft.to(device=device)
summary(model, input_size=(CHANNEL_NUMS, 340, 192))
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
import shutil 
""" file path you should define it """
def save_checkpoint(model, state, filename='runs/ckpt/model.ckpt'):
    torch.save(state, filename)
    torch.save(model, 'model_best.ckpt')
#     shutil.copyfile(filename, 'model_best.ckpt')
optimizer = torch.optim.Adam(model.parameters(), lr=LR)   # optimize all cnn parameters
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=0, verbose=True)
criterion = nn.CrossEntropyLoss().to(device=device)

# early stopping
min_val_loss = np.Inf
patience = 10
global_step = 1

""" file path you should define it """
write_file = 'runs/experiment_{}'.format(datetime.now().strftime('%f'))
#writer = SummaryWriter(write_file)

""" file path you should define it """
os.mkdir(write_file + '/ckpt')

model_ft.train()

for epoch in range(EPOCHES):
    for i, (img_batch, label_batch) in tqdm(enumerate(train_loader)):    
        optimizer.zero_grad()      
        img_batch = img_batch.to(device=device)
        label_batch = label_batch.to(device=device)  
        output = model_ft(img_batch)
        loss = criterion(output, label_batch.squeeze())
        
        # l2 Regularization loss
        l2_regularization = 0
        l1_regularization = 0
        for p in model.parameters():
            l1_regularization += torch.norm(p, 1)
            l2_regularization += torch.norm(p, 2)
        loss = loss + 1e-3 * l2_regularization + 1e-4 * l1_regularization

        loss.backward()
        # clip the grandient value for avoiding explosion
        nn.utils.clip_grad_norm_(model.parameters(), 0.9) 
        optimizer.step()

        # Compute accuracy
        _, predicted = torch.max(output.cpu().data, 1)
        accuracy = torch.sum(predicted == label_batch.cpu().data.view(-1), dtype=torch.float32) / BATCH_SIZE
        
        # Write tensorboard
        #writer.add_scalar('train/Accuracy', accuracy.item(), global_step)
        #writer.add_scalar('train/Loss', loss.item(), global_step)
        #writer.add_scalar('train/L1RegLoss', l1_regularization.item(), global_step)
        #writer.add_scalar('train/L2RegLoss', l2_regularization.item(), global_step)
        #writer.add_scalar('train/LR', get_lr(optimizer), global_step)
                
        global_step += 1
        
        if i % 50== 0:
            print('epoch {}, step {}, \
            total_loss={:.3f}, \
            accuracy={:.3f}'.format(epoch+1, i, loss.item(), accuracy.item()))
    
    
    print('--- Validation phase ---')
    eval_loss = 0
    with torch.no_grad():
        for i, (img_batch, label_batch) in enumerate(valid_loader):
            output = model(img_batch.to(device))
            _, predicted = torch.max(output.cpu().data, 1)
            loss = criterion(output, label_batch.to(device).squeeze())
            accuracy = torch.sum(predicted == label_batch.data.view(-1), dtype=torch.float32) / BATCH_SIZE
            eval_loss += loss.item()
            
            # Write tensorboard
            #writer.add_pr_curve('valid/pr_curve', label_batch.squeeze(), predicted.squeeze(), epoch*len(valid_loader)+i)
            #writer.add_images('valid/image_batch', img_batch, epoch*len(valid_loader)+i)
            #writer.add_scalar('valid/Accuracy', accuracy.item(), epoch*len(valid_loader)+i)
            #writer.add_scalar('valid/Loss', loss.item(), epoch*len(valid_loader)+i)
    
    eval_loss = eval_loss / len(valid_loader)
    
    scheduler.step(eval_loss)
    
    print('epoch {}, val_loss={:.3f}'.format(epoch+1, eval_loss))

    ## Early Stopping
    if eval_loss < min_val_loss:
        
        """ file path you should define it => 'ckpt/resNet_{}.ckpt' """
        save_checkpoint(model, 
            {
            'epoch': epoch+1,
            'state_dict': model.state_dict(),
            'best_loss': eval_loss,
            'optimizer' :optimizer.state_dict(),
            }, os.path.join(write_file, 'ckpt/resNet_{}.ckpt'.format(epoch+1)))
        min_val_loss = eval_loss
    else:
        patience-=1
    if patience == 0:
        print('Early stopping')
        break

#writer.close()
print('Finish all training !')
# CKPT_PATH = 'model_best.ckpt'
# model.load_state_dict(torch.load(CKPT_PATH)['state_dict'])
model.eval()
acc = 0
for i, (img_batch, label_batch) in enumerate(valid_loader):
    output = model(img_batch.to(device))
    _, predicted = torch.max(output.cpu().data, 1)
    accuracy = torch.sum(predicted == label_batch.data.view(-1), dtype=torch.float32) / BATCH_SIZE
    acc += accuracy
print('accuracy={}'.format(acc/len(valid_loader)))
# torch.save(model, 'whole_model.ckpt')
