!pip install natsort
import os
import natsort
import torch
from tqdm import tqdm
import numpy as np 
import glob
from PIL import Image
import albumentations
import torch.nn.functional as F
import torchvision
import torch.nn as nn
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import metrics
from torchvision import datasets, models , transforms
import matplotlib.pyplot as plt
%matplotlib inline
import cv2
import helper
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
Data_dir_images = "../input/CVC-ClinicDB/Original"
Data_dir_masks = "../input/CVC-ClinicDB/Ground_Truth"

#this size coz U-net paper has similar input size
image_size = (572,572)

epochs = 20

# check if CUDA is available
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
    device = "cpu"
else:
    print('CUDA is available!  Training on GPU ...')
    device ="cuda"

print(device)

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
batch_size = 2

# class_names = ["polyp"]

images_list = glob.glob(f"{Data_dir_images}/*")
images_list = natsort.natsorted(images_list, reverse = False)
print(len(images_list))
masks_list = glob.glob(f"{Data_dir_masks}/*")
masks_list = natsort.natsorted(masks_list, reverse = False)
print(len(masks_list))

split=0.1
total_size = len(images_list)
valid_size = int(split * total_size)
test_size = int(split * total_size)

train_img, valid_img , train_masks, valid_masks = model_selection.train_test_split(images_list, masks_list, test_size=valid_size, random_state = 42)

train_img, test_img , train_masks, test_masks = model_selection.train_test_split(train_img, train_masks, test_size=test_size, random_state = 42)
#traindataset 

class PolypDataset:
  def __init__(self, images, masks, img_transforms, mask_transforms):
    self.images = images
    self.masks = masks
    self.img_transforms = img_transforms
    self.mask_transforms = mask_transforms

  def __len__(self):
    return len(self.images)

  def __getitem__(self, item):
    image = cv2.imread(self.images[item])#we use cv2 because PIL didnt load the image and we convert later to PIL image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image) #Converting CV2 to PIL
    masks = cv2.imread(self.masks[item])#we use cv2 because PIL didnt load the image and we convert later to PIL image
    masks = cv2.cvtColor(masks, cv2.COLOR_BGR2RGB)
    masks = Image.fromarray(masks)#we use cv2 because PIL didnt load the image and we convert later to PIL image
    transforms_image = self.img_transforms(image)
    transforms_masks = self.mask_transforms(masks)

    return transforms_image , transforms_masks
#training data
train_transform_images = transforms.Compose([
                                      transforms.Resize(image_size),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean, std)])
train_transform_masks = transforms.Compose([
                                      transforms.Resize(image_size),
                                      transforms.ToTensor()])

train_dataset = PolypDataset(train_img, train_masks, train_transform_images, train_transform_masks)


train_dataloader = torch.utils.data.DataLoader(train_dataset,num_workers=4, batch_size=batch_size, shuffle=True)
#validation data
valid_transform_images = transforms.Compose([
                                      transforms.Resize(image_size),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean, std)])
valid_transform_masks = transforms.Compose([
                                      transforms.Resize(image_size),
                                      transforms.ToTensor()])

valid_dataset = PolypDataset(valid_img, valid_masks, valid_transform_images, valid_transform_masks)

valid_dataloader = torch.utils.data.DataLoader(valid_dataset,num_workers=4, batch_size=batch_size)
#testing data
test_transform_images = transforms.Compose([
                                      transforms.Resize(image_size),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean, std)])
test_transform_masks = transforms.Compose([
                                      transforms.Resize(image_size),
                                      transforms.ToTensor()])

test_dataset = PolypDataset(test_img, test_masks, test_transform_images, test_transform_masks)

test_dataloader = torch.utils.data.DataLoader(test_dataset,num_workers=4, batch_size=batch_size)

dataloaders = {
  'train': train_dataloader,
  'val': valid_dataloader
}
#visualising the images 
def imshow_images(dataloader, bs=2):
  fig = plt.figure(figsize=(24, 16))
  fig.tight_layout()
  images , masks = next(iter(dataloader))

  for num, (image, mask) in enumerate(zip(images[:bs], masks[:bs])):
      plt.subplot(4,6,num+1)
      plt.axis('off')
      image = image.cpu().numpy()
      out = np.transpose(image, (1,2,0))
      mean = np.array([0.485, 0.456, 0.406])
      std = np.array([0.229, 0.224, 0.225])
      inp = std * out + mean
      inp = np.clip(inp, 0, 1)
      plt.imshow(inp)


#visualising the images 
def imshow_masks(dataloader, bs=2):
  fig = plt.figure(figsize=(24, 16))
  fig.tight_layout()
  images , masks = next(iter(dataloader))

  for num, (image, mask) in enumerate(zip(images[:bs], masks[:bs])):
      plt.subplot(4,6,num+1)
      plt.axis('off')
      mask = mask.cpu().numpy()
      mask = np.transpose(mask, (1,2,0))
      plt.imshow(mask)

#show the images and masks
imshow_images(valid_dataloader)
imshow_masks(valid_dataloader)
'''
https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py

Some of the snippets are borrowed from this github repo
please check them out too ( they have explained in more depth)
'''


def db_conv(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True)
    )
    return conv


def down_sample(in_channels, out_channels):
    maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            db_conv(in_channels, out_channels)
        )
    return maxpool_conv

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels,):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels , out_channels, kernel_size=2, stride=2)
        self.conv = db_conv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(UNet, self).__init__()

        self.in_channels = in_channels
        self.classes = n_classes
        self.first_layer = db_conv(in_channels, 64)
        self.down_sample1 = down_sample(64, 128)
        self.down_sample2 = down_sample(128, 256)
        self.down_sample3 = down_sample(256, 512)
        self.down_sample4 = down_sample(512, 1024)

        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        
        self.output = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, image):
        #encoder
        x1 =self.first_layer(image)
        x2 = self.down_sample1(x1)
        x3 = self.down_sample2(x2)
        x4 = self.down_sample3(x3)
        x5 = self.down_sample4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x= self.output(x)
        return x

Unet = UNet(in_channels=3, n_classes=3)
from collections import defaultdict
import torch.nn.functional as F


def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()    

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    
    return loss.mean()

checkpoint_path = "checkpoint.pth"

def calc_loss(pred, target, metrics, bce_weight=0.5):
    # print(pred.shape,target.shape)
  
    bce = F.binary_cross_entropy_with_logits(pred, target)

    pred = torch.sigmoid(pred)
    dice = dice_loss(pred, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss

def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))

def train_model(model, optimizer, scheduler, num_epochs=25):
    best_loss = 1e10

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 30)

        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0

            for inputs, labels in tqdm(dataloaders[phase], total = len(dataloaders[phase])):
                inputs = inputs.to(device)
                labels = labels.to(device)
                

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = calc_loss(outputs, labels, metrics)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                epoch_samples += inputs.size(0)

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples

            if phase == 'train':
              scheduler.step()
              for param_group in optimizer.param_groups:
                  print("LR", param_group['lr'])

            # save the model weights
            if phase == 'val' and epoch_loss < best_loss:
                print(f"saving best model to {checkpoint_path}")
                best_loss = epoch_loss
                torch.save(model.state_dict(), checkpoint_path)

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(torch.load(checkpoint_path))
    return model
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import time


if train_on_gpu:
  Unet.cuda()


optimizer_ft = optim.Adam( Unet.parameters(), lr=1e-4)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=8, gamma=0.1)

model = train_model(Unet, optimizer_ft, exp_lr_scheduler, num_epochs=epochs)
def reverse_transform_mask(inp):
  inp = np.transpose(inp,(1, 2, 0))
  inp = np.clip(inp, 0, 1)
  inp = (inp * 255).astype(np.uint8)
  return inp

def reverse_transform_images(inp):
  inp =  np.transpose(inp,(1, 2, 0))
  mean = np.array([0.485, 0.456, 0.406])
  std = np.array([0.229, 0.224, 0.225])
  inp = std * inp + mean
  inp = np.clip(inp, 0, 1)

  return inp


model.eval()   # Set model to the evaluation mode
# Create a new simulation dataset for testing
# Get the first batch
images, masks = next(iter(test_dataloader))
images = images.to(device)
masks = masks.to(device)
# Predict
pred = model(images)

# The loss functions include the sigmoid function.
prediction = torch.sigmoid(pred)

prediction = pred.data.cpu().numpy()
fig = plt.figure(figsize=(24, 16))
fig.tight_layout()


for num, (img, mask, pred) in enumerate(zip(images[:batch_size], masks[:batch_size], prediction[:batch_size])):
      plt.subplot(4,6,num+1)
      plt.axis('off')
      img = img.cpu().numpy()
      mask = mask.cpu().numpy()
      image_original = reverse_transform_images(img)
      original_label = reverse_transform_mask(mask)
      predictions = reverse_transform_mask(pred)
    

      white_line = np.ones((572, 10, 3))

      all_images = [
          image_original, white_line,
          original_label, white_line,predictions]
      image = np.concatenate(all_images, axis=1)
      imgplot = plt.imshow(image)
