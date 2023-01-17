
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd# data processing, CSV file I/O (e.g. pd.read_csv)
import torch #version 0.3
from torchvision import transforms, models
from torch.autograd import Variable
from torch.utils import data
import torch.nn.functional as F
from torch import nn
import sklearn
import cv2
import imageio
import skimage

import glob
from tqdm import tqdm, tqdm_notebook
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.
# https://github.com/leigh-plt/cs231n_hw2018/blob/master/assignment2/pytorch_tutorial.ipynb
def save_checkpoint(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict(),
             'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)
    
def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)
def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)

class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            #先下采样
            ConvRelu(in_channels, middle_channels),
            #卷积转置是一种上采样
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)

class UNet11(nn.Module):
    def __init__(self, num_filters=32):
        """
        :param num_classes:
        :param num_filters:
        """
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

        # Convolutions are from VGG11
        self.encoder = models.vgg11().features
        
        # "relu" layer is taken from VGG probably for generality, but it's not clear 
        self.relu = self.encoder[1]
        
        self.conv1 = self.encoder[0]
        self.conv2 = self.encoder[3]
        self.conv3s = self.encoder[6]
        self.conv3 = self.encoder[8]
        self.conv4s = self.encoder[11]
        self.conv4 = self.encoder[13]
        self.conv5s = self.encoder[16]
        self.conv5 = self.encoder[18]
    
        self.center = DecoderBlock(num_filters * 8 * 2, num_filters * 8 * 2, num_filters * 8)
        self.dec5 = DecoderBlock(num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 8)
        self.dec4 = DecoderBlock(num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 4)
        self.dec3 = DecoderBlock(num_filters * (8 + 4), num_filters * 4 * 2, num_filters * 2)
        self.dec2 = DecoderBlock(num_filters * (4 + 2), num_filters * 2 * 2, num_filters)
        self.dec1 = ConvRelu(num_filters * (2 + 1), num_filters)
        
        self.final = nn.Conv2d(num_filters, 1, kernel_size=1, )
    
    def forward(self, x):
        conv1 = self.relu(self.conv1(x))
        conv2 = self.relu(self.conv2(self.pool(conv1)))
        conv3s = self.relu(self.conv3s(self.pool(conv2)))
        conv3 = self.relu(self.conv3(conv3s))
        conv4s = self.relu(self.conv4s(self.pool(conv3)))
        conv4 = self.relu(self.conv4(conv4s))
        conv5s = self.relu(self.conv5s(self.pool(conv4)))
        conv5 = self.relu(self.conv5(conv5s))

        center = self.center(self.pool(conv5))

        # Deconvolutions with copies of VGG11 layers of corresponding size 
        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))
        return F.sigmoid(self.final(dec1))

def get_model():
    model = UNet11()
    model.train() #
    return model.cuda()#注意这里别忘了调gpu!!!
def image_process(impath, is_mask=False):
    image = cv2.imread(impath) #(101, 101, 3)
    
    if is_mask:
        image = image[:,:,0:1] // 255
    else:
        image = image / 255.0
    
    top_pad = 13
    bottom_pad = 14
    left_pad = 13
    right_pad = 14
    #https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html?highlight=copymakeborder#copymakeborder
    image = cv2.copyMakeBorder(image, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_REFLECT_101)
    
    #tsfm = transforms.Compose([transforms.ToTensor()]) 
    #image = tsfm(image)  
    
    #return torch.tensor(np.transpose(image, (2, 0, 1)).astype('float32'), dtype=torch.FloatTensor)
    #pytorch 0.3 不能用torch.tensor(...)
    
    #return torch.FloatTensor(np.transpose(image, (2, 0, 1)).astype('float32'))
    #不知道为啥不行
    return torch.FloatTensor(image.astype('float32'))
class TGSSaltDataset(torch.utils.data.Dataset):
    
    def __init__(self, root_path, file_list, is_test=False):
        self.root_path = root_path
        self.file_list = file_list
        self.is_test = is_test
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        if index not in range(0, len(self.file_list)):
            return self.__getitem__(np.random.randint(0, self.__len__()))
        
        file_id = self.file_list[index]
        
        image_folder = os.path.join(self.root_path, "images")
        image_path = os.path.join(image_folder, file_id + ".png")
        
        #image = np.array(imageio.imread(image_path), dtype=np.uint8)
        image = image_process(image_path)
        
        if not self.is_test:
            mask_folder = os.path.join(self.root_path, "masks")
            mask_path = os.path.join(mask_folder, file_id + ".png")
 
            #mask = np.array(imageio.imread(mask_path), dtype=np.uint8)
            mask = image_process(mask_path, is_mask=True)
            return (image, mask) 
        else:
            return (image,)
series = pd.read_csv("../input/train.csv")['id']
file_list = [series[i] for i in range(len(series))]

val_file_list = file_list[::10]#val_file_list = [file_list[i] for i in range(len(file_list)) if i%10==0]
train_file_list = [f for f in file_list if f not in val_file_list]

train_dataset = TGSSaltDataset("../input/train", train_file_list)
val_dataset = TGSSaltDataset("../input/train/", val_file_list)

model = get_model()
epoch_num = 13 #13的时候val loss 最低
learning_rate = 1e-4
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#这里debug好艰难啊……各种shape对不上
for epoch in range(epoch_num):
    train_loss = []
    for image, mask in tqdm_notebook(data.DataLoader(train_dataset, batch_size=30, shuffle=True, )):
        #此时image [30, 128, 128, 3], mask不知道为啥少了一维是 [30, 128, 128]
        image = torch.transpose(image, 1, 3).cuda() #image变成[30, 3, 128, 128]       
        #image = image / 255
        y_pred = model(Variable(image)) #y_pred是[30, 1, 128, 128]
        #mask = mask // 255 #Convert mask to 0 and 1 format ??????
        mask = mask[:, np.newaxis, :, :]#mask变成[30, 1, 128, 128]
        mask = mask.type(torch.FloatTensor)#从bytetensor变成float tensor
        
        #一开始没有把数据归一化到[0,1]，导致loss都是负的（交叉熵里有ln（1-x），x必须在0，1之间）
        loss = criterion(y_pred, Variable(mask.cuda())) #torch.Size([1])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss.append(loss.data[0])#pytorch 0.3 里得这样提取出[1]tensor的值
    val_loss = []
    for image, mask in tqdm_notebook(data.DataLoader(val_dataset, batch_size=30, shuffle=True, )):
        image = torch.transpose(image, 1, 3).cuda()
        y_pred = model(Variable(image)) 
        mask = mask[:, np.newaxis, :, :]
        mask = mask.type(torch.FloatTensor)
        
        loss = criterion(y_pred, Variable(mask.cuda()))         
        val_loss.append(loss.data[0])
        
    print("Epoch: %d, Train Loss: %.3f, Val Loss: %.3f" % (epoch, np.mean(train_loss), np.mean(val_loss)))
print("Training Completed!")
save_checkpoint('tgs-%i.pth' % epoch_num, model, optimizer)
load_checkpoint('./tgs-13.pth', model, optimizer)
test_path = "../input/test/"
test_file_list = glob.glob(os.path.join(test_path, 'images', '*.png'))
test_file_list = [f.split('/')[-1].split('.')[0] for f in test_file_list]
test_dataset = TGSSaltDataset(test_path, test_file_list, is_test=True)
test_dataloader = data.DataLoader(test_dataset, batch_size=30) #为啥test也有batchsize？？
test_predictions = []
model.eval()
for image in test_dataloader:
    #????不知道为啥image是一个长为1的list，list[0]是[30, 128, 128, 3]的byteTensor
    image = image[0]
    image = np.transpose(image, (0,3,2,1)).type(torch.FloatTensor).cuda() #image变成[30, 3, 128, 128]       
    image = image / 255
    y_pred = model(Variable(image)) #[30, 1, 128, 128]
    test_predictions.append(y_pred)
test_predictions_stacked = np.vstack(test_predictions)[:, 0, :, :]
test_predictions_stacked = test_predictions_stacked[:, bottom_pad + 1:128 - top_pad - 1, left_pad + 1:128 - right_pad - 1]
test_predictions_stacked.shape
val_predictions = []
val_masks = []
for image, mask in tqdm_notebook(data.DataLoader(val_dataset, batch_size = 30)):
    image = torch.transpose(image, 1, 3).cuda()
    y_pred = model(Variable(image)) #[30, 1, 128, 128]
    val_predictions.append(y_pred.cpu().data.numpy())
    val_masks.append(mask)#[30,128,128]
    
val_predictions_stacked = np.vstack(val_predictions)[:, 0, :, :]
val_masks_stacked = np.vstack(val_masks)

val_predictions_stacked = val_predictions_stacked[:, bottom_pad + 1:128 - top_pad - 1, left_pad + 1:128 - right_pad - 1]
val_masks_stacked = val_masks_stacked[:, bottom_pad + 1:128 - top_pad - 1, left_pad + 1:128 - right_pad - 1]

val_masks_stacked.shape, val_predictions_stacked.shape
from sklearn.metrics import jaccard_similarity_score

metric_by_threshold = []
for threshold in np.linspace(0, 1, 11):
    val_binary_prediction = (val_predictions_stacked > threshold).astype(int)
    
    iou_values = []
    for y_mask, p_mask in zip(val_masks_stacked, val_binary_prediction):
        iou = jaccard_similarity_score(y_mask.flatten(), p_mask.flatten())
        iou_values.append(iou)
    iou_values = np.array(iou_values)
    
    accuracies = [
        np.mean(iou_values > iou_threshold) #之后要不要把mean改成max试试？
        for iou_threshold in np.linspace(0.5, 0.95, 10)
    ]
    print('Threshold: %.1f, Metric: %.3f' % (threshold, np.mean(accuracies)))
    metric_by_threshold.append((np.mean(accuracies), threshold))
    
best_metric, best_threshold = max(metric_by_threshold)#这怎么求max？？
threshold = best_threshold
binary_prediction = (all_predictions_stacked > threshold).astype(int)

def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b > prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

all_masks = []
for p_mask in list(binary_prediction):
    p_mask = rle_encoding(p_mask)
    all_masks.append(' '.join(map(str, p_mask)))
submit = pd.DataFrame([test_file_list, all_masks]).T
submit.columns = ['id', 'rle_mask']
submit.to_csv('submit_baseline_torch.csv', index = False)