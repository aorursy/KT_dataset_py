import numpy as np # linear algebra
import pandas as pd # data processing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from PIL import Image
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
from skimage import io
from torch.utils.data import (
    Dataset,
    DataLoader,
)
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import os
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision
import torch.optim as optim
import copy
from sklearn.externals import joblib
from skimage.transform import rotate, AffineTransform, warp
from skimage.util import random_noise
from skimage.filters import gaussian
from skimage.transform import resize
from sklearn.metrics import f1_score as f
from PIL import Image 
#enable GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
train=pd.read_csv('../input/identify-the-dance-form/test.csv')#specify input location
test=pd.read_csv('../input/identify-the-dance-form/test.csv')
Class_map={'manipuri':0, 'bharatanatyam':1, 'odissi':2 ,'kathakali':3, 'kathak':4, 'sattriya':5,
 'kuchipudi':6, 'mohiniyattam':7}
inverse_map={0:'manipuri', 1:'bharatanatyam', 2:'odissi' ,3:'kathakali',4: 'kathak', 5:'sattriya',
 6:'kuchipudi', 7:'mohiniyattam'}
train['target']=train['target'].map(Class_map)#repalcing names with repective labels
train
test_imgdir=[]#collecting all the locaions of test images
for img in list(test['Image']):
  test_imgdir.append('../input/identify-the-dance-form/test/'+str(img))
print(test_imgdir)
train_imgdir=[]#collecting all the locations of train images
for img in list(train['Image']):
  train_imgdir.append('../input/identify-the-dance-form/train/'+str(img))
print(train_imgdir)
train_label=[]#collecting labels of train images
for i in list(train['target']):
  train_label.append(i)
print(train_label)
#data agumentation by using various techniques
extend_train_imgdir=[]
extend_train_label=[]
for img,label in zip(train_imgdir,train_label):
  image=io.imread(img)
  name=img.split('/')[-1].split('.')[0]
  noice_img=random_noise(image,var=0.2**2)
  blurred_img = gaussian(image,sigma=1,multichannel=True)
  rotated_img = rotate(image, angle=45, mode = 'wrap')
  flipped_img=np.fliplr(image)
  #rotated_img2 = rotate(image, angle=315, mode = 'wrap')
  #updown_img=np.flipud(image)
  io.imsave('./dataset/train/'+name+'noice_img.jpg',noice_img)
  io.imsave('./dataset/train/'+name+'blurred_img.jpg',blurred_img)
  io.imsave('./dataset/train/'+name+'rotated_img.jpg',rotated_img)
  io.imsave('./dataset/train/'+name+'flipped_img.jpg',flipped_img)
  #io.imsave('./dataset/train/'+name+'inverserotated_img.jpg',rotated_img2)
  extend_train_imgdir.append('./dataset/train/'+name+'noice_img.jpg')
  extend_train_imgdir.append('./dataset/train/'+name+'blurred_img.jpg')
  extend_train_imgdir.append('./dataset/train/'+name+'rotated_img.jpg')
  extend_train_imgdir.append('./dataset/train/'+name+'flipped_img.jpg')
  #extend_train_imgdir.append('./dataset/train/'+name+'inverserotated_img.jpg')
  extend_train_label.append(label)
  extend_train_label.append(label)
  extend_train_label.append(label)
  extend_train_label.append(label)
  #extend_train_label.append(label)
print(extend_train_imgdir)
print(extend_train_label)
print(len(extend_train_imgdir))
print(len(extend_train_label))
train_imgdir.extend(extend_train_imgdir)
train_label.extend(extend_train_label)
print(train_imgdir)
print(train_label)
print(len(train_imgdir))
print(len(train_label))
x_train,x_val,y_train,y_val=train_test_split(train_imgdir,train_label,test_size=0.15,shuffle=True)
print(len(x_train))
print(len(x_val))
def returnpaths(imgdir,labeldir):
  paths=[]
  for dir,label in zip(imgdir,labeldir):
    paths.append((dir,label))
  return paths
train_paths=returnpaths(x_train,y_train)
val_paths=returnpaths(x_val,y_val)
print(train_paths)
class Load_testdata(Dataset):
    def __init__(self,paths,transform=None):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self,index):
        img=self.paths[index]
        name=img.split('/')[-1]
        img_path = os.path.join(img)
        image = io.imread(img_path)
        #image=cv2.resize(image,(224,224))
        #image=image/255
        image=resize(image, output_shape=(400,400,3), mode='constant', anti_aliasing=True)
        #y_label = torch.tensor(int(img[1]))

        if self.transform:
            transform_train = transforms.Compose([ 
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                            ])
            image = transform_train(image)

        return image,name
testdataset = Load_testdata(
    paths=test_imgdir,
    transform=transforms.ToTensor()
)
test_loader = DataLoader(dataset=testdataset)#specify batchsize if you need and modify other functions accordingly
its=iter(test_loader)
img,name=next(its)
print(img.shape)
print(name[0])
class Load_data(Dataset):
    def __init__(self,paths,transform=None):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self,index):
        img=self.paths[index]
        img_path = os.path.join(img[0])
        image = io.imread(img_path)
        #image=cv2.resize(image,(224,224)) 
        #image=image/255
        image=resize(image, output_shape=(400,400,3), mode='constant', anti_aliasing=True)       
        y_label = torch.tensor(int(img[1]))
        #print(y_label)
        if self.transform :
          transform_train = transforms.Compose([ 
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                            ])
          image = transform_train(image)

        return (image, y_label)
traindataset = Load_data(
    paths=train_paths,
    transform=transforms.ToTensor()
)
valdataset = Load_data(
    paths=val_paths,
    transform=transforms.ToTensor()
)
train_loader = DataLoader(dataset=traindataset)
val_loader=DataLoader(dataset=valdataset)
it=iter(val_loader)
image,label=next(it)
print(type(image))
print(type(label))
print(image)
print(label)
print(image.shape)
def imshow(img, title):
    npimg = img.numpy() / 2 + 0.5
    plt.figure(figsize=(8,4))
    plt.axis('off')
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.show()
def show_batch_images(dataloader):
    images, labels = dataloader
    img = torchvision.utils.make_grid(images)
    # images=images.to(device)
    # images=images.float()
    # val=inception(images)
    # print(val)
    label=labels
    imshow(img, title=(int(label),inverse_map[int(label)]))
var=iter(train_loader)
for i in range(4):
    show=next(var)
    show_batch_images(show)#printing images with labels
model=models.resnet34(pretrained=True)
# for param in model.parameters():
#     param.requires_grad = False
in_features = model.fc.in_features
model.fc = nn.Linear(in_features,8)
print(model)
#evaluation function for calculating accuracy
def evaluation(dataloader, model):
    total, correct = 0, 0
    y_true=[]
    y_pred=[]
    for data in dataloader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        inputs=inputs.float()
        output = model(inputs)
        pred = output.argmax(1)
        #print(pred)
        label=labels
        y_pred.append(pred)
        y_true.append(label)
        total += labels.size(0)
        correct += (pred == label).sum().item()
    #print(f(y_true,y_pred,average='weighted'))
    return 100 * correct / total
evaluation(val_loader,model)#checking wether function is working well 
model = model.to(device)#moving model to GPU
loss_fn = nn.CrossEntropyLoss()
#opt = optim.Adam(model.parameters(), lr=0.000155)
loss_epoch_arr = []
max_epochs =2
opt = optim.Adam(model.parameters(),lr=0.00000125)
min_loss = 100

n_iters =len(x_train)

for epoch in range(max_epochs):
    print('Epoch: %d/%d' % (epoch+1, max_epochs))
    for i, data in enumerate(train_loader, 0):

        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        inputs=inputs.float()
        opt.zero_grad()

        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        opt.step()
        # outputs, aux_outputs = model(inputs)
        # loss = loss_fn(outputs, labels) + 0.3 * loss_fn(aux_outputs, labels)
        # loss.backward()
        # opt.step()
        
        if min_loss > loss.item():
            min_loss = loss.item()
            best_model = copy.deepcopy(model.state_dict())
            print('Min loss %0.9f' % min_loss)
        
        if i % 100 == 0:
            print('Iteration: %d/%d, Loss: %0.7f' % (i, n_iters, loss.item()))
            
        del inputs, labels, outputs
        torch.cuda.empty_cache()
        
    loss_epoch_arr.append(loss.item())
        
    
    
    
plt.plot(loss_epoch_arr)
plt.show()
#model.load_state_dict(best_model)
#joblib.dump(model, 'dance_25.pkl')
#print(evaluation(train_loader,model), evaluation(val_loader,model))
# model2=models.resnet34(pretrained=True)
# # for param in model.parameters():
# #     param.requires_grad = False
# in_features = model2.fc.in_features
# model2.fc = nn.Linear(in_features,8)
model2.load_state_dict(best_model)#loading best model if you need to,but this best model will not assure to give heighest accuracy
#model2=model2.to(device)#uncomment this if you want to use model2
#model=joblib.load('dance_25.pkl')
#predicting test values
names=[]
preds=[]
for data in test_loader:
  img,name=data
  img=img.to(device)
  img=img.float()
  out=model(img)
  pred=int(out.argmax(1))
  preds.append(str(inverse_map[pred]))
  names.append(name[0])
print(names)
print(preds)
print(len(names))
print(len(preds))
final_result5 = np.array(list(map(list, zip(names,preds))))
df_final= pd.DataFrame(data=final_result5, columns=["Image", "target"])
df_final.to_csv('submission_dance_31.csv', index=False)

