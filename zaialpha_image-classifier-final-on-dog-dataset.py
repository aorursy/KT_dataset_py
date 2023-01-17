import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
torch.set_printoptions(linewidth=120)
torch.set_grad_enabled(True)
data_set = torchvision.datasets.ImageFolder(  #making train set
    root = '/kaggle/input/stanford-dogs-dataset/images/Images', 
    transform = transforms.Compose([transforms.Resize((224,224)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
)

data_loader = torch.utils.data.DataLoader( #making train loader/minibatch
    data_set, batch_size = 30, shuffle = True,
)

sample = next(iter(data_set))
image, lable = sample
print (image.shape)
npimg = image.numpy()
plt.imshow(np.transpose(npimg, (1, 2, 0)))
print ('lable : ',lable)
def load_splitset(dataset,test_split=0.3):
    #test_split = .2
    shuffle_dataset = True
    random_seed= 42
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(test_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]
    
    testset_size = len(test_indices)
    indices = list(range(testset_size))
    split = int(np.floor(0.5 * testset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    val_indices, test_indices = indices[split:], indices[:split]


    # Creating  data samplers and loaders:
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=30,
                                           sampler=train_sampler)

    test_loader = torch.utils.data.DataLoader(dataset, batch_size=30,
                                                sampler=test_sampler)
    
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=30,
                                                sampler=val_sampler)
    return train_loader,test_loader, val_loader
train_loader,test_loader, val_loader = load_splitset(data_set,test_split=0.3)
print(len(train_loader))
print(len(val_loader))
print(len(test_loader))
class Network(nn.Module): # nn.module class contains the functionality to keep track of layers weights
  def __init__(self):
    super(Network,self).__init__()
    self.conv1 = nn.Conv2d(3, 16, 3)
    self.conv2 = nn.Conv2d(16, 32, 3)
    self.conv3 = nn.Conv2d(32, 64, 3)
    self.conv4 = nn.Conv2d(64, 128, 3)
    self.conv5 = nn.Conv2d(128, 256, 3)
    self.fc1 = nn.Linear(256 * 6 * 6, 120)
    
    self.max_pool = nn.MaxPool2d(2, 2,ceil_mode=True)
    self.dropout = nn.Dropout(0.2)

    self.conv_bn1 = nn.BatchNorm2d(224,3)
    self.conv_bn2 = nn.BatchNorm2d(16)
    self.conv_bn3 = nn.BatchNorm2d(32)
    self.conv_bn4 = nn.BatchNorm2d(64)
    self.conv_bn5 = nn.BatchNorm2d(128)
    self.conv_bn6 = nn.BatchNorm2d(256)


  def forward(self, x):
        
    x = F.relu(self.conv1(x))
    x = self.max_pool(x)
    x = self.conv_bn2(x)
    
    x = F.relu(self.conv2(x))
    x = self.max_pool(x)
    x = self.conv_bn3(x)
    
    x = F.relu(self.conv3(x))
    x = self.max_pool(x)
    x = self.conv_bn4(x)
    
    x = F.relu(self.conv4(x))
    x = self.max_pool(x)
    x = self.conv_bn5(x)
    
    x = F.relu(self.conv5(x))
    x = self.max_pool(x)
    x = self.conv_bn6(x)
    
    x = x.view(-1, 256 * 6 * 6)
    
    x = self.dropout(x)
    x = self.fc1(x)
    return x
network = Network()
use_cuda = True
if use_cuda and torch.cuda.is_available():
  network.cuda()
  print('cuda')

optimizer = optim.SGD(network.parameters(), lr=0.001, momentum=0.9)


for epoch in range(15):
  total_loss = 0
  total_correct = 0
  x = 0
  total_val = 0
  total_train = 0
  for data in (train_loader):

      images, labels = data
      if use_cuda and torch.cuda.is_available():
        images = images.cuda()
        labels = labels.cuda()
      pred = network(images)
      loss = F.cross_entropy(pred,labels)
      total_loss += loss.item()
      total_train += len(pred)
      optimizer.zero_grad() # because each time its adds gradients into previous gradients
      loss.backward() # calculating gradient
      optimizer.step() # update weights / thetas
      
      total_correct += pred.argmax(dim = 1).eq(labels).sum()
  print("epoch : ",epoch,"Traning Accuracy : ",total_correct*1.0/total_train,"Train Loss : ",total_loss*1.0/len(train_loader) )
    
  total_loss = 0
  val_total_correct = 0
  for batch in (val_loader):
      images, labels = batch
      if use_cuda and torch.cuda.is_available():
          images = images.cuda()
          labels = labels.cuda()
      pred = network(images)
      loss = F.cross_entropy(pred,labels)
      total_loss += loss.item()
      total_val += len(pred)
      x += 1
      val_total_correct += pred.argmax(dim = 1).eq(labels).sum()
  print("epoch : ",epoch,"Val Accuracy : ",val_total_correct*1.0/total_val,"Val Loss : ",total_loss*1.0/len(val_loader) )
  y += 30
  torch.cuda.empty_cache()
  
test_total_correct = 0
total_test = 0
x = 0
for batch in (test_loader):
    images, labels = batch 
    if use_cuda and torch.cuda.is_available():
        images = images.cuda()
        labels = labels.cuda()
    pred = network(images)
    total_test += len(pred)
    x += 1
    test_total_correct += pred.argmax(dim = 1).eq(labels).sum()
print("Test Accuracy : ",test_total_correct*1.0/total_test, )