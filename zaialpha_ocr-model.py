import tarfile
my_tar = tarfile.open('/kaggle/input/english-typed-alphabets-and-numbers-dataset/EnglishFnt.tgz')
my_tar.extractall('./EnglishFnt') # specify which folder to extract to
my_tar.close()
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
torch.set_printoptions(linewidth=120)
torch.set_grad_enabled(True)
data_set = torchvision.datasets.ImageFolder(  #making train set
    root = '/kaggle/working/EnglishFnt/English/Fnt', 
    transform = transforms.Compose([transforms.Resize((48,48)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, ), (0.5, ))])
)

sample = next(iter(data_set))
image, lable = sample
print (image.shape)
npimg = image.numpy()
plt.imshow(np.transpose(npimg, (1, 2, 0)))
print ('lable : ',lable)
def load_splitset(dataset,batch_size,test_split=0.3):
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

    train_loader = torch.utils.data.DataLoader(dataset, batch_size,
                                           sampler=train_sampler)

    test_loader = torch.utils.data.DataLoader(dataset, batch_size,
                                                sampler=test_sampler)
    
    val_loader = torch.utils.data.DataLoader(dataset, batch_size,
                                                sampler=val_sampler)
    return train_loader,test_loader, val_loader
batch_size = 36
train_loader,test_loader, val_loader = load_splitset(data_set,batch_size,test_split=0.3)
print(len(train_loader))
print(len(val_loader))
print(len(test_loader))
class Network(nn.Module): # nn.module class contains the functionality to keep track of layers weights
    def __init__(self):
        super(Network,self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 9 * 9, 62)
        
        self.max_pool = nn.MaxPool2d(2, 2,ceil_mode=True)
        self.dropout = nn.Dropout(0.2)

        self.conv_bn1 = nn.BatchNorm2d(48,3)
        self.conv_bn2 = nn.BatchNorm2d(16)
        self.conv_bn3 = nn.BatchNorm2d(32)
        self.conv_bn4 = nn.BatchNorm2d(64)


    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = self.max_pool(x)
        x = self.conv_bn2(x)

        x = F.relu(self.conv2(x))
        x = self.max_pool(x)
        x = self.conv_bn3(x)

        x = F.relu(self.conv3(x))
        #x = self.max_pool(x)
        x = self.conv_bn4(x)
        

        x = x.view(-1, 64 * 9 * 9)

        x = self.dropout(x)
        x = self.fc1(x)
        return x
def to_one_hot(lables, pred_size):
    one_hot_encoded = torch.zeros(len(lables), pred_size)
    #print(one_hot_encoded.shape)
    y = 0
    for x in lables:
        one_hot_encoded[y][x] = 1
        y += 1
    return one_hot_encoded
class MyLoss(torch.autograd.Function):  
    @staticmethod
    def forward(ctx, pred, lables):
        y = to_one_hot(lables, len(pred[0]))
        y = y.cuda()
        ctx.save_for_backward(y, pred)
        loss = - y * torch.log(pred)
        loss = loss.sum()/len(lables)
        return loss


    @staticmethod
    def backward(ctx, grad_output):
        y, pred = ctx.saved_tensors
        grad_input = (- y / pred) - y 
        grad_input = grad_input/len(pred)
        return grad_input, grad_output
class MyCEL(torch.nn.Module):

    def __init__(self):
        super(MyCEL, self).__init__()

    def forward(self, pred, lables):
        y = to_one_hot(lables, len(pred[0]))
        y = y.cuda()
        #ctx.save_for_backward(y, pred)
        loss = - y * torch.log(pred)
        loss = loss.sum()/len(lables)
        return loss
network = Network()
use_cuda = True
if use_cuda and torch.cuda.is_available():
    network.cuda()
    print('cuda')

optimizer = optim.SGD(network.parameters(), lr=0.01, momentum=0.9)

epoch = 0
max_epoch = 5
end = False
myloss = MyCEL()
while epoch < max_epoch and not end:
    epoch += 1
    total_loss = 0
    total_correct = 0
    total_val = 0
    total_train = 0
    for data in (train_loader):
        
        images, labels = data
        if use_cuda and torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
        pred = network(images)
        pred = F.softmax(pred)
        #myloss = MyLoss.apply
        #print(len(pred))
        loss = myloss(pred,labels)
        #loss = F.cross_entropy(pred,labels)
        total_loss += loss.item()
        total_train += len(pred)
        optimizer.zero_grad() # because each time its adds gradients into previous gradients
        loss.backward() # calculating gradient
        optimizer.step() # update weights / thetas

        total_correct += pred.argmax(dim = 1).eq(labels).sum()
        
    print("epoch : ",epoch,"Traning Accuracy : ",total_correct*1.0/total_train,"Train Loss : ",total_loss*1.0/len(train_loader) )
    
    if total_correct*1.0/total_train >= 0.98:
        end = True
    
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
        val_total_correct += pred.argmax(dim = 1).eq(labels).sum()
    print("epoch : ",epoch,"Val Accuracy : ",val_total_correct*1.0/total_val,"Val Loss : ",total_loss*1.0/len(val_loader) )
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