import pandas as pd

import numpy as np



import torch

import torch.nn as nn

import torch.optim as optim

from torch.optim import lr_scheduler

from torch.utils.data import DataLoader, Dataset

from torchvision import transforms, models



from torchvision.utils import make_grid

import matplotlib.pyplot as plt

%matplotlib inline
train_csv_path = '../input/train.csv'

test_csv_path = '../input/test.csv'



train_df = pd.read_csv(train_csv_path)

test_df = pd.read_csv(test_csv_path)



# have a glimpse of train dataframe structure

n_train = len(train_df)

n_pixels = len(train_df.columns) - 1

n_class = len(set(train_df['label']))

print('Number of training samples: {0}'.format(n_train))

print('Number of training pixels: {0}'.format(n_pixels))

print('Number of classes: {0}'.format(n_class))

print(train_df.head())



# have a glimpse of test dataframe structure

n_test = len(test_df)

n_pixels = len(test_df.columns)

print('Number of test samples: {0}'.format(n_test))

print('Number of test pixels: {0}'.format(n_pixels))

print(test_df.head())
random_sel = np.random.randint(n_train, size=8)



grid = make_grid(torch.Tensor((train_df.iloc[random_sel, 1:].values/255.).reshape((-1, 28, 28))).unsqueeze(1), nrow=8)

plt.rcParams['figure.figsize'] = (16, 2)

plt.imshow(grid.numpy().transpose((1,2,0)))

plt.axis('off')

print(*list(train_df.iloc[random_sel, 0].values), sep = ', ')
class MNISTDataset(Dataset):

    """MNIST dtaa set"""

    

    def __init__(self, dataframe, 

                 transform = transforms.Compose([transforms.ToPILImage(),

                                                 transforms.ToTensor(),

                                                 transforms.Normalize(mean=(0.5,), std=(0.5,))])

                ):

        df = dataframe

        # for MNIST dataset n_pixels should be 784

        self.n_pixels = 784

        

        if len(df.columns) == self.n_pixels:

            # test data

            self.X = df.values.reshape((-1,28,28)).astype(np.uint8)[:,:,:,None]

            self.y = None

        else:

            # training data

            self.X = df.iloc[:,1:].values.reshape((-1,28,28)).astype(np.uint8)[:,:,:,None]

            self.y = torch.from_numpy(df.iloc[:,0].values)

            

        self.transform = transform

    

    def __len__(self):

        return len(self.X)



    def __getitem__(self, idx):

        if self.y is not None:

            return self.transform(self.X[idx]), self.y[idx]

        else:

            return self.transform(self.X[idx])
RandAffine = transforms.RandomAffine(degrees=45, translate=(0.1, 0.1), scale=(0.8, 1.2))
rotate = transforms.RandomRotation(degrees=45)

shift = RandAffine

composed = transforms.Compose([rotate,

                               shift])



# Apply each of the above transforms on sample.

fig = plt.figure()

sample = transforms.ToPILImage()(train_df.iloc[65,1:].values.reshape((28,28)).astype(np.uint8)[:,:,None])

for i, tsfrm in enumerate([rotate, shift, composed]):

    transformed_sample = tsfrm(sample)



    ax = plt.subplot(1, 3, i + 1)

    plt.tight_layout()

    ax.set_title(type(tsfrm).__name__)

    ax.imshow(np.reshape(np.array(list(transformed_sample.getdata())), (-1,28)), cmap='gray')    



plt.show()
batch_size = 64



train_transforms = transforms.Compose(

    [transforms.ToPILImage(),

     RandAffine,

     transforms.ToTensor(),

     transforms.Normalize(mean=(0.5,), std=(0.5,))])



val_test_transforms = transforms.Compose(

    [transforms.ToPILImage(),

     transforms.ToTensor(),

     transforms.Normalize(mean=(0.5,), std=(0.5,))])



def get_dataset(dataframe, dataset=MNISTDataset,

                transform=transforms.Compose([transforms.ToPILImage(),

                                              transforms.ToTensor(),

                                              transforms.Normalize(mean=(0.5,), std=(0.5,))])):

    return dataset(dataframe, transform=transform)
"""

Code snippet is used for introduction, there is no need to run this cell

"""



# def resnet18(pretrained=False, **kwargs):

#     """Constructs a ResNet-18 model.

#     Args:

#         pretrained (bool): If True, returns a model pre-trained on ImageNet

#     """

#     model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

#     if pretrained:

#         model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))

#     return model
"""

Code snippet is used for introduction, there is no need to run this cell

"""



# class ResNet(nn.Module):



#     def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,

#                  groups=1, width_per_group=64, replace_stride_with_dilation=None,

#                  norm_layer=None):

#         super(ResNet, self).__init__()

#         ...

        

#         # declaration of first convolutional layer

#         self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,bias=False)

#         ...

        

#         self.fc = nn.Linear(512 * block.expansion, num_classes)

#         ...

    

#     def forward(self, x):

#         x = self.conv1(x)

#         ....
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck



class MNISTResNet(ResNet):

    def __init__(self):

        super(MNISTResNet, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=10) # Based on ResNet18

        # super(MNISTResNet, self).__init__(BasicBlock, [3, 4, 6, 3], num_classes=10) # Based on ResNet34

        # super(MNISTResNet, self).__init__(Bottleneck, [3, 4, 6, 3], num_classes=10) # Based on ResNet50

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3,bias=False)



model = MNISTResNet()

print(model)
def train(train_loader, model, criterion, optimizer, epoch):

    model.train()



    for batch_idx, (data, target) in enumerate(train_loader):

        # if GPU available, move data and target to GPU

        if torch.cuda.is_available():

            data = data.cuda()

            target = target.cuda()

        

        # compute output and loss

        output = model(data)

        loss = criterion(output, target)

        

        # TODO:

        # 1. add batch metric (acc1, acc5)

        # 2. add average metric top1=sum(acc1)/batch_idx, top5 = sum(acc5)/batch_idx

        

        # backward and update model

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        

        if (batch_idx + 1)% 100 == 0:

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(

                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),

                100. * (batch_idx + 1) / len(train_loader), loss.data.item()))
def validate(val_loader, model, criterion):

    model.eval()

    loss = 0

    correct = 0

    

    for _, (data, target) in enumerate(val_loader):

        if torch.cuda.is_available():

            data = data.cuda()

            target = target.cuda()

        

        output = model(data)

        

        loss += criterion(output, target).data.item()



        pred = output.data.max(1, keepdim=True)[1]

        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        

    loss /= len(val_loader.dataset)

        

    print('\nOn Val set Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(

        loss, correct, len(val_loader.dataset),

        100.0 * float(correct) / len(val_loader.dataset)))
# example config, use the comments to get higher accuracy

total_epoches = 20 # 50

step_size = 5     # 10

base_lr = 0.01    # 0.01



optimizer = optim.Adam(model.parameters(), lr=base_lr)

criterion = nn.CrossEntropyLoss()

exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)



if torch.cuda.is_available():

    model = model.cuda()

    criterion = criterion.cuda()
def split_dataframe(dataframe=None, fraction=0.9, rand_seed=1):

    df_1 = dataframe.sample(frac=fraction, random_state=rand_seed)

    df_2 = dataframe.drop(df_1.index)

    return df_1, df_2



for epoch in range(total_epoches):

    print("\nTrain Epoch {}: lr = {}".format(epoch, exp_lr_scheduler.get_lr()[0]))



    train_df_new, val_df = split_dataframe(dataframe=train_df, fraction=0.9, rand_seed=epoch)

    

    train_dataset = get_dataset(train_df_new, transform=train_transforms)

    val_dataset = get_dataset(val_df, transform=val_test_transforms)



    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,

                                               batch_size=batch_size, shuffle=True)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,

                                             batch_size=batch_size, shuffle=False)



    train(train_loader=train_loader, model=model, criterion=criterion, optimizer=optimizer, epoch=epoch)

    validate(val_loader=val_loader, model=model, criterion=criterion)

    exp_lr_scheduler.step()
"""

The following code train model with extra original MNIST data.

It's unfair.

"""



# from torchvision.datasets import MNIST



# train_transforms = transforms.Compose(

#     [RandAffine,

#      transforms.ToTensor(),

#      transforms.Normalize(mean=(0.5,), std=(0.5,))])



# val_test_transforms = transforms.Compose(

#     [transforms.ToTensor(),

#      transforms.Normalize(mean=(0.5,), std=(0.5,))])



# train_dataset_all = MNIST('.', train=True, download=True,

#                           transform=train_transforms)

# test_dataset_all = MNIST('.', train=False, download=True,

#                          transform=train_transforms)



# train_loader = torch.utils.data.DataLoader(dataset=train_dataset_all,

#                                            batch_size=batch_size, shuffle=True)

# val_loader = torch.utils.data.DataLoader(dataset=test_dataset_all,

#                                          batch_size=batch_size, shuffle=False)





# for epoch in range(total_epoches):

#     print("\nTrain Epoch {}: lr = {}".format(epoch, exp_lr_scheduler.get_lr()[0]))

#     train(train_loader=train_loader, model=model, criterion=criterion, optimizer=optimizer, epoch=epoch)

#     validate(val_loader=val_loader, model=model, criterion=criterion)

#     exp_lr_scheduler.step()
def prediciton(test_loader, model):

    model.eval()

    test_pred = torch.LongTensor()

    

    for i, data in enumerate(test_loader):

        if torch.cuda.is_available():

            data = data.cuda()

            

        output = model(data)

        

        pred = output.cpu().data.max(1, keepdim=True)[1]

        test_pred = torch.cat((test_pred, pred), dim=0)

        

    return test_pred
test_batch_size = 64

test_dataset = get_dataset(test_df, transform=val_test_transforms)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,

                                          batch_size=test_batch_size, shuffle=False)



# tensor prediction

test_pred = prediciton(test_loader, model)



# tensor -> numpy.ndarray -> pandas.DataFrame

test_pred_df = pd.DataFrame(np.c_[np.arange(1, len(test_dataset)+1), test_pred.numpy()], 

                      columns=['ImageId', 'Label'])



# show part of prediction dataframe

print(test_pred_df.head())
# import the modules we'll need

from IPython.display import HTML

import base64



# function that takes in a dataframe and creates a text link to  

# download it (will only work for files < 2MB or so)

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



# create a link and click to download

create_download_link(test_pred_df, filename="submission.csv")
test_pred_df.to_csv('submission.csv', index=False)