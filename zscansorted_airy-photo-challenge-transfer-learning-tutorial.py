# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

import matplotlib.pyplot as plt

from skimage import io, transform

import torch

from torchvision import transforms, utils, datasets

from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm



import torchvision



from torch.utils.data import Dataset, DataLoader



import torch.nn as nn

import torch.optim as optim

from torch.optim import lr_scheduler

from torchvision import models
train_dir = '/kaggle/input/airy-photo/train'

test_dir = '/kaggle/input/airy-photo/test'
def create_train_df_meta_from_dir(train_dir):

    """

        make metadata from train directory into 

        dataframe containing: class, filename, and full_path 

    """

    meta = list()

    for dirname, _, filenames in tqdm(os.walk(train_dir)):

        for filename in filenames:

            _class = dirname.split('/')[-1]

            full_path = os.path.join(dirname, filename)

            meta.append([

                _class,

                filename,

                full_path

            ])

    

    return pd.DataFrame(

        meta, 

        columns=['class', 'filename', 'full_path']

    )



def create_test_list_meta_from_dir(test_dir):

    """

    load all full path of test data

    """

    full_paths = list()

    for dirname, _, filenames in tqdm(os.walk(test_dir)):

        full_paths.append([

            os.path.join(dirname, filename)

            for filename in filenames

        ])

    # flatten list of list

    return np.array(full_paths).ravel() 
df_train_meta = create_train_df_meta_from_dir(train_dir)

test_image_list = create_test_list_meta_from_dir(test_dir)
mode = ['train']



data_transforms = dict(

    train=transforms.Compose([

        transforms.RandomSizedCrop(224),

        transforms.RandomHorizontalFlip(),

        transforms.ToTensor(),

        transforms.Normalize(

            [0.485, 0.456, 0.406],

            [0.229, 0.224, 0.225]

        )

    ]),

    test=transforms.Compose([

        transforms.ToPILImage(),

        transforms.Resize(256),

        transforms.CenterCrop(224),

        transforms.ToTensor(),

        transforms.Normalize(

            [0.485, 0.456, 0.406],

            [0.229, 0.224, 0.225]

        )

    ])

)



image_datasets = {

    x: datasets.ImageFolder(

        train_dir,

        data_transforms[x]

    )

    

    for x in mode

}



# pretrained models in pytorch uses batch_size=4

batch_size = 4



dataloaders = {

    x: DataLoader(

        image_datasets[x], 

        batch_size=batch_size, 

        shuffle=True, 

        num_workers=batch_size

    )

    for x in mode

}



dataset_sizes = {

    x: len(image_datasets[x])

    for x in mode

}

class_names = image_datasets['train'].classes

device = torch.device(

    'cuda:0' if torch.cuda.is_available() else 'cpu'

)
class AiryPhotoTestDataset(Dataset):

    def __init__(self, image_list, transform=None):

        self.image_list = image_list

        self.transform = transform

    

    def __len__(self):

        return len(self.image_list)

    

    def __getitem__(self, idx):

        if torch.is_tensor(idx):

            idx = idx.tolist()

            

        image = io.imread(self.image_list[idx])

        # label = None

        sample = (image, '')

        if self.transform:

            sample = (self.transform(image), '')

        return sample



mode = 'test'

image_datasets[mode] = AiryPhotoTestDataset(

    test_image_list,

    transform=data_transforms[mode]

)



dataloaders[mode] = DataLoader(

    image_datasets[mode],

    batch_size=batch_size,

    shuffle=False,

    num_workers=batch_size

)



dataset_sizes[mode] = len(image_datasets[mode])
inputs, classes = next(iter(dataloaders['train']))

out = torchvision.utils.make_grid(inputs)

print(classes)
inp = out.numpy().transpose((1,2,0))

plt.imshow(inp)

plt.title(','.join([class_names[x] for x in classes]))

plt.show()
# download pretrained model

mdl = models.resnet18(pretrained=True)



# freeze all the parameters in each model's cells

for param in mdl.parameters():

    param.requires_grad = False



# change prediction layer

# I already know that the last layer is called `fc` in this resnet18

# you can output `mdl` object to peek what variable is in each layer

num_features = mdl.fc.in_features

# need to recalibrate this into current class number

mdl.fc = nn.Linear(num_features, len(class_names))

# change the model to use current available device

mdl = mdl.to(device)





criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(

    mdl.parameters(), lr=1e3

)
def train(mdl, num_epoch, dataloader, optimizer, criterion, dataset_size, device=device):

    epoch_losses = list()

    epoch_accs = list()

    

    for epoch in range(num_epoch):

        mdl.train()



        running_loss = 0.0

        running_corrects = 0



        for inputs, labels in dataloader:

            inputs = inputs.to(device)

            labels = labels.to(device)



            optimizer.zero_grad()



            with torch.set_grad_enabled(True):

                outputs = mdl(inputs)

                _, preds = torch.max(outputs, 1)

                loss = criterion(outputs, labels.to(device))



                loss.backward()

                optimizer.step()



            running_loss += loss.item() * inputs.size(0)

            running_corrects += torch.sum(preds == labels.data)



        epoch_loss = running_loss / dataset_size

        epoch_acc = running_corrects.double() / dataset_size

        

        epoch_losses.append(epoch_loss)

        epoch_accs.append(epoch_acc.cpu().numpy())

        print(

            'epoch: {}, loss: {:.4f}, acc: {:.4f}'.format(

                epoch, epoch_loss, epoch_acc

            )

        )

    return epoch_losses, epoch_accs
num_epoch = 5

epoch_losses, epoch_accs = train(

    mdl,

    num_epoch,

    dataloaders['train'],

    optimizer,

    criterion,

    dataset_sizes['train']

)
x = np.arange(num_epoch)



f, ax = plt.subplots(figsize=(12, 14))

plt.subplot(211)

plt.plot(x, epoch_losses)

plt.title('Loss Per Epoch')



plt.subplot(212)

plt.plot(x, epoch_accs)

plt.title('Acc Per Epoch')



plt.show()
def predict(mdl, inputs, labels):

    mdl.eval()

    with torch.no_grad():

        inputs = inputs.to(device)



        outputs = mdl(inputs)

        _, preds = torch.max(outputs, 1)

    return preds
inputs, labels = next(iter(dataloaders['test']))



preds = predict(mdl, inputs, labels)



print('Predictions: {}'.format([class_names[x] for x in preds.cpu().numpy()]))
out = torchvision.utils.make_grid(inputs)

inp = out.numpy().transpose((1,2,0))

plt.imshow(inp)

plt.title(','.join([class_names[x] for x in preds]))

plt.show()