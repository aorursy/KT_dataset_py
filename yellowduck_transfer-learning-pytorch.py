import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# print(os.listdir("../input/train"))
import shutil 

from tqdm import tqdm



train_dir = 'train'

val_dir = 'val'

test_dir = 'test'



shutil.copytree('../input/test', os.path.join(test_dir, 'unknown'))



for dir_name in [train_dir, val_dir]:

    for class_name in ['dog', 'cat']:

        os.makedirs(os.path.join(dir_name, class_name), exist_ok=True)



for i, file_name in enumerate(tqdm(os.listdir('../input/train'))):

    dest_dir = train_dir if i % 5 != 0 else val_dir

    

    for class_name in ['dog', 'cat']:

        if file_name.startswith(class_name):

            shutil.copy(os.path.join('../input/train', file_name),

                        os.path.join(dest_dir, class_name, file_name))
import torch

import numpy as np

import torchvision

import matplotlib.pyplot as plt

import time

import copy



from torchvision import transforms, models

train_transforms = transforms.Compose([

    transforms.RandomResizedCrop(224),

    transforms.RandomHorizontalFlip(),

    transforms.ToTensor(),

    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

])



val_transforms = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

])



train_dataset = torchvision.datasets.ImageFolder(train_dir, train_transforms)

val_dataset = torchvision.datasets.ImageFolder(val_dir, val_transforms)

test_dataset = torchvision.datasets.ImageFolder(test_dir, val_transforms)



batch_size = 32

train_dataloader = torch.utils.data.DataLoader(

    train_dataset, batch_size=batch_size, shuffle=True, num_workers=batch_size)

val_dataloader = torch.utils.data.DataLoader(

    val_dataset, batch_size=batch_size, shuffle=False, num_workers=batch_size)

test_dataloader = torch.utils.data.DataLoader(

    test_dataset, batch_size=batch_size, shuffle=False, num_workers=batch_size)



class_names = train_dataset.classes



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
len(train_dataloader), len(train_dataset)
device
X_batch, y_batch = next(iter(val_dataloader))

mean = np.array([0.485, 0.456, 0.406])

std = np.array([0.229, 0.224, 0.225])

plt.imshow(X_batch[0].permute(1, 2, 0).numpy() * std + mean);
def show_input(input_tensor, title=''):

    image = input_tensor.permute(1, 2, 0).numpy()

    image = std * image + mean

    plt.imshow(image.clip(0, 1))

    plt.title(title)

    plt.show()



X_batch, y_batch = next(iter(train_dataloader))



for x_item, y_item in zip(X_batch, y_batch):

    show_input(x_item, title=class_names[y_item])

    plt.pause(0.001)
def train_model(model, loss, optimizer, scheduler, num_epochs=25):

    for epoch in range(num_epochs):

        print('Epoch {}/{}:'.format(epoch, num_epochs - 1), flush=True)



        # Each epoch has a training and validation phase

        for phase in ['train', 'val']:

            if phase == 'train':

                dataloader = train_dataloader

                scheduler.step()

                model.train()  # Set model to training mode

            else:

                dataloader = val_dataloader

                model.eval()   # Set model to evaluate mode



            running_loss = 0.

            running_acc = 0.



            # Iterate over data.

            for inputs, labels in tqdm(dataloader):

                inputs = inputs.to(device)

                labels = labels.to(device)



                optimizer.zero_grad()



                # forward and backward

                with torch.set_grad_enabled(phase == 'train'):

                    preds = model(inputs)

                    loss_value = loss(preds, labels)

                    preds_class = preds.argmax(dim=1)



                    # backward + optimize only if in training phase

                    if phase == 'train':

                        loss_value.backward()

                        optimizer.step()



                # statistics

                running_loss += loss_value.item()

                running_acc += (preds_class == labels.data).float().mean()



            epoch_loss = running_loss / len(dataloader)

            epoch_acc = running_acc / len(dataloader)



            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))



        print()



    return model
model = models.resnet18(pretrained=True)



# Disable grad for all conv layers

for param in model.parameters():

    param.requires_grad = False



model.fc = torch.nn.Linear(model.fc.in_features, 2)



model = model.to(device)

loss = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)



# Decay LR by a factor of 0.1 every 7 epochs

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
train_model(model, loss, optimizer, scheduler, num_epochs=1)
model.eval()



test_predictions = []

for inputs, labels in tqdm(test_dataloader):

    inputs = inputs.to(device)

    labels = labels.to(device)

    with torch.set_grad_enabled(False):

        preds = model(inputs)

    test_predictions.append(

        torch.nn.functional.softmax(preds, dim=1)[:,1].data.cpu().numpy())

    

test_predictions = np.concatenate(test_predictions)
X_batch, y_batch = next(iter(test_dataloader))



for x_item, pred in zip(X_batch, test_predictions):

    show_input(x_item, title=pred)

    plt.pause(0.001)
test_file_names = os.listdir(os.path.join(test_dir, 'unknown'))



submission_df = pd.DataFrame.from_dict({'id': np.arange(1, len(test_predictions) + 1),

                                        'label': np.zeros(len(test_predictions))})

for i, file_name in enumerate(test_file_names):

    index = int(file_name[:file_name.rfind('.')])

    submission_df.at[index-1, 'label'] = test_predictions[i]

submission_df.to_csv('submission.csv', index=False)
submission_df.head()
!rm -rf train val test