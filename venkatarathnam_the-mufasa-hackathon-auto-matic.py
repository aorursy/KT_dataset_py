import base64

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os, glob, time, copy

import torch

import torch.nn as nn

import torch.optim as optim

from torch.optim import lr_scheduler

from torch.utils.data import Dataset, DataLoader

from torchvision import datasets, transforms, models

from PIL import Image

from sklearn.metrics import classification_report

from IPython.display import HTML
###Training Phase 1###

###A Classifier to identify the manufacturer of the car###

INDIVIDUAL_CLASSES = [x.split(' ')[0] for x in os.listdir('../input/car_data/car_data/train/')]

COMPANIES = list(set(INDIVIDUAL_CLASSES))

NUM_CLASSES = len(COMPANIES)

ROOT_DIR = '../input/car_data/car_data'

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DATA_TRANSFORMS = {

    'train': transforms.Compose([

        transforms.RandomResizedCrop(224),

        transforms.RandomHorizontalFlip(),

        transforms.ToTensor(),

        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    ]),

    'test': transforms.Compose([

        transforms.RandomResizedCrop(224),

        transforms.ToTensor(),

        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    ]),

}
###Analyzing the dataset###

c = 0

for company in COMPANIES:

    models_count = INDIVIDUAL_CLASSES.count(company)

    print ("{0}: {1}".format(company, models_count))

    if (models_count != 1): c += 1

print ("Companies with more than 1 model: {0}".format(c))
class CarDS(Dataset):

    def __init__(self, root, phase, transforms):

        self.filenames = []

        self.root = root

        self.phase = phase

        self.transform = transforms

        self.classes = os.listdir(root)

        self.labels = []

        for dir in os.listdir(root):

            path = os.path.join(self.root, dir)

            filenames = glob.glob(os.path.join(path, '*'))

            for fn in filenames:

                self.filenames.append(fn)

                self.labels.append(COMPANIES.index(dir.split(' ')[0]))

        self.labels = np.array(self.labels)

        self.len = len(self.filenames)

        

    def __getitem__(self, index):

        image = Image.open(self.filenames[index])

        image = image.convert('RGB')

        image = self.transform(image)

        return image, self.labels[index]



    def __len__(self):

        return self.len
def plot_xy(x, y, title="", xlabel="", ylabel=""):

    plt.figure()

    plt.xlabel(xlabel)

    plt.ylabel(ylabel)

    plt.title(title)

    for i in range(len(y)):

        plt.plot(x, y[i], label = str(i))

    plt.show()



def get_count_per_class(data_dir, phase = 'train'):

    train_labels_count = [0]*NUM_CLASSES

    phase_path = os.path.join(data_dir, phase)

    for ind, dir in enumerate(os.listdir(phase_path)):

        path, dirs, files = next(os.walk(os.path.join(phase_path, dir)))

        file_count = len(files)

        act_ind = COMPANIES.index(dir.split(' ')[0])

        train_labels_count[act_ind] += file_count

    return train_labels_count



def plot_images_per_class(labels_count=None, phase = 'train'):

    if (labels_count is None):

        labels_count = get_count_per_class(phase)

    plt.figure()

    f, ax = plt.subplots(figsize=(25,10))

    plt.bar(COMPANIES, labels_count)

    plt.xticks(rotation=90)

    #plt.xticks(np.arange(102), np.arange(102))

    plt.ylabel("No. of samples")

    plt.xlabel("Classes")

    plt.title(phase)

    plt.show()

    

def train_model(dataloaders, model, criterion, optimizer, scheduler, num_epochs=25):

    since = time.time()



    best_model_wts = copy.deepcopy(model.state_dict())

    best_acc = 0.0

    losses = {'train': [], 'valid':[]}

    acc = {'train': [], 'valid': []}



    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))

        print('-' * 10)



        # Each epoch has a training and validation phase

        for phase in ['train', 'valid']:

            if phase == 'train':

                scheduler.step()

                model.train()  # Set model to training mode

            else:

                model.eval()   # Set model to evaluate mode



            running_loss = 0.0

            running_corrects = 0



            # Iterate over data.

            for inputs, labels in dataloaders[phase]:

                inputs = inputs.to(DEVICE)

                labels = labels.to(DEVICE)



                # zero the parameter gradients

                optimizer.zero_grad()



                # forward

                # track history if only in train

                with torch.set_grad_enabled(phase == 'train'):

                    outputs = model(inputs)

                    _, preds = torch.max(outputs, 1)

                    loss = criterion(outputs, labels)



                    # backward + optimize only if in training phase

                    if phase == 'train':

                        loss.backward()

                        optimizer.step()



                running_loss += loss.item() * inputs.size(0)

                running_corrects += torch.sum(preds == labels.data)



            epoch_loss = running_loss / dataset_sizes[phase]

            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            losses[phase].append(epoch_loss)

            acc[phase].append(epoch_acc)



            print('{} Loss: {:.4f} Acc: {:.4f}'.format(

                phase, epoch_loss, epoch_acc))



            # deep copy the model

            if phase == 'valid' and epoch_acc > best_acc:

                best_acc = epoch_acc

                best_model_wts = copy.deepcopy(model.state_dict())

                

            if (phase == 'valid' and epoch + 1 == num_epochs):

                print ("-" * 15)

                print ("Final Classification Report")

                print ("-" * 15)

                print (classification_report(preds.cpu(), labels.cpu()))



        print()



    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(

        time_elapsed // 60, time_elapsed % 60))

    print('Best val Acc: {:4f}'.format(best_acc))

    

    plot_xy(np.arange(num_epochs), [losses['train'], losses['valid']], xlabel = 'Epochs', ylabel = 'Loss', title = 'Loss Plot')

    plot_xy(np.arange(num_epochs), [acc['train'], acc['valid']], xlabel = 'Epochs', ylabel = 'Accuracy', title = 'Accuracy Plot')



    # load best model weights

    model.load_state_dict(best_model_wts)

    return model
dataset = CarDS(os.path.join(ROOT_DIR, 'train'), phase = 'train', transforms = DATA_TRANSFORMS['train'])

train_size = int(0.90 * len(dataset))

valid_size = len(dataset) - train_size

print ("Train Size: {0}".format(train_size))

print ("Validation Size: {0}".format(valid_size))

labels_count = get_count_per_class('../input/car_data/car_data', phase='train')

plot_images_per_class(labels_count)

dataset_sizes = {'train': train_size, 'valid': valid_size}

train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])

dataloaders = {'train': DataLoader(train_dataset, batch_size = 64, shuffle = True),

              'valid': DataLoader(valid_dataset, batch_size = 64, shuffle = False)}
model = models.resnet101(pretrained=True)

num_ftrs = model.fc.in_features

model.fc = nn.Linear(num_ftrs, NUM_CLASSES)

model = model.to(DEVICE)

class_weights = [1-(float(labels_count[class_id])/(len(dataset))) for class_id in range(NUM_CLASSES)]

criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(DEVICE))

optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
model = train_model(dataloaders, model, criterion, optimizer, exp_lr_scheduler, num_epochs=25)
###Training Phase 2###

###Retraining the classfier to identify the model of the car###

INDIVIDUAL_CLASSES = os.listdir('../input/car_data/car_data/train/')

NUM_CLASSES = len(INDIVIDUAL_CLASSES)

ROOT_DIR = '../input/car_data/car_data'

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DATA_TRANSFORMS = {

    'train': transforms.Compose([

        transforms.RandomResizedCrop(224),

        transforms.RandomHorizontalFlip(),

        transforms.ToTensor(),

        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    ]),

    'test': transforms.Compose([

        transforms.RandomResizedCrop(224),

        transforms.ToTensor(),

        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    ]),

}
class CarDS(Dataset):

    def __init__(self, root, phase, transforms):

        self.filenames = []

        self.root = root

        self.phase = phase

        self.transform = transforms

        self.classes = os.listdir(root)

        self.labels = []

        for dir in os.listdir(root):

            path = os.path.join(self.root, dir)

            filenames = glob.glob(os.path.join(path, '*'))

            for fn in filenames:

                self.filenames.append(fn)

                self.labels.append(INDIVIDUAL_CLASSES.index(dir))

        self.labels = np.array(self.labels)

        self.len = len(self.filenames)

        

    def __getitem__(self, index):

        image = Image.open(self.filenames[index])

        image = image.convert('RGB')

        image = self.transform(image)

        if (self.phase == 'test'):

            return image

        return image, self.labels[index]



    def __len__(self):

        return self.len
dataset = CarDS(os.path.join(ROOT_DIR, 'train'), phase = 'train', transforms = DATA_TRANSFORMS['train'])

train_size = int(0.9 * len(dataset))

valid_size = len(dataset) - train_size

dataset_sizes = {'train': train_size, 'valid': valid_size}

train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])

dataloaders = {'train': DataLoader(train_dataset, batch_size = 64, shuffle = True),

              'valid': DataLoader(valid_dataset, batch_size = 64, shuffle = False)}
num_ftrs = model.fc.in_features

model.fc = nn.Linear(num_ftrs, NUM_CLASSES)

model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
model = train_model(dataloaders, model, criterion, optimizer, exp_lr_scheduler, num_epochs=25)
###Testing the model with test samples###

pd.set_option('display.max_rows', 9000)

names = pd.read_csv("../input/names.csv", header=None)

names = names.rename(columns={0: 'Model'})

names.Model = names.Model.replace({"Ram C/V Cargo Van Minivan 2012": "Ram C-V Cargo Van Minivan 2012"})

LOAD_MODEL = False

with torch.no_grad():

    model.eval()

    print ("Test set prediction results:")

    test_set = CarDS('../input/car_data/car_data/test', transforms=DATA_TRANSFORMS['test'], phase='test')

    test_loader = DataLoader(test_set, batch_size = 64, shuffle=False)

    index = 0

    results = []

    for inputs in test_loader:

        outputs = model(inputs.to(DEVICE))

        _, pred = torch.max(outputs, 1)

        for i in range(len(inputs)):

            Id = str(test_set.filenames[index].split("/")[-1].split(".")[0])

            Predicted_cls = INDIVIDUAL_CLASSES[int(pred[i])]

            Predicted_index = names.loc[names['Model'] == Predicted_cls].index[0]+1

            results.append((Id,Predicted_index))

            index += 1

    result_df = pd.DataFrame(results, columns=['Id', 'Predicted'])

    #result_df = result_df.sort_values(by=['Filename'])

    print (result_df)

    result_df.to_csv('../working/test_results.csv')
###Downloading the results file###

def create_download_link(df, title = "Download Result file", filename = "results.csv"):  

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



create_download_link(result_df)