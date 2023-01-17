###Importing the necessary libraries###

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import torch.nn as nn

import torch.optim as optim

from torch.utils.data import (Dataset,

                              DataLoader)

from torch.optim import lr_scheduler

from torchvision import (datasets, 

                         transforms, 

                         models)

from PIL import Image

from sklearn.metrics import classification_report

import  time, copy, glob, torchvision, torch, os, json

pd.set_option('display.max_rows', 500)
###Defining the global variables###

NUM_CLASSES = 102

ROOT_DIR = '../input/flower_data/flower_data'

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DATA_TRANSFORMS = {

    'train': transforms.Compose([

        transforms.RandomResizedCrop(224),

        transforms.RandomHorizontalFlip(),

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ]),

    'valid': transforms.Compose([

        transforms.RandomResizedCrop(224),

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ]),

}

CAT_TO_NAMES = json.load(open('../input/cat_to_name.json', 'r'))
class FlowerDS(Dataset):

    def __init__(self, root, phase, transforms):

        self.filenames = []

        self.root = root

        self.phase = phase

        self.transform = transforms

        self.classes = os.listdir(root)

        self.labels = []

        if self.phase == 'test':

            filenames = glob.glob(os.path.join(root, '*'))

            self.filenames.extend(filenames)

        else:

            for dir in os.listdir(root):

                path = os.path.join(self.root, dir)

                filenames = glob.glob(os.path.join(path, '*'))

                for fn in filenames:

                    self.filenames.append(fn)

                    self.labels.append(int(dir)-1)

        self.labels = np.array(self.labels)

        self.len = len(self.filenames)

        

    def __getitem__(self, index):

        image = Image.open(self.filenames[index])

        image = image.convert('RGB')

        if self.phase == 'train':

            if (torch.randn(1)[0] > 0):

                if(torch.randn(1)[0] < 0):

                    image = transforms.functional.adjust_gamma(image, gamma = 0.5)

                else: image = transforms.functional.adjust_gamma(image, gamma = 1.0)

        image = self.transform(image)

        if (self.phase == 'test'):

            return image

        return image, self.labels[index]



    def __len__(self):

        return self.len
###Defining all the required funtions###

def get_count_per_class(data_dir, phase = 'train'):

    train_labels_count = [0]*NUM_CLASSES

    phase_path = os.path.join(data_dir, phase)

    for ind, dir in enumerate(os.listdir(phase_path)):

        path, dirs, files = next(os.walk(os.path.join(phase_path, dir)))

        file_count = len(files)

        train_labels_count[ind] = file_count

    return train_labels_count



def plot_images_per_class(labels_count=None, phase = 'train'):

    if (labels_count is None):

        labels_count = get_count_per_class(phase)

    plt.figure()

    f, ax = plt.subplots(figsize=(25,10))

    plt.bar(np.arange(102), labels_count)

    plt.xticks(np.arange(102), np.arange(102))

    plt.ylabel("No. of samples")

    plt.xlabel("Classes")

    plt.title(phase)

    plt.show()

    

def plot_xy(x, y, title="", xlabel="", ylabel=""):

    plt.figure()

    plt.xlabel(xlabel)

    plt.ylabel(ylabel)

    plt.title(title)

    for i in range(len(y)):

        plt.plot(x, y[i], label = str(i))

    plt.show()



def create_dataset(data_dir, phase = 'train'):

    #image_datasets = datasets.ImageFolder(os.path.join(data_dir, phase), DATA_TRANSFORMS[phase])

    image_dataset = FlowerDS(os.path.join(data_dir, phase), phase = phase, transforms = DATA_TRANSFORMS[phase])

    return image_dataset



def get_data_loader(data_dir, phase = 'train', batch_size = 64, doShuffle = True, no_workers = 4):

    image_dataset = create_dataset(data_dir, phase=phase)

    return DataLoader(image_dataset, batch_size=batch_size, shuffle=doShuffle, num_workers=no_workers)



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

                print ("--------------")

                print ("Final Classification Report")

                print ("--------------")

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



def do_transform(path, gammas):

    image = Image.open(path)

    for gamma in gammas:

        new_image = transforms.functional.adjust_gamma(image, gamma = gamma)

        plt.figure()

        plt.imshow(new_image, aspect=1.0)



def save(model, path):

    torch.save(model, path)



def load(path):

    return torch.load(path)
###Analysis over the dataset and certain transforms###

dataset_sizes = {x: len(create_dataset(ROOT_DIR, x)) for x in ['train', 'valid']}

print ("Train Size : {0}".format(dataset_sizes['train']))

print ("Validation Size : {0}".format(dataset_sizes['valid']))

do_transform("../input/flower_data/flower_data/train/24/image_06816.jpg", gammas = [0.5, 1.0, 2.0])

train_labels_count = get_count_per_class(ROOT_DIR, phase='train')

plot_images_per_class(train_labels_count, phase='train')
###Creating the model###

model = models.resnet101(pretrained=True)

num_ftrs = model.fc.in_features

model.fc = nn.Linear(num_ftrs, NUM_CLASSES)

model = model.to(DEVICE)

class_weights = [1-(float(train_labels_count[class_id])/(dataset_sizes['train'])) for class_id in range(NUM_CLASSES)]

criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(DEVICE))

optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
###Training the model###

doTrain = True

dataloaders = {'train': get_data_loader(ROOT_DIR, 'train'),

               'valid': get_data_loader(ROOT_DIR, 'valid', batch_size = len(create_dataset(ROOT_DIR, phase='valid')), doShuffle=False)}

if doTrain:

    model = train_model(dataloaders, model, criterion, optimizer, exp_lr_scheduler, num_epochs=25)

    save(model, "../working/flower_classification.pt")
###Testing with random validation samples###

with torch.no_grad():

    model.eval()

    print ("Evaluating random validation samples:")

    test_data, test_targets = next(iter(dataloaders['valid']))

    outputs = model(test_data[0].unsqueeze(0).to(DEVICE))

    _, pred = torch.max(outputs, 1)

    print ("Actual Class - {0}".format(CAT_TO_NAMES[str(int(test_targets[0]) + 1)]))

    print ("Predicted Class - {0}".format(CAT_TO_NAMES[str(int(pred) + 1)]))
###Testing with the test set samples###

LOAD_MODEL = False

with torch.no_grad():

    if (LOAD_MODEL):

        model = load("../working/flower_classification.pt")

    model.eval()

    print ("Test set prediction results:")

    test_set = FlowerDS('../input/test set/test set', transforms=DATA_TRANSFORMS['valid'], phase='test')

    test_loader = DataLoader(test_set, batch_size = len(test_set), shuffle=False)

    test_data = next(iter(test_loader))

    outputs = model(test_data.to(DEVICE))

    _, pred = torch.max(outputs, 1)

    results = []

    for index, filename in enumerate(test_set.filenames):

        results.append((filename.split("/")[-1], int(pred[index]) + 1, CAT_TO_NAMES[str(int(pred[index] + 1))]))

    result_df = pd.DataFrame(results, columns=['Filename', 'Class ID', 'Class Name'])

    result_df = result_df.sort_values(by=['Filename'])

    print (result_df)

    result_df.to_csv('../working/test_results.csv')