import copy
import matplotlib.pyplot as plt
import numpy as np 
import os
import pandas as pd
from PIL import Image
from skimage import io, transform
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
from torchvision import transforms, utils, models
print(os.listdir("../input/honey-bee-annotated-images/"))
PATH = '../input/honey-bee-annotated-images/bee_data.csv'
# df = dataframe
df = pd.read_csv(PATH)
df.head()
df.subspecies.value_counts()
# Return the count of unique values
df['health'].value_counts()
df['health'] = df['health'].map({'healthy': 0,
                                 'few varrao, hive beetles': 1,
                                 'Varroa, Small Hive Beetles': 2,
                                 'ant problems': 3,
                                 'hive being robbed': 4,
                                 'missing queen': 5})

df["health"].value_counts()
df.head()
# Plot the graphs wrt different columns
f, ax = plt.subplots(nrows=2, ncols=2, figsize=(12,10))

df.subspecies.value_counts().plot(kind='bar',ax=ax[0, 0])
ax[0,0].set_title('Subspecies')

df.location.value_counts().plot(kind='bar', ax=ax[0, 1])
ax[0,1].set_title('Location')

df.caste.value_counts().plot(kind='bar', ax=ax[1, 0])
ax[1,0].set_title('Caste')
ax[1,0].set_ylabel('Count')

df.health.value_counts().plot(kind='bar', ax=ax[1,1])
ax[1,1].set_title('Health')
ax[1,1].set_ylabel('Count')

f.tight_layout()
plt.show()
transform = {'train': transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          #torchvision.transforms.ColorJitter(hue=.05, saturation=.05),
                                          #transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])]),
             
             'val': transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                             [0.229, 0.224, 0.225])]),
            
             'test': transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], 
                                                              [0.229, 0.224, 0.225])])}

# Check for cuda
device = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")
print(device)
class HoneyBeeDataset(Dataset):
    # instance attributes
    def __init__(self, df, csv_file, root_dir, transform=None):
        self.data = df
        self.root_dir = root_dir
        self.labels = np.asarray(self.data.iloc[:, 6])
        self.transform = transform
        
    # length of the dataset passed to this class method    
    def __len__(self):
        return len(self.data)
    
    # get the specific image and labels given the index
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name)
        image = image.convert('RGB')
        image_label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, image_label
dataset = HoneyBeeDataset(df=df,
                          csv_file=PATH,
                          root_dir='../input/honey-bee-annotated-images/bee_imgs/bee_imgs')
validation_split = 0.2
te_split = 0.5
dataset_size = len(dataset)
indices = list(range(dataset_size))
np.random.shuffle(indices)
val_split = int(np.floor(validation_split * dataset_size))
test_split = int(np.floor(te_split * val_split))
train_indices = indices[val_split:]
rest_indices = indices[:val_split]
val_indices, test_indices = rest_indices[test_split:], rest_indices[:test_split]
# Sanity Check
# Expected outcome: 4138, 517, 517
len(train_indices), len(val_indices), len(test_indices)
dataset_sizes = {'train': len(train_indices), 'val': len(val_indices), 'test': len(test_indices)}

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)
test_sampler = SubsetRandomSampler(test_indices)

train_dataset = HoneyBeeDataset(df=df,
                                csv_file=PATH,
                                root_dir='../input/honey-bee-annotated-images/bee_imgs/bee_imgs',
                                transform=transform['train'])

val_dataset = HoneyBeeDataset(df=df,
                              csv_file=PATH,
                              root_dir='../input/honey-bee-annotated-images/bee_imgs/bee_imgs',
                              transform=transform['val'])

test_dataset = HoneyBeeDataset(df=df,
                               csv_file=PATH,
                               root_dir='../input/honey-bee-annotated-images/bee_imgs/bee_imgs',
                               transform=transform['test'])
dataloaders = {'train' : torch.utils.data.DataLoader(train_dataset, batch_size=4, sampler=train_sampler),
               'val': torch.utils.data.DataLoader(val_dataset, batch_size=4, sampler=valid_sampler),
               'test': torch.utils.data.DataLoader(test_dataset, batch_size=1, sampler=test_sampler)}
def train_model(model, criterion, optimizer, scheduler, num_epochs):
    #copy the best model weights
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print("Epoch: {}/{}".format(epoch, num_epochs-1))
        print("="*10)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()
                
            running_loss = 0.0
            running_corrects = 0
            
            for data in dataloaders[phase]:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase=='train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase=='train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item()*inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
        
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
    
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
        
# Load VGG16 Model
model_pre = models.vgg16()
model_pre.load_state_dict(torch.load("../input/vgg16/vgg16-397923af.pth"))
model_pre
# don't calculate gradient since we will use the weights of pretrained model
for param in model_pre.features.parameters():
    param.required_grad = False

num_features = model_pre.classifier[6].in_features
# Remove last layer
features = list(model_pre.classifier.children())[:-1] 
# Add new layer with out_features = len(health_classes)
features.extend([nn.Linear(num_features, 6)])
# Replace the model classifier with new classifier
model_pre.classifier = nn.Sequential(*features) 
print(model_pre.classifier)
# load gpu
model_pre = model_pre.to(device)
# loss function
criterion = nn.CrossEntropyLoss()
# Observe that all parameters are being optimized
optimizer = torch.optim.SGD(model_pre.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 10 epochs
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
EPOCHS = 35
# train
model_pre = train_model(model_pre, criterion, optimizer, exp_lr_scheduler, num_epochs=EPOCHS)
def test_model():
    running_correct = 0
    running_total = 0
    true_labels = []
    pred_labels = []
    # no gradient calculation
    with torch.no_grad():
        for data in dataloaders['test']:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            true_labels.append(labels.item())
            outputs = model_pre(inputs)
            _, preds = torch.max(outputs.data, 1)
            pred_labels.append(preds.item())
            running_total += labels.size(0)
            running_correct += (preds == labels).sum().item()
    return (true_labels, pred_labels, running_correct, running_total)
true_labels, pred_labels, running_correct, running_total= test_model()
print('Correct: {}, Total: {}'.format(running_correct, running_total))
print('Test Accuracy: ', (running_correct/running_total))