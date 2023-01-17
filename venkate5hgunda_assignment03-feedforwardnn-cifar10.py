# Uncomment and run the commands below if imports fail
# !conda install numpy pandas pytorch torchvision cpuonly -c pytorch -y
# !pip install matplotlib --upgrade --quiet
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
%matplotlib inline
# Project name used for jovian.commit
project_name = '03-cifar10-feedforward'
dataset = CIFAR10(root='data/', download=True, transform=ToTensor())
test_dataset = CIFAR10(root='data/', train=False, transform=ToTensor())
dataset_size = len(dataset)
print("dataset total length: ", dataset_size)
test_dataset_size = len(test_dataset)
print("test_dataset total length: ", test_dataset_size)
classes = dataset.classes
num_classes = len(classes)
print("the dataset contains", num_classes, "classes", "and they are: ", classes)
img, label = dataset[0]
img_shape = img.shape
print("the shape of each image in the dataset is:", img_shape)
img, label = dataset[0]
plt.imshow(img.permute((1, 2, 0)))
print('Label (numeric):', label)
print('Label (textual):', classes[label])
class_count = [0]*num_classes
for _, label in dataset:
    class_count[label] += 1
print("Number of elements in each class:")
for i in range(1,num_classes):
    print(classes[i],":",class_count[i])
!pip install jovian --upgrade --quiet
import jovian
jovian.commit(project=project_name, environment=None)
torch.manual_seed(127) # Random Number
val_size = (int)(dataset_size*0.1) # 10 percent of training dataset is a good measure. Downcasting to int to remove floating point precision issues
train_size = dataset_size - val_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])
print("length of training dataset  :",len(train_ds))
print("length of validation dataset:",len(val_ds))
batch_size=512
train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size*2, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size*2, num_workers=4, pin_memory=True)
for images, _ in train_loader:
    print('images.shape:', images.shape)
    plt.figure(figsize=(16,8))
    plt.axis('off')
    plt.imshow(make_grid(images, nrow=16).permute((1, 2, 0)))
    break
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))
class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))
def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase 
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result)
        history.append(result)
    return history
torch.cuda.is_available()
def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
device = get_default_device()
device
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)
def plot_losses(history):
    losses = [x['val_loss'] for x in history]
    plt.plot(losses, '-x')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss vs. No. of epochs');
def plot_accuracies(history):
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs');
train_loader = DeviceDataLoader(train_loader, device)
val_loader = DeviceDataLoader(val_loader, device)
test_loader = DeviceDataLoader(test_loader, device)
class CIFAR10Model_IHHHHO(ImageClassificationBase): # 1 Hidden Layer
    def __init__(self, input_size, hidden1_size, hidden2_size, hidden3_size, hidden4_size, output_size):
        super().__init__()
        # hidden layer1
        self.linear1 = nn.Linear(input_size, hidden1_size)
        # hidden layer2
        self.linear2 = nn.Linear(hidden1_size, hidden2_size)
        # hidden layer 3
        self.linear3 = nn.Linear(hidden2_size, hidden3_size)
        # hidden layer 4
        self.linear4 = nn.Linear(hidden3_size, hidden4_size)
        #output layer
        self.linear5 = nn.Linear(hidden4_size,output_size)
        
    def forward(self, xb):
        # Flatten images into vectors
        out = xb.view(xb.size(0), -1)
        # Apply layers & activation functions
        # Hidden Layer: linear1
        out = self.linear1(out)
        # Output Layer: linear2
        out = self.linear2(out)
        return out
input_size = torch.numel(dataset[0][0])
print("Input Layer   :",input_size)
hidden1_size = (int)((input_size + num_classes)*2) # Random Number. taking combinations of input and output layers
print("Hidden Layer 1:",hidden1_size)
hidden2_size = (int)(hidden1_size/4)
print("Hidden Layer 2:",hidden2_size)
hidden3_size = (int)(hidden2_size/4)
print("Hidden Layer 3:",hidden3_size)
hidden4_size = (int)(hidden3_size/4)
print("Hidden Layer 4:",hidden4_size)
output_size = num_classes
print("Output Layer  :",output_size)
model = to_device(CIFAR10Model_IHHHHO(input_size, hidden1_size, hidden2_size, hidden3_size, hidden4_size, output_size), device)
history = [evaluate(model, val_loader)]
history
# First Run. More Epochs, High Learning Rate 
history += fit(100, 1e-2, model, train_loader, val_loader)
history += fit(50, 1e-3, model, train_loader, val_loader)
history += fit(25, 1e-4, model, train_loader, val_loader)
history += fit(20, 1e-5, model, train_loader, val_loader)
plot_losses(history)
plot_accuracies(history)
evaluate(model, test_loader)
jovian.commit(project=project_name, outputs=['cifar10-feedforward.pth'], environment=None)
arch = '5 layers: (',input_size,',',hidden1_size,',',hidden2_size,',',hidden3_size,',',hidden4_size,',',output_size,') batch_size: 512' 
lrs = [1e-2, 1e-3, 1e-4, 1e-5]
epochs = [100, 50, 25, 20]
test = evaluate(model, test_loader)
test_acc = test['val_acc']
test_loss = test['val_loss']
print("test_acc:",test_acc)
print("test_loss:",test_loss)
torch.save(model.state_dict(), 'cifar10-feedforward.pth')
# Clear previously recorded hyperparams & metrics
jovian.reset()
jovian.log_hyperparams(arch=arch, 
                       lrs=lrs, 
                       epochs=epochs)
jovian.log_metrics(test_loss=test_loss, test_acc=test_acc)
jovian.commit(project=project_name, outputs=['cifar10-feedforward.pth'], environment=None)
