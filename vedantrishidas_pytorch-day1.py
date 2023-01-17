from traitlets.config.manager import BaseJSONConfigManager
from pathlib import Path
path = Path.home() / ".jupyter" / "nbconfig"
cm = BaseJSONConfigManager(config_dir=str(path))
cm.update(
    "rise",
    {
        "scroll": True,
        "enable_chalkboard": True
     }
)
# PLEASE IGNORE ABOVE CELL, JUST TO CONFIGURE SLIDES
# RUN THIS CELL ONCE TO ENSURE SMOOTH SLIDES FUNCTIONING
# AFTER EXECUTING THIS CELL ONCE, SAVE AND THEN RELOAD PAGE TO ENSURE CHANGES ARE SAVED
# %%capture
# !pip install pytorch_pretrained_bert
# !pip3 install http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp36-cp36m-linux_x86_64.whl 
# !pip3 install torchvision
! pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
# !pip install imbalanced-learn
import torch
x = torch.empty(3, 4)
print(x)
print(x.size())
x = torch.rand(3, 4)
print(x)
print(x.size())
x = torch.zeros(3, 4, dtype=torch.long)
print(x)
x = torch.zeros(3, 4, dtype=torch.float32)
print(x)
x = torch.ones(3, 4, dtype=torch.long)
print(x)
x = torch.ones(3, 4, dtype=torch.float32)
print(x)
x = torch.eye(4)
print(x)
x = torch.eye(4, dtype=torch.long)
print(x)
print(x.type())
x = torch.tensor([10.5, 9.2, 7])
print(x)
print(x.size())
x.type()
x = torch.tensor([[10.5, 9.2, 7], [1, 2, 3]])
print(x)
print(x.size())
print(x)
x = torch.ones_like(x)
print(x)
print(x.size())
x = torch.ones(x.size())
print(x)
x = torch.rand(3, 4)
y = torch.randn(x.size())
print('x = ', x)
print('y = ', y)
torch.add(x, 10)
x+10
x+y
torch.add(x,y)
x+y*10
torch.add(x, y, alpha=10)
res = torch.add(x,y)
print(res)
x = x+y
print(x)
x.add_(y)
print(x[:, 1])
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())

x = torch.randn(1)
print(x)
print(x.item())

import numpy as np
a = torch.ones(5)
print(a)
b = a.numpy()
print(b)
a.add_(1)
print(a)
print(b)
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)
# let us run this cell only if CUDA is available
# We will use ``torch.device`` objects to move tensors in and out of GPU
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)                       # or just use strings ``.to("cuda")``
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
trainset = torchvision.datasets.FashionMNIST(root = "./data", 
                                             train = True, 
                                             download = True, 
                                             transform = transforms.ToTensor())
testset = torchvision.datasets.FashionMNIST(root = "./data", 
                                            train = False, 
                                            download = True, 
                                            transform = transforms.ToTensor())
#loading the training data from trainset
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle = True)
#loading the test data from testset
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)
class_labels = ['T-Shirt','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot']
fig = plt.figure(figsize=(15,8));
columns = 5;
rows = 3;
for i in range(1, columns*rows +1):
    index = np.random.randint(len(trainset))
    img = trainset[index][0][0, :, :]
    fig.add_subplot(rows, columns, i)
    plt.title(class_labels[trainset[index][1]])
    plt.axis('off')
    plt.imshow(img, cmap='gray')
plt.show()
print(class_labels[trainset[0][1]])
plt.imshow(trainset[0][0][0, :, :], cmap='gray')
class SimpleNeuralNet(nn.Module):
    
    def __init__(self):
        super(SimpleNeuralNet, self).__init__()
#         flattening a 28x28 image into 784 dimensional 1d tensor
        self.flatten1 = nn.Flatten()
        self.layer1 = nn.Linear(784, 512)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, 128)
        self.layer4 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
#         print('Initial size = ', x.size())
        x = self.flatten1(x)
#         print('After size = ', x.size())
        out = F.relu(self.layer1(x))
        out = F.relu(self.layer2(out))
#         out = self.dropout(out)
        out = F.relu(self.layer3(out))
        out = self.layer4(out)
        
        return out
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
model = SimpleNeuralNet()
print(device)
model = model.to(device)
print(model)

cross_entropy_loss = nn.CrossEntropyLoss()

adam_optim = torch.optim.Adam(model.parameters(), lr=0.005)
print(adam_optim)
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

def eval_perfomance(dataloader, print_results=False):
    actual, preds = [], []
    #keeping the network in evaluation mode  
    model.eval() 
    for data in dataloader:
        inputs, labels = data
        actual +=[i.item() for i in labels]
        
        #moving the inputs and labels to gpu
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, pred = torch.max(outputs.data, 1)
        pred = pred.to(device)
        preds += [i.item() for i in pred]
    acc = accuracy_score(actual, preds)
    cm = confusion_matrix(actual, preds)
    cr = classification_report(actual, preds)
    if(print_results):
        print(f'Total accuracy = {acc*100}%')
        print('\n\nConfusion matrix:\n')
        print(cm)
        print('\n\nClassification Report:\n')
        print(cr)
    
    return acc
intial_acc = eval_perfomance(testloader, True)
# %%time
loss_arr = []
loss_epoch_arr = []
max_epochs = 5
iter_list = []
train_acc_arr = []
test_acc_arr = []
ctr = 0
for epoch in range(max_epochs):
    for i, data in enumerate(trainloader, 0):
        
        model.train()
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)      
        
        loss = cross_entropy_loss(outputs, labels)    
        adam_optim.zero_grad()     

        loss.backward()     
        adam_optim.step()     
        loss_arr.append(loss.item())
        
        ctr+=1
        iter_list+=[ctr]
    
        
    train_acc = eval_perfomance(trainloader)
    train_acc_arr+=[train_acc.item()]
        
    test_acc = eval_perfomance(testloader)
    test_acc_arr+=[test_acc.item()]
        
    if((epoch+1)%1==0):
        print(f"Iteration: {epoch+1}, Loss: {loss.item()}, Train Acc:{train_acc}, Val Acc: {test_acc}")

    loss_epoch_arr.append(loss.item()) 
plt.plot([i for i in range(len(loss_epoch_arr))], loss_epoch_arr)
plt.xlabel("No. of Iteration")
plt.ylabel("Loss")
plt.title("Loss vs Iterations")
plt.show()
plt.plot([i for i in range(len(train_acc_arr))], train_acc_arr)
plt.xlabel("No. of Iteration")
plt.ylabel("Train Accuracy")
plt.title("Train Accuracy")
plt.show()
plt.plot([i for i in range(len(test_acc_arr))], test_acc_arr)
plt.xlabel("No. of Iteration")
plt.ylabel("Test Accuracy")
plt.title("Test Accuracy")
plt.show()
class_correct = [0. for _ in range(10)]
total_correct = [0. for _ in range(10)]

with torch.no_grad():
    for images, act_labels in testloader:
#         images, labels = images.to(device), labels.to(device)
#         test = Variable(images)
        images, act_labels = images.to(device), act_labels.to(device)
        outputs = model(images)
        #Returns a namedtuple (values, indices) where values is the maximum value of each row of the input tensor in the given dimension dim.
        predicted = torch.max(outputs, 1)[1] 
        #print("Predicted: ", predicted)
        #print("predicted ==  act_labels", predicted == act_labels)
        c = (predicted == act_labels).squeeze()
        #print("c :",c)
        
        for i in range(4):
            label = act_labels[i]
            class_correct[label] += c[i].item()
            total_correct[label] += 1
        
for i in range(10):
    print("Accuracy of {}: {:.2f}%".format(class_labels[i], class_correct[i] * 100 / total_correct[i]))
final_acc_test = eval_perfomance(testloader, print_results=True)