import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import os
from torchvision import datasets, transforms  
from torch.autograd import Variable
import numpy as np
from PIL import Image
print(torch.__version__)
class Net(nn.Module): 
    def __init__(self):
        super(Net, self).__init__()

        # Define two 2D convolutional layers (1 x 10, 10 x 20 each)
        # with convolution kernel of size (5 x 5).
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)

        # Define a dropout layer
        self.conv2_drop = nn.Dropout2d()

        # Define a fully-connected layer (320 x 10)
        self.fc = nn.Linear(320, 10)

    def forward(self, x):
        # Input image size: 28 x 28, input channel: 1, batch size (training): 64 

        # Input (64 x 1 x 28 x 28) -> Conv1 (64 x 10 x 24 x 24) -> Max Pooling (64 x 10 x 12 x 12) -> ReLU -> ...
        x = F.relu(F.max_pool2d(self.conv1(x), 2))

        # ... -> Conv2 (64 x 20 x 8 x 8) -> Dropout -> Max Pooling (64 x 20 x 4 x 4) -> ReLU -> ...
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))

        # ... -> Flatten (64 x 320) -> ...
        x = x.view(-1, 320)

        # ... -> FC (64 x 10) -> ...
        x = self.fc(x)

        # ... -> Log Softmax -> Output
        return F.log_softmax(x, dim=1)
train_batch_size = 64
test_batch_size = 1000
epochs = 10
lr = 0.01
momentum = 0.5
log_interval = 100
# the same results can be reproduced
torch.manual_seed(22)
# Fetch training data: total 60000 samples
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=train_batch_size, shuffle=True)
# Fetch test data: total 10000 samples
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=test_batch_size, shuffle=True)
model = Net()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
def train(model, optimizer, epoch, train_loader, log_interval):
    # State that you are training the model
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        # Wrap the input and target output in the Variable wrapper
        data, target = Variable(data), Variable(target)

        # Clear the gradients, since PyTorch accumulates them
        optimizer.zero_grad()

        # Forward propagation
        output = model(data)

        # Calculate negative log likelihood loss
        loss = F.nll_loss(output, target)

        # Backward propagation
        loss.backward()

        # Update the parameters(weight,bias)
        optimizer.step()

        # print log
        if batch_idx % log_interval == 0:
            print('Train set, Epoch {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), 
                loss.data[0]))
#                 loss.data.item()))
def test(model, epoch, test_loader):
    # State that you are testing the model; this prevents layers e.g. Dropout to take effect
    model.eval()

    test_loss = 0
    correct = 0

    # Iterate over data
    for data, target in test_loader:
        data, target = Variable(data), Variable(target)
        
        # Forward propagation
        output = model(data)

        # Calculate & accumulate loss
#         test_loss += F.nll_loss(output, target, reduction='sum').data.item()
        test_loss += F.nll_loss(output, target, size_average=False).data[0]

        # Get the index of the max log-probability (the predicted output label)
#         pred = output.data.argmax(1)
        pred = np.argmax(output.data, axis=1)

        # If correct, increment correct prediction accumulator
#         correct += pred.eq(target.data).sum()
        correct = correct + np.equal(pred, target.data).sum() 

    # print log
    test_loss /= len(test_loader.dataset)
    print('\nTest set, Epoch {} , Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(epoch,
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
for epoch in range(1, epochs + 1):
    train(model, optimizer, epoch, train_loader, log_interval=log_interval)
    test(model, epoch, test_loader)
torch.save(model.state_dict(), 'saved_model')
# model.load_state_dict(torch.load('saved_model'))
# Show image
Image.open('../input/test_2.png')
# Load & transform image
ori_img = Image.open('../input/test_2.png').convert('L')
t = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
img = torch.autograd.Variable(t(ori_img).unsqueeze(0))
ori_img.close()
# Predict
model.eval()
output = model(img)
pred = output.data.max(1, keepdim=True)[1][0][0]
print('Prediction: {}'.format(pred))
# Show image
Image.open('../input/test_4.png')
# Load & transform image
ori_img = Image.open('../input/test_4.png').convert('L')
t = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
img = torch.autograd.Variable(t(ori_img).unsqueeze(0))
ori_img.close()
# Predict
model.eval()
output = model(img)
pred = output.data.max(1, keepdim=True)[1][0][0]
print('Prediction: {}'.format(pred))
# Show image
Image.open('../input/test_6.png')
# Load & transform image
ori_img = Image.open('../input/test_6.png').convert('L')
t = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
img = torch.autograd.Variable(t(ori_img).unsqueeze(0))
ori_img.close()
# Predict
model.eval()
output = model(img)
pred = output.data.max(1, keepdim=True)[1][0][0]
print('Prediction: {}'.format(pred))
