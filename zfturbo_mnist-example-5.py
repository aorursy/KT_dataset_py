import argparse

import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from torchvision import datasets, transforms

from torch.optim.lr_scheduler import StepLR

# from torchsummary import summary





class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 4, 3, padding=1)

        self.conv2 = nn.Conv2d(4, 4, 3, padding=1)

        self.conv3 = nn.Conv2d(4, 8, 3, padding=1)

        self.conv4 = nn.Conv2d(8, 8, 3, padding=1)

        self.conv5 = nn.Conv2d(8, 16, 3, padding=1)

        self.conv6 = nn.Conv2d(16, 16, 3, padding=1)

        self.dropout1 = nn.Dropout2d(0.5)

        self.maxpool1 = nn.MaxPool2d(7)

        self.fc1 = nn.Linear(16, 32)

        self.fc2 = nn.Linear(32, 10)



    def forward(self, x):

        x = self.conv1(x)

        x = F.relu(x)

        x = self.conv2(x)

        x = F.relu(x)

        x = F.max_pool2d(x, 2)

        x = self.conv3(x)

        x = F.relu(x)

        x = self.conv4(x)

        x = F.relu(x)

        x = F.max_pool2d(x, 2)

        x = self.conv5(x)

        x = F.relu(x)

        x = self.conv6(x)

        x = F.relu(x)

        x = self.maxpool1(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)

        x = F.relu(x)

        x = self.dropout1(x)

        x = self.fc2(x)

        output = F.log_softmax(x, dim=1)

        return output





def train(model, device, train_loader, optimizer, epoch):

    log_interval = 10

    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        output = model(data)

        loss = F.nll_loss(output, target)

        loss.backward()

        optimizer.step()

        if batch_idx % log_interval == 0:

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(

                epoch, batch_idx * len(data), len(train_loader.dataset),

                100. * batch_idx / len(train_loader), loss.item()))





def tst(model, device, test_loader):

    model.eval()

    test_loss = 0

    correct = 0

    with torch.no_grad():

        for data, target in test_loader:

            data, target = data.to(device), target.to(device)

            output = model(data)

            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss

            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

            correct += pred.eq(target.view_as(pred)).sum().item()



    test_loss /= len(test_loader.dataset)



    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(

        test_loss, correct, len(test_loader.dataset),

        100. * correct / len(test_loader.dataset)))





def main():

    batch_size = 64

    learning_rate = 1.0

    reduce_lr_gamma = 0.7

    epochs = 14

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    kwargs = {'batch_size': batch_size}

    if torch.cuda.is_available():

        kwargs.update({'num_workers': 1, 'pin_memory': True, 'shuffle': True})



    transform = transforms.Compose([

        transforms.ToTensor(),

        transforms.Normalize((0.1307,), (0.3081,))

    ])



    dataset1 = datasets.MNIST('../data', train=True, download=True, transform=transform)

    dataset2 = datasets.MNIST('../data', train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset1, **kwargs)

    test_loader = torch.utils.data.DataLoader(dataset2, **kwargs)



    model = Net().to(device)

    # summary(model, input_size=(1, 28, 28))

    optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)



    scheduler = StepLR(optimizer, step_size=1, gamma=reduce_lr_gamma)

    for epoch in range(1, epochs + 1):

        train(model, device, train_loader, optimizer, epoch)

        tst(model, device, test_loader)

        scheduler.step()



    torch.save(model.state_dict(), "mnist_cnn.pt")





if __name__ == '__main__':

    main()
