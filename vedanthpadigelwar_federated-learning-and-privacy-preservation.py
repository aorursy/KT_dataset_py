# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



#import numpy as np # linear algebra

#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



#import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


!pip install syft

# In case you encounter an installation error regarding zstd,

# run the below command and then try installing syft again.

# pip install - upgrade - force-reinstall zstd
import torchvision

from torch import nn

import torch.optim as optim

import torch.nn.functional as F

from torchvision import datasets, transforms
import torch 

import syft as sy

hook = sy.TorchHook(torch)



jake = sy.VirtualWorker(hook, id="jake")

print("Jake has: " + str(jake._objects))
jake = sy.VirtualWorker(hook, id="jake")

print("Jake has: " + str(jake._objects))


x = torch.tensor([1, 2, 3, 4, 5])

x = x.send(jake)

print("x: " + str(x))

print("Jake has: " + str(jake._objects))
x = x.get()

print("x: " + str(x))

print("Jake has: " + str(jake._objects))
john = sy.VirtualWorker(hook, id="john")

x = torch.tensor([1, 2, 3, 4, 5])

x = x.send(jake)

x = x.send(john)

print("x: " + str(x))

print("John has: " + str(john._objects))

print("Jake has: " + str(jake._objects))


jake.clear_objects()

john.clear_objects()

print("Jake has: " + str(jake._objects))

print("John has: " + str(john._objects))
import random



# setting Q to a very large prime number

Q = 23740629843760239486723





def encrypt(x, n_share=3):

    r"""Returns a tuple containg n_share number of shares

    obtained after encrypting the value x."""



    shares = list()

    for i in range(n_share - 1):

        shares.append(random.randint(0, Q))

    shares.append(Q - (sum(shares) % Q) + x)

    return tuple(shares)





print("Shares: " + str(encrypt(3)))


def decrypt(shares):

    r"""Returns a value obtained by decrypting the shares."""



    return sum(shares) % Q





print("Value after decrypting: " + str(decrypt(encrypt(3))))



def add(a, b):

    r"""Returns a value obtained by adding the shares a and b."""



    c = list()

    for i in range(len(a)):

        c.append((a[i] + b[i]) % Q)

    return tuple(c)





x, y = 6, 8

a = encrypt(x)

b = encrypt(y)

c = add(a, b)

print("Shares encrypting x: " + str(a))

print("Shares encrypting y: " + str(b))

print("Sum of shares: " + str(c))

print("Sum of original values (x + y): " + str(decrypt(c)))



jake = sy.VirtualWorker(hook, id="jake")

john = sy.VirtualWorker(hook, id="john")

secure_worker = sy.VirtualWorker(hook, id="secure_worker")



jake.add_workers([john, secure_worker])

john.add_workers([jake, secure_worker])

secure_worker.add_workers([jake, john])



print("Jake has: " + str(jake._objects))

print("John has: " + str(john._objects))

print("Secure_worker has: " + str(secure_worker._objects))


x = torch.tensor([6])

x = x.share(jake, john, secure_worker)

print("x: " + str(x))



print("Jake has: " + str(jake._objects))

print("John has: " + str(john._objects))

print("Secure_worker has: " + str(secure_worker._objects))
y = torch.tensor([8])

y = y.share(jake, john, secure_worker)

print(y)
z = x + y

print(z)


z = z.get()

print(z)
transform = transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize((0.5, ), (0.5, )),

])



train_set = datasets.MNIST("~/.pytorch/MNIST_data/", train=True, download=True, transform=transform)

test_set = datasets.MNIST("~/.pytorch/MNIST_data/", train=False, download=True, transform=transform)



federated_train_loader = sy.FederatedDataLoader(train_set.federate((jake, john)), batch_size=64, shuffle=True)



test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)


class Model(nn.Module):

    def __init__(self):

        super(Model, self).__init__()

        self.fc1 = nn.Linear(784, 500)

        self.fc2 = nn.Linear(500, 10)



    def forward(self, x):

        x = x.view(-1, 784)

        x = self.fc1(x)

        x = F.relu(x)

        x = self.fc2(x)

        return F.log_softmax(x, dim=1)





model = Model()

optimizer = optim.SGD(model.parameters(), lr=0.01)
for epoch in range(0, 5):

    model.train()

    for batch_idx, (data, target) in enumerate(federated_train_loader):

        # send the model to the client device where the data is present

        model.send(data.location)

        # training the model

        optimizer.zero_grad()

        output = model(data)

        loss = F.nll_loss(output, target)

        loss.backward()

        optimizer.step()

        # get back the improved model

        model.get()

        if batch_idx % 100 == 0:

            # get back the loss

            loss = loss.get()

            print('Epoch: {:2d} [{:5d}/{:5d} ({:3.0f}%)]\tLoss: {:.6f}'.format(

                epoch+1,

                batch_idx * 64,

                len(federated_train_loader) * 64,

                100. * batch_idx / len(federated_train_loader),

                loss.item()))
model.eval()

test_loss = 0

correct = 0

with torch.no_grad():

    for data, target in test_loader:

        output = model(data)

        test_loss += F.nll_loss(

            output, target, reduction='sum').item()

        # get the index of the max log-probability

        pred = output.argmax(1, keepdim=True)

        correct += pred.eq(target.view_as(pred)).sum().item()



test_loss /= len(test_loader.dataset)



print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(

    test_loss,

    correct,

    len(test_loader.dataset),

    100. * correct / len(test_loader.dataset)))
model.fix_precision().share(jake, john, crypto_provider=secure_worker)
