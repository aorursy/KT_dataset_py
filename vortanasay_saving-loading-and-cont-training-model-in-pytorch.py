%pwd
%ls
# uncomment if you want to create directory checkpoint, best_model

%mkdir checkpoint best_model
# uncomment if you want to remove folder best_model and checkpoint

# %rm -r best_model/ checkpoint/
%matplotlib inline

%config InlineBackend.figure_format = 'retina'



import matplotlib.pyplot as plt



import torch

from torch import nn

from torch import optim

import torch.nn.functional as F

from torchvision import datasets, transforms

import numpy as np

# check if CUDA is available

use_cuda = torch.cuda.is_available()
# Define your network ( Simple Example )

class FashionClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        input_size = 784

        self.fc1 = nn.Linear(input_size, 512)

        self.fc2 = nn.Linear(512, 256)

        self.fc3 = nn.Linear(256, 128)

        self.fc4 = nn.Linear(128, 64)

        self.fc5 = nn.Linear(64,10)

        self.dropout = nn.Dropout(p=0.2)

        

    def forward(self, x):

        x = x.view(x.shape[0], -1)

        x = self.dropout(F.relu(self.fc1(x)))

        x = self.dropout(F.relu(self.fc2(x)))

        x = self.dropout(F.relu(self.fc3(x)))

        x = self.dropout(F.relu(self.fc4(x)))

        x = F.log_softmax(self.fc5(x), dim=1)

        return x
# Define a transform to normalize the data

transform = transforms.Compose([transforms.ToTensor(),

                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Download and load the training data

trainset = datasets.FashionMNIST('F_MNIST_data/', download=True, train=True, transform=transform)



# Download and load the test data

testset = datasets.FashionMNIST('F_MNIST_data/', download=True, train=False, transform=transform)



loaders = {

    'train' : torch.utils.data.DataLoader(trainset,batch_size = 64,shuffle=True),

    'test'  : torch.utils.data.DataLoader(testset,batch_size = 64,shuffle=True),

}
# Create the network, define the criterion and optimizer

model = FashionClassifier()

# move model to GPU if CUDA is available

if use_cuda:

    model = model.cuda()



print(model)
#define loss function and optimizer

criterion = nn.NLLLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)
import torch

import shutil

def save_ckp(state, is_best, checkpoint_path, best_model_path):

    f_path = checkpoint_path

    torch.save(state, f_path)

    if is_best:

        best_fpath = best_model_path

        shutil.copyfile(f_path, best_fpath)
def train(start_epochs, n_epochs, valid_loss_min_input, loaders, model, optimizer, criterion, use_cuda, checkpoint_path, best_model_path):

    """

    Keyword arguments:

    start_epochs -- the real part (default 0.0)

    n_epochs -- the imaginary part (default 0.0)

    valid_loss_min_input

    loaders

    model

    optimizer

    criterion

    use_cuda

    checkpoint_path

    best_model_path

    

    returns trained model

    """

    # initialize tracker for minimum validation loss

    valid_loss_min = valid_loss_min_input 

    

    for epoch in range(start_epochs, n_epochs+1):

        # initialize variables to monitor training and validation loss

        train_loss = 0.0

        valid_loss = 0.0

        

        ###################

        # train the model #

        ###################

        model.train()

        for batch_idx, (data, target) in enumerate(loaders['train']):

            # move to GPU

            if use_cuda:

                data, target = data.cuda(), target.cuda()

            ## find the loss and update the model parameters accordingly

            # clear the gradients of all optimized variables

            optimizer.zero_grad()

            # forward pass: compute predicted outputs by passing inputs to the model

            output = model(data)

            # calculate the batch loss

            loss = criterion(output, target)

            # backward pass: compute gradient of the loss with respect to model parameters

            loss.backward()

            # perform a single optimization step (parameter update)

            optimizer.step()

            ## record the average training loss, using something like

            ## train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))

            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))

        

        ######################    

        # validate the model #

        ######################

        model.eval()

        for batch_idx, (data, target) in enumerate(loaders['test']):

            # move to GPU

            if use_cuda:

                data, target = data.cuda(), target.cuda()

            ## update the average validation loss

            # forward pass: compute predicted outputs by passing inputs to the model

            output = model(data)

            # calculate the batch loss

            loss = criterion(output, target)

            # update average validation loss 

            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))

            

        # calculate average losses

        train_loss = train_loss/len(loaders['train'].dataset)

        valid_loss = valid_loss/len(loaders['test'].dataset)



        # print training/validation statistics 

        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(

            epoch, 

            train_loss,

            valid_loss

            ))

        

        checkpoint = {

            'epoch': epoch + 1,

            'valid_loss_min': valid_loss,

            'state_dict': model.state_dict(),

            'optimizer': optimizer.state_dict(),

        }

        

        save_ckp(checkpoint, False, checkpoint_path, best_model_path)

        ## TODO: save the model if validation loss has decreased

        if valid_loss <= valid_loss_min:

            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,valid_loss))

            save_ckp(checkpoint, True, checkpoint_path, best_model_path)

            valid_loss_min = valid_loss

            

    # return trained model

    return model
trained_model = train(1, 3, np.Inf, loaders, model, optimizer, criterion, use_cuda, "./checkpoint/current_checkpoint.pt", "./best_model/best_model.pt")
%ls ./best_model/
%ls ./checkpoint/
#torch.save(checkpoint, 'checkpoint.pth')
"""

checkpoint = {

            'epoch': epoch + 1,

            'valid_loss_min': valid_loss,

            'state_dict': model.state_dict(),

            'optimizer': optimizer.state_dict(),

        }

"""
# in train method, above we use this to save checkpoint file

# save_ckp(checkpoint, False, checkpoint_path, best_model_path)
# in train method, above we use this to save best_model file

# save_ckp(checkpoint, False, checkpoint_path, best_model_path)
def load_ckp(checkpoint_fpath, model, optimizer):

    checkpoint = torch.load(checkpoint_fpath)

    model.load_state_dict(checkpoint['state_dict'])

    optimizer.load_state_dict(checkpoint['optimizer'])

    valid_loss_min = checkpoint['valid_loss_min']

    return model, optimizer, checkpoint['epoch'], valid_loss_min.item()
%pwd

%ls
model = FashionClassifier()

# move model to GPU if CUDA is available

if use_cuda:

    model = model.cuda()



print(model)
optimizer = optim.Adam(model.parameters(), lr=0.001)

ckp_path = "./checkpoint/current_checkpoint.pt"

model, optimizer, start_epoch, valid_loss_min = load_ckp(ckp_path, model, optimizer)
print("model = ", model)

print("optimizer = ", optimizer)

print("start_epoch = ", start_epoch)

print("valid_loss_min = ", valid_loss_min)

print("valid_loss_min = {:.6f}".format(valid_loss_min))
trained_model = train(start_epoch, 6, valid_loss_min, loaders, model, optimizer, criterion, use_cuda, "./checkpoint/current_checkpoint.pt", "./best_model/best_model.pt")
trained_model.eval()
test_acc = 0.0

for samples, labels in loaders['test']:

    with torch.no_grad():

        samples, labels = samples.cuda(), labels.cuda()

        output = trained_model(samples)

        # calculate accuracy

        pred = torch.argmax(output, dim=1)

        correct = pred.eq(labels)

        test_acc += torch.mean(correct.float())



print('Accuracy of the network on {} test images: {}%'.format(len(testset), round(test_acc.item()*100.0/len(loaders['test']), 2)))