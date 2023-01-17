import torch

import torch.nn.functional as F

from torch import nn, optim

from torch.utils.data.sampler import SubsetRandomSampler

from torchvision import transforms, models

from torch.optim.lr_scheduler import _LRScheduler

from torch.optim.optimizer import Optimizer



import matplotlib.pyplot as plt

from bisect import bisect_right,bisect_left





import pandas as pd

import numpy as np

import math

import time

import random

import cv2

from PIL import Image





import os

print(os.listdir("../input"))
# Checking GPU is available

train_on_gpu = torch.cuda.is_available()



if not train_on_gpu:

    print('Training on CPU...')

else:

    print('Training on GPU...')
class CyclicCosAnnealingLR(_LRScheduler):

    r"""

    Implements reset on milestones inspired from CosineAnnealingLR pytorch

    

    Set the learning rate of each parameter group using a cosine annealing

    schedule, where :math:`\eta_{max}` is set to the initial lr and

    :math:`T_{cur}` is the number of epochs since the last restart in SGDR:



    .. math::



        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 +

        \cos(\frac{T_{cur}}{T_{max}}\pi))



    When last_epoch > last set milestone, lr is automatically set to \eta_{min}



    It has been proposed in

    `SGDR: Stochastic Gradient Descent with Warm Restarts`_. Note that this only

    implements the cosine annealing part of SGDR, and not the restarts.



    Args:

        optimizer (Optimizer): Wrapped optimizer.

        milestones (list of ints): List of epoch indices. Must be increasing.

        eta_min (float): Minimum learning rate. Default: 0.

        last_epoch (int): The index of last epoch. Default: -1.



    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:

        https://arxiv.org/abs/1608.03983

    """



    def __init__(self, optimizer,milestones, eta_min=0, last_epoch=-1):

        if not list(milestones) == sorted(milestones):

            raise ValueError('Milestones should be a list of'

                             ' increasing integers. Got {}', milestones)

        self.eta_min = eta_min

        self.milestones=milestones

        super(CyclicCosAnnealingLR, self).__init__(optimizer, last_epoch)



    def get_lr(self):

        

        if self.last_epoch >= self.milestones[-1]:

            return [self.eta_min for base_lr in self.base_lrs]



        idx = bisect_right(self.milestones,self.last_epoch)

        

        left_barrier = 0 if idx==0 else self.milestones[idx-1]

        right_barrier = self.milestones[idx]



        width = right_barrier - left_barrier

        curr_pos = self.last_epoch- left_barrier 

    

        return [self.eta_min + (base_lr - self.eta_min) *

               (1 + math.cos(math.pi * curr_pos/ width)) / 2

                for base_lr in self.base_lrs]



 ##warm up

class LearningRateWarmUP(object):

    def __init__(self, optimizer, target_iteration, target_lr, after_scheduler=None):

        self.optimizer = optimizer

        self.target_iteration = target_iteration

        self.target_lr = target_lr

        self.num_iterations = 0

        self.after_scheduler = after_scheduler



    def warmup_learning_rate(self, cur_iteration):

        warmup_lr = self.target_lr*float(cur_iteration)/float(self.target_iteration)

        for param_group in self.optimizer.param_groups:

            param_group['lr'] = warmup_lr



    def step(self, cur_iteration):

        if cur_iteration <= self.target_iteration:

            self.warmup_learning_rate(cur_iteration)

        else:

            self.after_scheduler.step(cur_iteration)
class DualCompose:

    def __init__(self, transforms):

        self.transforms = transforms



    def __call__(self, x):

        for t in self.transforms:

            x= t(x)

        return x





class VerticalFlip:

    def __init__(self, prob=0.5):

        self.prob = prob



    def __call__(self, img):

        if random.random() < self.prob:

            img = cv2.flip(img, 0)

            return img.copy()





class HorizontalFlip:

    def __init__(self, prob=0.6):

        self.prob = prob



    def __call__(self, img):

        if random.random() < self.prob:

            img = cv2.flip(img, 1)

            return img.copy()





class RandomFlip:

    def __init__(self, prob=0.6):

        self.prob = prob



    def __call__(self, img):

        if random.random() < self.prob:

            d = random.randint(-1, 1)

            img = cv2.flip(img, d)

            return img.copy()





class RandomRotate90:

    def __init__(self, prob=0.6):

        self.prob = prob



    def __call__(self, img):

        if random.random() < self.prob:

            factor = random.randint(0, 4)

            img = np.rot90(img, factor)

            return img.copy()





class Rotate:

    def __init__(self, limit=90, prob=0.5):

        self.prob = prob

        self.limit = limit



    def __call__(self, img):

        if random.random() < self.prob:

            angle = random.uniform(-self.limit, self.limit)

            height, width = img.shape[0:2]

            mat = cv2.getRotationMatrix2D((width/2, height/2), angle, 1.0)

            img = cv2.warpAffine(img, mat, (height, width),

                                 flags=cv2.INTER_LINEAR,

                                 borderMode=cv2.BORDER_REFLECT_101)

            return img.copy()





class Shift:

    def __init__(self, limit=50, prob=0.5):

        self.limit = limit

        self.prob = prob



    def __call__(self, img):

        if random.random() < self.prob:

            limit = self.limit

            dx = round(random.uniform(-limit, limit))

            dy = round(random.uniform(-limit, limit))



            height, width, channel = img.shape

            y1 = limit + 1 + dy

            y2 = y1 + height

            x1 = limit + 1 + dx

            x2 = x1 + width



            img1 = cv2.copyMakeBorder(img, limit+1, limit + 1, limit + 1, limit +1,

                                      borderType=cv2.BORDER_REFLECT_101)

            img = img1[y1:y2, x1:x2, :]

            return img.copy()
# Dataset responsible for manipulating data for training as well as training tests.

class DatasetMNIST(torch.utils.data.Dataset):

    def __init__(self, data, transform=None):

        self.data = data

        self.transform = transform

#         self.JointTransform = DualCompose([

#                                 RandomFlip(),

#                                 RandomRotate90(),

#                                 Rotate(),

#                                    ])

        

    def __len__(self):

        return len(self.data)

    

    def __getitem__(self, index):

        item = self.data.iloc[index]

                

        image = item[1:].values.astype(np.uint8).reshape((28, 28))

        label = item[0]

#         print(image.shape)

#         image = np.array(self.JointTransform(image))



#         image = Image.fromarray(image)

        

        if self.transform is not None:

            image = self.transform(image)

            

        return image, label
BATCH_SIZE = 100

VALID_SIZE = 0.15 # percentage of data for validation



transform_train = transforms.Compose([

    transforms.ToPILImage(),

#     transforms.RandomRotation(0, 0.5),

    transforms.ToTensor(),

    transforms.Normalize(mean=(0.5,), std=(0.5,))

])



transform_valid = transforms.Compose([

    transforms.ToPILImage(),

    transforms.ToTensor(),

    transforms.Normalize(mean=(0.5,), std=(0.5,))

])



# Importing data that will be used for training and validation

dataset = pd.read_csv('../input/train.csv')

test_dataset = pd.read_csv('../input/test.csv')



# Creating datasets for training and validation

train_data = DatasetMNIST(dataset, transform=transform_train)

valid_data = DatasetMNIST(dataset, transform=transform_valid)

test_data = DatasetMNIST(test_dataset, transform=transform_valid)



# Shuffling data and choosing data that will be used for training and validation

num_train = len(train_data)

indices = list(range(num_train))

np.random.shuffle(indices)

split = int(np.floor(VALID_SIZE * num_train))

train_idx, valid_idx = indices[split:], indices[:split]



train_sampler = SubsetRandomSampler(train_idx)

valid_sampler = SubsetRandomSampler(valid_idx)



train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, sampler=train_sampler)

valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=BATCH_SIZE, sampler=valid_sampler)

test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE)



print(f"Length train: {len(train_idx)}")

print(f"Length valid: {len(valid_idx)}")

print(f"Length test: {len(test_loader.dataset)}")
# Viewing data examples used for training

fig, axis = plt.subplots(3, 10, figsize=(15, 10))

images, labels = next(iter(train_loader))



for i, ax in enumerate(axis.flat):

    with torch.no_grad():

        image, label = images[i], labels[i]



        ax.imshow(image.view(28, 28), cmap='binary') # add image

        ax.set(title = f"{label}") # add label
# Viewing data examples used for validation

fig, axis = plt.subplots(3, 10, figsize=(15, 10))

images, labels = next(iter(valid_loader))



for i, ax in enumerate(axis.flat):

    with torch.no_grad():

        image, label = images[i], labels[i]



        ax.imshow(image.view(28, 28), cmap='binary') # add image

        ax.set(title = f"{label}") # add label
from torch.optim.optimizer import Optimizer



class RAdam(Optimizer):



    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        self.buffer = [[None, None, None] for ind in range(10)]

        super(RAdam, self).__init__(params, defaults)



    def __setstate__(self, state):

        super(RAdam, self).__setstate__(state)



    def step(self, closure=None):



        loss = None

        if closure is not None:

            loss = closure()



        for group in self.param_groups:



            for p in group['params']:

                if p.grad is None:

                    continue

                grad = p.grad.data.float()

                if grad.is_sparse:

                    raise RuntimeError('RAdam does not support sparse gradients')



                p_data_fp32 = p.data.float()



                state = self.state[p]



                if len(state) == 0:

                    state['step'] = 0

                    state['exp_avg'] = torch.zeros_like(p_data_fp32)

                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)

                else:

                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)

                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)



                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                beta1, beta2 = group['betas']



                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                exp_avg.mul_(beta1).add_(1 - beta1, grad)



                state['step'] += 1

                buffered = self.buffer[int(state['step'] % 10)]

                if state['step'] == buffered[0]:

                    N_sma, step_size = buffered[1], buffered[2]

                else:

                    buffered[0] = state['step']

                    beta2_t = beta2 ** state['step']

                    N_sma_max = 2 / (1 - beta2) - 1

                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)

                    buffered[1] = N_sma



                    # more conservative since it's an approximated value

                    if N_sma >= 5:

                        step_size = group['lr'] * math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])

                    else:

                        step_size = group['lr'] / (1 - beta1 ** state['step'])

                    buffered[2] = step_size



                if group['weight_decay'] != 0:

                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)



                # more conservative since it's an approximated value

                if N_sma >= 5:            

                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                    p_data_fp32.addcdiv_(-step_size, exp_avg, denom)

                else:

                    p_data_fp32.add_(-step_size, exp_avg)



                p.data.copy_(p_data_fp32)



        return loss
from torchvision.models import ResNet





def conv3x3(in_planes, out_planes, stride=1):

    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)





class SELayer(nn.Module):

    def __init__(self, channel, reduction=16):

        super(SELayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(

            nn.Linear(channel, channel // reduction, bias=False),

            nn.ReLU(inplace=True),

            nn.Linear(channel // reduction, channel, bias=False),

            nn.Sigmoid()

        )



    def forward(self, x):

        b, c, _, _ = x.size()

        y = self.avg_pool(x).view(b, c)

        y = self.fc(y).view(b, c, 1, 1)

        return x * y.expand_as(x)

    



class SEBasicBlock(nn.Module):

    expansion = 1



    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,

                 base_width=64, dilation=1, norm_layer=None,

                 *, reduction=16):

        super(SEBasicBlock, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride)

        self.bn1 = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes, 1)

        self.bn2 = nn.BatchNorm2d(planes)

        self.se = SELayer(planes, reduction)

        self.downsample = downsample

        self.stride = stride



    def forward(self, x):

        residual = x

        out = self.conv1(x)

        out = self.bn1(out)

        out = self.relu(out)



        out = self.conv2(out)

        out = self.bn2(out)

        out = self.se(out)



        if self.downsample is not None:

            residual = self.downsample(x)



        out += residual

        out = self.relu(out)



        return out

    



def se_resnet18(num_classes=1000):

    """Constructs a ResNet-18 model.



    Args:

        pretrained (bool): If True, returns a model pre-trained on ImageNet

    """

    model = ResNet(SEBasicBlock, [2, 2, 2, 2], num_classes=num_classes)

    model.avgpool = nn.AdaptiveAvgPool2d(1)

    return model

# num_classes = 10

# model = models.resnet18(num_classes=num_classes)

# model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,

#                                bias=False)

# print(model)

model = se_resnet18()

model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,

                       bias=False)

model.fc = nn.Linear(512, 10)

print(model)

if train_on_gpu:

    model.cuda()
class LabelSmoothingCrossEntropy(nn.Module):

    def __init__(self):

        super(LabelSmoothingCrossEntropy, self).__init__()

        

    def forward(self, x, target, smoothing=0.1):

        confidence = 1. - smoothing

        logprobs = F.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))

        nll_loss = nll_loss.squeeze(1)

        smooth_loss = -logprobs.mean(dim=-1)

        loss = confidence * nll_loss + smoothing * smooth_loss

        return loss.mean()
LEARNING_RATE = 0.001680



criterion = LabelSmoothingCrossEntropy()

optimizer = RAdam(params=model.parameters(), lr=0.003, weight_decay=0.0001)

milestones = [10 + x * 40 for x in range(20)]

print(milestones)

scheduler_c = CyclicCosAnnealingLR(optimizer,milestones=milestones,eta_min=5e-5)

scheduler = LearningRateWarmUP(optimizer=optimizer, target_iteration=10, target_lr=0.003,

                                   after_scheduler=scheduler_c)
epochs = 50

valid_accuracy_max = -100

train_losses, valid_losses = [], []

history_accuracy = []



for e in range(1, epochs+1):

    running_loss = 0

    scheduler.step(e)



    for images, labels in train_loader:

#         print(f"labels:{labels}")

        if train_on_gpu:

            images, labels = images.cuda(), labels.cuda()

        # Clear the gradients, do this because gradients are accumulated.

        optimizer.zero_grad()

        

        # Forward pass, get our log-probabilities.

        ps = model(images)



        # Calculate the loss with the logps and the labels.

        loss = criterion(ps, labels)

        

        # Turning loss back.

        loss.backward()

        

        # Take an update step and few the new weights.

        optimizer.step()

        

        running_loss += loss.item()

    else:

        valid_loss = 0

        accuracy = 0

        

        # Turn off gradients for validation, saves memory and computations.

        with torch.no_grad():

            model.eval() # change the network to evaluation mode

            for images, labels in valid_loader:

                if train_on_gpu:

                    images, labels = images.cuda(), labels.cuda()

                # Forward pass, get our log-probabilities.

                #log_ps = model(images)

                ps = model(images)

                

                # Calculating probabilities for each class.

                #ps = torch.exp(log_ps)

                

                # Capturing the class more likely.

                _, top_class = ps.topk(1, dim=1)

                

                # Verifying the prediction with the labels provided.

                equals = top_class == labels.view(*top_class.shape)

                

                valid_loss += criterion(ps, labels)

                accuracy += torch.mean(equals.type(torch.FloatTensor))

                

        model.train() # change the network to training mode

        

        train_losses.append(running_loss/len(train_loader))

        valid_losses.append(valid_loss/len(valid_loader))

        history_accuracy.append(accuracy/len(valid_loader))

        

        network_learned = accuracy/len(valid_loader) > valid_accuracy_max

        # update learning rate according to accuracy

#         scheduler.step(accuracy)  



        if e == 1 or e % 5 == 0 or network_learned:

            start = time.strftime("%H:%M:%S")

            print(f"Epoch: {e}/{epochs}..  | ⏰: {start}",

                  f"Training Loss: {running_loss/len(train_loader):.6f}.. ",

                  f"Validation Loss: {valid_loss/len(valid_loader):.6f}.. ",

                  f"Test Accuracy: {accuracy/len(valid_loader):.6f}")

        

        if network_learned:

            valid_accuracy_max = accuracy/len(valid_loader)

            torch.save(model.state_dict(), 'model_mtl_mnist.pt')

            print('Detected network improvement, saving current model')
# Dataset responsible for manipulating data for training as well as training tests.

class TestDatasetMNIST(torch.utils.data.Dataset):

    def __init__(self, data, transform=None):

        self.data = data

        self.transform = transform

#         self.JointTransform = DualCompose([

#                                 RandomFlip(),

#                                 RandomRotate90(),

#                                 Rotate(),

#                                    ])

        

    def __len__(self):

        return len(self.data)

    

    def __getitem__(self, index):

        item = self.data.iloc[index]

                

        image = item[:].values.astype(np.uint8).reshape((28, 28))

#         print(image.shape)

#         image = np.array(self.JointTransform(image))



#         image = Image.fromarray(image)

        

        if self.transform is not None:

            image = self.transform(image)

            

        return image



test_data = TestDatasetMNIST(test_dataset, transform=transform_valid)

test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE)



print(f"Length test: {len(test_loader.dataset)}")
for img in test_loader:

    print(img.size())
T1 = 10

T2 = 70

af = 3



def alpha_weight(step):

    if step < T1:

        return 0.0

    elif step > T2:

        return af

    else:

         return ((step-T1) / (T2-T1))*af
# Concept from : https://github.com/peimengsui/semi_supervised_mnist



from tqdm import tqdm_notebook





def semisup_train(model, train_loader, unlabeled_loader, test_loader):

    valid_accuracy_max = -100

    train_losses, valid_losses = [], []

    history_accuracy = []



    acc_scores = []

    unlabel = []

    pseudo_label = []



    alpha_log = []

    test_acc_log = []

    test_loss_log = []

    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.003, weight_decay=0.0001)

    milestones = [x * 50 for x in range(5)]

    print(milestones)

    scheduler = CyclicCosAnnealingLR(optimizer,milestones=milestones,eta_min=5e-5)

    EPOCHS = 150

    

    # Instead of using current epoch we use a "step" variable to calculate alpha_weight

    # This helps the model converge faster

    step = 10 

    

    model.train()

    for epoch in tqdm_notebook(range(EPOCHS)):

        scheduler.step(epoch)

        for batch_idx, x_unlabeled in enumerate(unlabeled_loader):

            

            

            # Forward Pass to get the pseudo labels

            x_unlabeled = x_unlabeled.cuda()

            model.eval()

            output_unlabeled = model(x_unlabeled)

            _, pseudo_labeled = torch.max(output_unlabeled, 1)

            model.train()

            

            

            # Now calculate the unlabeled loss using the pseudo label

            output = model(x_unlabeled)

            unlabeled_loss = alpha_weight(step) * criterion(output, pseudo_labeled)   

            

            # Backpropogate

            optimizer.zero_grad()

            unlabeled_loss.backward()

            optimizer.step()

            

                

        # Normal training procedure

        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):

            X_batch = X_batch.cuda()

            y_batch = y_batch.cuda()

            output = model(X_batch)

            labeled_loss = criterion(output, y_batch)



            optimizer.zero_grad()

            labeled_loss.backward()

            optimizer.step()



        # Now we increment step by 1

        step += 1





        # Turn off gradients for validation, saves memory and computations.

        valid_loss = 0

        accuracy = 0

        with torch.no_grad():

            model.eval() # change the network to evaluation mode

            for images, labels in test_loader:

                if train_on_gpu:

                    images, labels = images.cuda(), labels.cuda()

                # Forward pass, get our log-probabilities.

                #log_ps = model(images)

                ps = model(images)

                

                # Calculating probabilities for each class.

                #ps = torch.exp(log_ps)

                

                # Capturing the class more likely.

                _, top_class = ps.topk(1, dim=1)

                

                # Verifying the prediction with the labels provided.

                equals = top_class == labels.view(*top_class.shape)

                

                valid_loss += criterion(ps, labels)

                accuracy += torch.mean(equals.type(torch.FloatTensor))

                

        model.train() # change the network to training mode

        

#         train_losses.append(running_loss/len(train_loader))

#         valid_losses.append(valid_loss/len(valid_loader))

#         history_accuracy.append(accuracy/len(valid_loader))

        

        network_learned = accuracy/len(valid_loader) > valid_accuracy_max

        # update learning rate according to accuracy

#         scheduler.step(accuracy)  



        if e == 1 or e % 5 == 0 or network_learned:

            start = time.strftime("%H:%M:%S")

            print(f"Epoch: {epoch}/{EPOCHS}..  | ⏰: {start}",

                  f"Test Accuracy: {accuracy/len(valid_loader):.8f}")

        

        if network_learned:

            valid_accuracy_max = accuracy/len(valid_loader)

            torch.save(model.state_dict(), 'model_mtl_mnist.pt')

            print('Detected network improvement, saving current model')

 
model.load_state_dict(torch.load('model_mtl_mnist.pt'))

semisup_train(model, train_loader, test_loader, valid_loader)
# Viewing training information

%matplotlib inline

%config InlineBackend.figure_format = 'retina'



import matplotlib.pyplot as plt



plt.plot(train_losses, label='Training Loss')

plt.plot(valid_losses, label='Validation Loss')

plt.legend(frameon=False)
plt.plot(history_accuracy, label='Validation Accuracy')

plt.legend(frameon=False)
# Importing trained Network with better loss of validation

model.load_state_dict(torch.load('model_mtl_mnist.pt'))



print(model)
# specify the image classes

classes = ['0', '1', '2', '3', '4',

           '5', '6', '7', '8', '9']



# track test loss

test_loss = 0.0

class_correct = list(0. for i in range(10))

class_total = list(0. for i in range(10))



model.eval()

# iterate over test data

for data, target in valid_loader:

    # move tensors to GPU if CUDA is available

    if train_on_gpu:

        data, target = data.cuda(), target.cuda()

    # forward pass: compute predicted outputs by passing inputs to the model

    output = model(data)

    # calculate the batch loss

    loss = criterion(output, target)

    # update test loss 

    test_loss += loss.item()*data.size(0)

    # convert output probabilities to predicted class

    _, pred = torch.max(output, 1)    

    # compare predictions to true label

    correct_tensor = pred.eq(target.data.view_as(pred))

    correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())

    # calculate test accuracy for each object class

    for i in range(BATCH_SIZE):

        label = target.data[i]

        class_correct[label] += correct[i].item()

        class_total[label] += 1



# average test loss

test_loss = test_loss/len(valid_loader.dataset)

print('Test Loss: {:.6f}\n'.format(test_loss))



for i in range(10):

    if class_total[i] > 0:

        print('Test Accuracy of %5s: %0.4f%% (%2d/%2d)' % (

            classes[i], class_correct[i] / class_total[i],

            np.sum(class_correct[i]), np.sum(class_total[i])))

    else:

        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))



print('\nTest Accuracy (Overall): %2.2f%% (%2d/%2d)' % (

    100. * np.sum(class_correct) / np.sum(class_total),

    np.sum(class_correct), np.sum(class_total)))
class DatasetSubmissionMNIST(torch.utils.data.Dataset):

    def __init__(self, file_path, transform=None):

        self.data = pd.read_csv(file_path)

        self.transform = transform

        

    def __len__(self):

        return len(self.data)

    

    def __getitem__(self, index):

        image = self.data.iloc[index].values.astype(np.uint8).reshape((28, 28, 1))



        

        if self.transform is not None:

            image = self.transform(image)

            

        return image
transform = transforms.Compose([

    transforms.ToPILImage(),

    transforms.ToTensor(),

    transforms.Normalize(mean=(0.5,), std=(0.5,))

])



submissionset = DatasetSubmissionMNIST('../input/test.csv', transform=transform)

submissionloader = torch.utils.data.DataLoader(submissionset, batch_size=BATCH_SIZE, shuffle=False)
submission = [['ImageId', 'Label']]



with torch.no_grad():

    model.eval()

    image_id = 1



    for images in submissionloader:

        if train_on_gpu:

            images = images.cuda()

        log_ps = model(images)

        ps = torch.exp(log_ps)

        top_p, top_class = ps.topk(1, dim=1)

        

        for prediction in top_class:

            submission.append([image_id, prediction.item()])

            image_id += 1

            

print(len(submission) - 1)
import csv



with open('submission.csv', 'w') as submissionFile:

    writer = csv.writer(submissionFile)

    writer.writerows(submission)

    

print('Submission Complete!')