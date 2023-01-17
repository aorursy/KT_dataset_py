# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

print(os.listdir("../input/dataset/dataset"))

# Any results you write to the current directory are saved as output.
import torch

print(torch.__version__)

print(torch.cuda.device_count())

print(torch.cuda.is_available())
import os

import cv2

import torch

import torch.nn as nn

import torch.optim as optim

import torchvision.transforms as transforms



import numpy as np

import pandas as pd



from torch.utils import data

from torch.utils.data import DataLoader

from torch.optim.lr_scheduler import MultiStepLR
dataroot = "../input/dataset/dataset/"

ckptroot = "./"



lr = 1e-4

weight_decay = 1e-5

batch_size = 32

num_workers = 8

test_size = 0.8

shuffle = True



epochs = 80

start_epoch = 0

resume = False
def toDevice(datas, device):

    """Enable cuda."""

    imgs, angles = datas

    return imgs.float().to(device), angles.float().to(device)





def augment(dataroot, imgName, angle):

    """Data augmentation."""

    name = dataroot + 'IMG/' + imgName.split('\\')[-1]

    current_image = cv2.imread(name)



    if current_image is None:

        print(name)



    current_image = current_image[65:-25, :, :]

    if np.random.rand() < 0.5:

        current_image = cv2.flip(current_image, 1)

        angle = angle * -1.0



    return current_image, angle
def load_data(data_dir, test_size):

    """Load training data and train validation split"""

    pass



    # reads CSV file into a single dataframe variable

    data_df = pd.read_csv(os.path.join(data_dir, 'driving_log.csv'),

                          names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])



    # Divide the data into training set and validation set

    train_len = int(test_size * data_df.shape[0])

    valid_len = data_df.shape[0] - train_len

    trainset, valset = data.random_split(

        data_df.values.tolist(), lengths=[train_len, valid_len])



    return trainset, valset



trainset, valset = load_data(dataroot, test_size)
class TripletDataset(data.Dataset):



    def __init__(self, dataroot, samples, transform=None):

        self.samples = samples

        self.dataroot = dataroot

        self.transform = transform



    def __getitem__(self, index):

        batch_samples = self.samples[index]

        steering_angle = float(batch_samples[3])



        center_img, steering_angle_center = augment(self.dataroot, batch_samples[0], steering_angle)

        left_img, steering_angle_left     = augment(self.dataroot, batch_samples[1], steering_angle + 0.4)

        right_img, steering_angle_right   = augment(self.dataroot, batch_samples[2], steering_angle - 0.4)



        center_img = self.transform(center_img)

        left_img   = self.transform(left_img)

        right_img  = self.transform(right_img)



        return (center_img, steering_angle_center), (left_img, steering_angle_left), (right_img, steering_angle_right)



    def __len__(self):

        return len(self.samples)
print("==> Preparing dataset ...")

def data_loader(dataroot, trainset, valset, batch_size, shuffle, num_workers):

    """Self-Driving vehicles simulator dataset Loader.



    Args:

        trainset: training set

        valset: validation set

        batch_size: training set input batch size

        shuffle: whether shuffle during training process

        num_workers: number of workers in DataLoader



    Returns:

        trainloader (torch.utils.data.DataLoader): DataLoader for training set

        testloader (torch.utils.data.DataLoader): DataLoader for validation set

    """

    transformations = transforms.Compose(

        [transforms.Lambda(lambda x: (x / 127.5) - 1.0)])



    # Load training data and validation data

    training_set = TripletDataset(dataroot, trainset, transformations)

    trainloader = DataLoader(training_set,

                             batch_size=batch_size,

                             shuffle=shuffle,

                             num_workers=num_workers)



    validation_set = TripletDataset(dataroot, valset, transformations)

    valloader = DataLoader(validation_set,

                           batch_size=batch_size,

                           shuffle=shuffle,

                           num_workers=num_workers)



    return trainloader, valloader





trainloader, validationloader = data_loader(dataroot,

                                            trainset, valset,

                                            batch_size,

                                            shuffle,

                                            num_workers)
class NetworkNvidia(nn.Module):

    """NVIDIA model used in the paper."""



    def __init__(self):

        """Initialize NVIDIA model.



        NVIDIA model used

            Image normalization to avoid saturation and make gradients work better.

            Convolution: 5x5, filter: 24, strides: 2x2, activation: ELU

            Convolution: 5x5, filter: 36, strides: 2x2, activation: ELU

            Convolution: 5x5, filter: 48, strides: 2x2, activation: ELU

            Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU

            Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU

            Drop out (0.5)

            Fully connected: neurons: 100, activation: ELU

            Fully connected: neurons: 50, activation: ELU

            Fully connected: neurons: 10, activation: ELU

            Fully connected: neurons: 1 (output)



        the convolution layers are meant to handle feature engineering

        the fully connected layer for predicting the steering angle.

        """

        super(NetworkNvidia, self).__init__()

        self.conv_layers = nn.Sequential(

            nn.Conv2d(3, 24, 5, stride=2),

            nn.ELU(),

            nn.Conv2d(24, 36, 5, stride=2),

            nn.ELU(),

            nn.Conv2d(36, 48, 5, stride=2),

            nn.ELU(),

            nn.Conv2d(48, 64, 3),

            nn.ELU(),

            nn.Conv2d(64, 64, 3),

            nn.Dropout(0.5)

        )

        self.linear_layers = nn.Sequential(

            nn.Linear(in_features=64 * 2 * 33, out_features=100),

            nn.ELU(),

            nn.Linear(in_features=100, out_features=50),

            nn.ELU(),

            nn.Linear(in_features=50, out_features=10),

            nn.Linear(in_features=10, out_features=1)

        )



    def forward(self, input):

        """Forward pass."""

        input = input.view(input.size(0), 3, 70, 320)

        output = self.conv_layers(input)

        # print(output.shape)

        output = output.view(output.size(0), -1)

        output = self.linear_layers(output)

        return output





# Define model

print("==> Initialize model ...")

model = NetworkNvidia()

print("==> Initialize model done ...")
# Define optimizer and criterion

optimizer = optim.Adam(model.parameters(),

                       lr=lr,

                       weight_decay=weight_decay)

criterion = nn.MSELoss()
# learning rate scheduler

scheduler = MultiStepLR(optimizer, milestones=[30, 50], gamma=0.1)



# transfer to gpu

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)
if resume:

    print("==> Loading checkpoint ...")

    checkpoint = torch.load("../input/pretrainedmodels/both-nvidia-model-61.h5",

                            map_location=lambda storage, loc: storage)

    start_epoch = checkpoint['epoch']

    model.load_state_dict(checkpoint['state_dict'])

    optimizer.load_state_dict(checkpoint['optimizer'])

    scheduler.load_state_dict(checkpoint['scheduler'])
class Trainer(object):

    """Trainer."""



    def __init__(self,

                 ckptroot,

                 model,

                 device,

                 epochs,

                 criterion,

                 optimizer,

                 scheduler,

                 start_epoch,

                 trainloader,

                 validationloader):

        """Self-Driving car Trainer.



        Args:

            model:

            device:

            epochs:

            criterion:

            optimizer:

            start_epoch:

            trainloader:

            validationloader:



        """

        super(Trainer, self).__init__()



        self.model = model

        self.device = device

        self.epochs = epochs

        self.ckptroot = ckptroot

        self.criterion = criterion

        self.optimizer = optimizer

        self.scheduler = scheduler

        self.start_epoch = start_epoch

        self.trainloader = trainloader

        self.validationloader = validationloader



    def train(self):

        """Training process."""

        self.model.to(self.device)

        for epoch in range(self.start_epoch, self.epochs + self.start_epoch):

            self.scheduler.step()

            

            # Training

            train_loss = 0.0

            self.model.train()



            for local_batch, (centers, lefts, rights) in enumerate(self.trainloader):

                # Transfer to GPU

                centers, lefts, rights = toDevice(centers, self.device), toDevice(

                    lefts, self.device), toDevice(rights, self.device)



                # Model computations

                self.optimizer.zero_grad()

                datas = [centers, lefts, rights]

                for data in datas:

                    imgs, angles = data

                    # print("training image: ", imgs.shape)

                    outputs = self.model(imgs)

                    loss = self.criterion(outputs, angles.unsqueeze(1))

                    loss.backward()

                    self.optimizer.step()



                    train_loss += loss.data.item()



                if local_batch % 100 == 0:



                    print("Training Epoch: {} | Loss: {}".format(epoch, train_loss / (local_batch + 1)))





            # Validation

            self.model.eval()

            valid_loss = 0

            with torch.set_grad_enabled(False):

                for local_batch, (centers, lefts, rights) in enumerate(self.validationloader):

                    # Transfer to GPU

                    centers, lefts, rights = toDevice(centers, self.device), toDevice(

                        lefts, self.device), toDevice(rights, self.device)



                    # Model computations

                    self.optimizer.zero_grad()

                    datas = [centers, lefts, rights]

                    for data in datas:

                        imgs, angles = data

                        outputs = self.model(imgs)

                        loss = self.criterion(outputs, angles.unsqueeze(1))



                        valid_loss += loss.data.item()



                    if local_batch % 100 == 0:

                        print("Validation Loss: {}".format(valid_loss / (local_batch + 1)))





            print()

            # Save model

            if epoch % 5 == 0 or epoch == self.epochs + self.start_epoch - 1:



                state = {

                    'epoch': epoch + 1,

                    'state_dict': self.model.state_dict(),

                    'optimizer': self.optimizer.state_dict(),

                    'scheduler': self.scheduler.state_dict(),

                }



                self.save_checkpoint(state)



    def save_checkpoint(self, state):

        """Save checkpoint."""

        print("==> Save checkpoint ...")

        if not os.path.exists(self.ckptroot):

            os.makedirs(self.ckptroot)



        torch.save(state, self.ckptroot + 'both-nvidia-model-{}.h5'.format(state['epoch']))

print("==> Start training ...")

trainer = Trainer(ckptroot,

                  model,

                  device,

                  epochs,

                  criterion,

                  optimizer,

                  scheduler,

                  start_epoch,

                  trainloader,

                  validationloader)

trainer.train()