%matplotlib inline
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torch import autograd
from torch.autograd import Variable
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
fashion_df = pd.read_csv("../input/fashion-mnist_train.csv")
fashion_df.head()
x = fashion_df.iloc[:,1:].values.astype('uint8').reshape(-1, 28,28)
Image.fromarray(x[0])
class FashionMNIST(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        fashion_df = pd.read_csv("../input/fashion-mnist_train.csv")
        self.labels = fashion_df.label.values
        self.images = fashion_df.iloc[:,1:].values.astype('uint8').reshape(-1, 28, 28)
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx])
        #img = self.images[idx]
        if self.transform:
            img = self.transform(img)
        return img
transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        transforms.Normalize(mean=(0, 0, 0), std=(1, 1, 1))
])
dataset = FashionMNIST(transform=transform)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.z_dim = 100
        self.model = nn.Sequential(
            nn.Linear(self.z_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 784),
            nn.Tanh()
        )
    
    def forward(self, z):
        # z = batch_size*100(z_dim) dim tensor
        out = self.model(z)
        return out
        
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
                nn.Linear(784, 1024),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.3),
                nn.Linear(1024, 512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.3),
                nn.Linear(128, 1),
                nn.Sigmoid()
        )
        
    def forward(self, x):
        # x is batch_size*28*28 dim tensor if sampled from the Fashion-MNIST
        # if its from the generator, a 784 tensor
        x = x.view(x.size(0), 784)
        out = self.model(x)
        return out
    
generator = Generator().cuda()
discriminator = Discriminator().cuda()
criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
g_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4)
def generator_step(batch_size):
    g_optimizer.zero_grad()
    z = Variable(torch.randn(batch_size, 100)).cuda()
    
    # Generator needs to generate images that can fool the discriminator, hence the discriminator should classify
    # its generated image as a real one(label: 1)
    
    # Generator's loss is basically binary cross entropy loss between the discriminator's prediction and 1
    fake_images = generator(z)
    validity = discriminator(fake_images)
    g_loss = criterion(validity, Variable(torch.ones(batch_size)).cuda())
    g_loss.backward()
    g_optimizer.step()
    return g_loss.data
    
def discriminator_step(batch_size, real_images):
    d_optimizer.zero_grad()
    
    # Discriminator's loss is basically comprised of two things:
    # 1) the real images should be classified as 1, hence a BCE between it's prediction for real images and 1
    # 2) fake images should be classified as 0, hence a BCE between it's prediction for fake images and 0
    
    # real loss
    real_validity = discriminator(real_images)
    real_loss = criterion(real_validity, Variable(torch.ones(batch_size)).cuda())
    
    # fake loss
    z = Variable(torch.randn(batch_size, 100)).cuda()
    fake_images = generator(z)
    validity = discriminator(fake_images)
    fake_loss = criterion(validity, Variable(torch.zeros(batch_size)).cuda())
    d_loss = real_loss + fake_loss
    d_loss.backward()
    d_optimizer.step()
    return d_loss.data
NUM_EPOCHS = 50
for epoch in range(NUM_EPOCHS):
    print("Epoch: ", epoch+1)
    for i, images in enumerate(data_loader):
        real_images = Variable(images).cuda()
        batch_size = real_images.size(0)
        generator.train()
        d_loss = discriminator_step(batch_size, real_images)
        g_loss = generator_step(batch_size)
        if i == 100:
            break
    print("g_loss: {} d_loss: {}".format(g_loss, d_loss))
    generator.eval()
    z = Variable(torch.randn(9, 100)).cuda()
    #sample_images = generator(z).data
    sample_images = generator(z)
    sample_images = sample_images.view(sample_images.size(0), 28, 28).unsqueeze(1).data.cpu()
    grid = make_grid(sample_images, nrow=3, normalize=True).permute(1,2,0).numpy()
    plt.imshow(grid)
    plt.show()
    
        
