import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch.autograd.variable import Variable
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score
print(sys.version)
device='cuda'

train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')
image_size = 64
dataset = torchvision.datasets.ImageFolder(root="../input/celeba-dataset/img_align_celeba",
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=128,
                                         shuffle=True, num_workers=2)

# Decide which device we want to run on
device = torch.device("cuda:0")
# Plot some training images
real_batch = next(iter(dataloader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(torchvision.utils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
class GeneratorNet(torch.nn.Module):
  def __init__(self):
    super(GeneratorNet, self).__init__()
    self.main = nn.Sequential(
        nn.ConvTranspose2d(100, 1024, kernel_size = 4, stride = 1, padding = 0, bias = False),
        #nn.BatchNorm2d(1024),
        nn.ReLU(inplace = True),

        nn.ConvTranspose2d(1024, 512, kernel_size = 4, stride = 2, padding = 1, bias =False),
        #nn.BatchNorm2d(512),
        nn.ReLU(inplace = True),

        nn.ConvTranspose2d(512, 256, kernel_size = 4, stride = 2, padding = 1, bias=False),
        #nn.BatchNorm2d(256),
        nn.ReLU(inplace = True),

        nn.ConvTranspose2d(256, 128, kernel_size = 4, stride = 2, padding = 1, bias=False),
        #nn.BatchNorm2d(128),
        nn.ReLU(inplace = True),

        nn.ConvTranspose2d(128, 3, kernel_size = 4, stride = 2, padding = 1, bias=False),
        nn.Tanh()
    )
    

  def forward(self, x):
    #print(x)
    x = self.main(x)
    #print(x.shape)
    return x

generator = GeneratorNet()
generator.float()
generator.to(device)

generator.apply(weights_init)

print(generator)
class DiscriminatorNet(torch.nn.Module):
  def __init__(self):
    super(DiscriminatorNet, self).__init__()
    self.main = nn.Sequential(
      nn.Conv2d(3, 128, kernel_size = 5, stride = 2, padding = 2, bias = False),
      nn.LeakyReLU(0.2, inplace=True),

      nn.Conv2d(128, 256, kernel_size = 5, stride = 2, padding = 2, bias = False),
      nn.BatchNorm2d(256),
      nn.LeakyReLU(0.2, inplace=True),

      nn.Conv2d(256, 512, kernel_size = 5, stride = 2, padding =2, bias = False),
      nn.BatchNorm2d(512),
      nn.LeakyReLU(0.2, inplace=True),

      nn.Conv2d(512, 1024, kernel_size = 5, stride = 2, padding = 2, bias = False),
      nn.BatchNorm2d(1024),
      nn.LeakyReLU(0.2, inplace=True),

      nn.Conv2d(1024, 1, kernel_size = 4, stride = 1, padding = 0, bias = False)
    )
    
  def forward(self, x):
    x = self.main(x)
    return x

discriminator = DiscriminatorNet()
discriminator.float()
discriminator.to(device)

discriminator.apply(weights_init)

print(discriminator)
criterion = nn.BCEWithLogitsLoss()
optimizerG = optim.Adam(generator.parameters(), lr = 0.0002, betas = (0.5, 0.999))
optimizerD = optim.Adam(discriminator.parameters(), lr = 0.0002, betas = (0.5, 0.999))
def noise(size):
  n = Variable(torch.randn(size, 100, 1, 1))
  return n.to(device)

samples = 16
fixed_noise = noise(samples)
lossesD = []
lossesG = []

num_epochs = 100
for epoch in range(num_epochs):
  discriminator.train()
  generator.train()
  lossD = 0
  lossG = 0
  prob_real = 0
  prob_fake = 0 
  for num_iter, (real_batch, _) in enumerate(dataloader):

    x_real = Variable(real_batch).to(device)
    optimizerD.zero_grad()
    pred_real = discriminator(x_real)
    pred_real.to(device)
    loss_real = criterion(pred_real.view(-1,1), (torch.ones(x_real.size(0),1)).to(device))
    loss_real.backward()
    z = noise(x_real.size(0)).to(device)
    x_fake = generator(z).to(device)
    x_fake.detach()
    pred_fake = discriminator(x_fake)
    pred_fake.to(device)
    loss_fake = criterion(pred_fake.view(-1,1), (torch.zeros(x_real.size(0),1)).to(device))
    loss_fake.backward()
    optimizerD.step()
    lossD = lossD + loss_real + loss_fake
    #prob_real = prob_real + np.mean(pred_real.detach().cpu().numpy())
    #prob_fake = prob_fake + np.mean(pred_fake.detach().cpu().numpy())
    #binary_pred_real = np.zeros((x_real.size(0),1))
    #binary_pred_fake = np.zeros((x_real.size(0),1))
    #binary_pred_real[pred_real.detach().cpu().numpy()>0.5]=1
    #binary_pred_fake[pred_fake.detach().cpu().numpy()>0.5]=1
    #accuracy_real = accuracy_real + accuracy_score(np.ones((x_real.size(0),1)),binary_pred_real)
    #accuracy_fake = accuracy_fake + accuracy_score(np.zeros((x_real.size(0),1)),binary_pred_fake)

    #zg = random_noise(x_real.size(0)).to(device)
    fake_x = generator(z).to(device)
    optimizerG.zero_grad()
    fake_pred = discriminator(fake_x)
    loss_gen = criterion(fake_pred.view(-1,1), (torch.ones(x_real.size(0),1)).to(device))
    loss_gen.backward()
    optimizerG.step()
    lossG = lossG + loss_gen

  lossesD.append(lossD/len(dataloader))
  lossesG.append(lossG/len(dataloader))
  print("Epoch No. = "+ str(epoch+1))
  print("Discriminator Loss = "+ str(lossesD[epoch].item()), "Generator Loss = "+ str(lossesG[epoch].item()))
  #print("Discriminator Confidence on Real Data = "+ str(prob_real/len(train_dl)), "Discriminator Confidence on Fake Data = "+str(prob_fake/len(train_dl)))

  with torch.no_grad():
    generated_images = generator(fixed_noise.detach())
    for i in range(16):
      plt.subplot(4, 4, 1 + i)
      plt.axis('off')
      plt.imshow(np.transpose(generated_images.cpu().numpy()[i],(1,2,0)))
    plt.show()  

plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(lossesG,label="G")
plt.plot(lossesD,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()    
test_noise = noise(100)
with torch.no_grad():
  test_images = generator(test_noise.detach())
  for i in range(100):
	  plt.subplot(10, 10, 1 + i)
	  plt.axis('off')
	  plt.imshow(np.transpose(test_images.cpu().numpy()[i],(1,2,0)))
  plt.show()
test_noise = noise(4)
with torch.no_grad():
  test_images = generator(test_noise.detach())
  for i in range(4):
	  plt.subplot(2, 2, 1 + i)
	  plt.axis('off')
	  plt.imshow(np.transpose(test_images.cpu().numpy()[i],(1,2,0)))
  plt.show()
import os
torch.save(generator.state_dict(),'g_epoch-{}.pth'.format(15))
torch.save(discriminator.state_dict(), 'd_epoch-{}.pth'.format(15))
lossesD = []
lossesG = []

num_epochs = 5
for epoch in range(num_epochs):
  discriminator.train()
  generator.train()
  lossD = 0
  lossG = 0
  prob_real = 0
  prob_fake = 0 
  for num_iter, (real_batch, _) in enumerate(dataloader):

    x_real = Variable(real_batch).to(device)
    optimizerD.zero_grad()
    pred_real = discriminator(x_real)
    pred_real.to(device)
    loss_real = criterion(pred_real.view(-1,1), (torch.ones(x_real.size(0),1)).to(device))
    loss_real.backward()
    z = noise(x_real.size(0)).to(device)
    x_fake = generator(z).to(device)
    x_fake.detach()
    pred_fake = discriminator(x_fake)
    pred_fake.to(device)
    loss_fake = criterion(pred_fake.view(-1,1), (torch.zeros(x_real.size(0),1)).to(device))
    loss_fake.backward()
    optimizerD.step()
    lossD = lossD + loss_real + loss_fake
    #prob_real = prob_real + np.mean(pred_real.detach().cpu().numpy())
    #prob_fake = prob_fake + np.mean(pred_fake.detach().cpu().numpy())
    #binary_pred_real = np.zeros((x_real.size(0),1))
    #binary_pred_fake = np.zeros((x_real.size(0),1))
    #binary_pred_real[pred_real.detach().cpu().numpy()>0.5]=1
    #binary_pred_fake[pred_fake.detach().cpu().numpy()>0.5]=1
    #accuracy_real = accuracy_real + accuracy_score(np.ones((x_real.size(0),1)),binary_pred_real)
    #accuracy_fake = accuracy_fake + accuracy_score(np.zeros((x_real.size(0),1)),binary_pred_fake)

    #zg = random_noise(x_real.size(0)).to(device)
    fake_x = generator(z).to(device)
    optimizerG.zero_grad()
    fake_pred = discriminator(fake_x)
    loss_gen = criterion(fake_pred.view(-1,1), (torch.ones(x_real.size(0),1)).to(device))
    loss_gen.backward()
    optimizerG.step()
    lossG = lossG + loss_gen

  lossesD.append(lossD/len(dataloader))
  lossesG.append(lossG/len(dataloader))
  print("Epoch No. = "+ str(epoch+1+15))
  print("Discriminator Loss = "+ str(lossesD[epoch].item()), "Generator Loss = "+ str(lossesG[epoch].item()))
  #print("Discriminator Confidence on Real Data = "+ str(prob_real/len(train_dl)), "Discriminator Confidence on Fake Data = "+str(prob_fake/len(train_dl)))
  torch.save(generator.state_dict(),'g_epoch-{}.pth'.format(epoch+1+15))
  torch.save(discriminator.state_dict(), 'd_epoch-{}.pth'.format(epoch+1+15))

  with torch.no_grad():
    generated_images = generator(fixed_noise.detach())
    for i in range(16):
      plt.subplot(4, 4, 1 + i)
      plt.axis('off')
      plt.imshow(np.transpose(generated_images.cpu().numpy()[i],(1,2,0)))
    plt.show()  

plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(lossesG,label="G")
plt.plot(lossesD,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()
with torch.no_grad():
  test_images = generator(test_noise.detach())
  for i in range(4):
	  plt.subplot(2, 2, 1 + i)
	  plt.axis('off')
	  plt.imshow(np.transpose(test_images.cpu().numpy()[i],(1,2,0)))
  plt.show()
lossesD = []
lossesG = []

num_epochs = 5
for epoch in range(num_epochs):
  discriminator.train()
  generator.train()
  lossD = 0
  lossG = 0
  prob_real = 0
  prob_fake = 0 
  for num_iter, (real_batch, _) in enumerate(dataloader):

    x_real = Variable(real_batch).to(device)
    optimizerD.zero_grad()
    pred_real = discriminator(x_real)
    pred_real.to(device)
    loss_real = criterion(pred_real.view(-1,1), (torch.ones(x_real.size(0),1)).to(device))
    loss_real.backward()
    z = noise(x_real.size(0)).to(device)
    x_fake = generator(z).to(device)
    x_fake.detach()
    pred_fake = discriminator(x_fake)
    pred_fake.to(device)
    loss_fake = criterion(pred_fake.view(-1,1), (torch.zeros(x_real.size(0),1)).to(device))
    loss_fake.backward()
    optimizerD.step()
    lossD = lossD + loss_real + loss_fake
    #prob_real = prob_real + np.mean(pred_real.detach().cpu().numpy())
    #prob_fake = prob_fake + np.mean(pred_fake.detach().cpu().numpy())
    #binary_pred_real = np.zeros((x_real.size(0),1))
    #binary_pred_fake = np.zeros((x_real.size(0),1))
    #binary_pred_real[pred_real.detach().cpu().numpy()>0.5]=1
    #binary_pred_fake[pred_fake.detach().cpu().numpy()>0.5]=1
    #accuracy_real = accuracy_real + accuracy_score(np.ones((x_real.size(0),1)),binary_pred_real)
    #accuracy_fake = accuracy_fake + accuracy_score(np.zeros((x_real.size(0),1)),binary_pred_fake)

    #zg = random_noise(x_real.size(0)).to(device)
    fake_x = generator(z).to(device)
    optimizerG.zero_grad()
    fake_pred = discriminator(fake_x)
    loss_gen = criterion(fake_pred.view(-1,1), (torch.ones(x_real.size(0),1)).to(device))
    loss_gen.backward()
    optimizerG.step()
    lossG = lossG + loss_gen

  lossesD.append(lossD/len(dataloader))
  lossesG.append(lossG/len(dataloader))
  print("Epoch No. = "+ str(epoch+1+20))
  print("Discriminator Loss = "+ str(lossesD[epoch].item()), "Generator Loss = "+ str(lossesG[epoch].item()))
  #print("Discriminator Confidence on Real Data = "+ str(prob_real/len(train_dl)), "Discriminator Confidence on Fake Data = "+str(prob_fake/len(train_dl)))
  torch.save(generator.state_dict(),'g_epoch-{}.pth'.format(epoch+1+20))
  torch.save(discriminator.state_dict(), 'd_epoch-{}.pth'.format(epoch+1+20))

  with torch.no_grad():
    generated_images = generator(fixed_noise.detach())
    for i in range(16):
      plt.subplot(4, 4, 1 + i)
      plt.axis('off')
      plt.imshow(np.transpose(generated_images.cpu().numpy()[i],(1,2,0)))
    plt.show()  

plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(lossesG,label="G")
plt.plot(lossesD,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()
with torch.no_grad():
  test_images = generator(test_noise.detach())
  for i in range(4):
	  plt.subplot(2, 2, 1 + i)
	  plt.axis('off')
	  plt.imshow(np.transpose(test_images.cpu().numpy()[i],(1,2,0)))
  plt.show()
test_noise = noise(4)
with torch.no_grad():
  test_images = generator(test_noise.detach())
  for i in range(4):
	  plt.subplot(2, 2, 1 + i)
	  plt.axis('off')
	  plt.imshow(np.transpose(test_images.cpu().numpy()[i],(1,2,0)))
  plt.show()
test_noise = noise(4)
with torch.no_grad():
  test_images = generator(test_noise.detach())
  for i in range(4):
	  plt.subplot(2, 2, 1 + i)
	  plt.axis('off')
	  plt.imshow(np.transpose(test_images.cpu().numpy()[i],(1,2,0)))
  plt.show()