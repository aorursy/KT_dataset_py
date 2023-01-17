import torch

from torch import nn, optim

import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from torchvision import datasets, transforms

from sklearn import model_selection



import matplotlib.pyplot as plt

import pandas as pd

import numpy as np
class MNIST(Dataset):

    def __init__(self, df, labels=True, transform=None):

        

        self.df = df

        self.labels = labels

        self.transform = transform

    

    def __len__(self):

        return len(self.df)

    

    def __getitem__(self, index):

        if torch.is_tensor(index):

            index = index.tolist()

   

        if self.labels:

            label = self.df.iloc[index, 0]

            data  = self.df.iloc[index, 1:].to_numpy(np.uint8).reshape(28,28, 1)

            if self.transform:

                data = self.transform(data)

            return data, label

    

        data  = self.df.iloc[index].to_numpy(dtype=np.uint8).reshape(28,28, 1)

        

        if self.transform:

            data = self.transform(data)

        

        return data
augment = transforms.Compose([transforms.ToPILImage(),

                                transforms.RandomRotation(10, fill=(0,)),

                                transforms.RandomPerspective(),

                                transforms.RandomAffine(10),

                                transforms.ToTensor(),

                                transforms.Normalize((0.5,), (0.5,))])



tfs = transforms.Compose([transforms.ToTensor(),

                          transforms.Normalize((0.5,), (0.5,))])
train_df = pd.read_csv('../input/digit-recognizer/train.csv')

X, y = train_df.iloc[:, 1:], train_df.iloc[:, 0]

x_train, x_valid, y_train, y_valid = model_selection.train_test_split(X, y)



trainset = MNIST(pd.concat([y_train, x_train], axis=1), transform=augment)

validset = MNIST(pd.concat([y_valid, x_valid], axis=1), transform=tfs)



trainload = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4)

validload = DataLoader(validset, batch_size=32, shuffle=True, num_workers=4)
images, labels = iter(validload).next()



fig = plt.figure(figsize=(25,4))



for i in range(20):

    ax = fig.add_subplot(2, 20/2, i+1, xticks=[], yticks=[])

    ax.set_title(f'Label: {labels[i].item()}')

    ax.imshow(images[i].squeeze())



plt.tight_layout()

plt.show()



images[0].dtype, images[0].shape, images.shape
class LeNet(nn.Module):

    def __init__(self):

        super().__init__()

        

        self.conv1 = nn.Sequential(

            nn.Conv2d(1, 6, 5),

            nn.ReLU(),

            nn.AvgPool2d(2, stride=2),

        )

        

        self.conv2 = nn.Sequential(

            nn.Conv2d(6, 16, 5),

            nn.ReLU(),

            nn.AvgPool2d(2, stride=2)

        )

        

        self.fc = nn.Sequential(

            nn.Flatten(),

            nn.Linear(4*4*16, 120),

            nn.ReLU(),

            nn.Linear(120, 84),

            nn.ReLU()

        )

        

        self.out = nn.Linear(84, 10)

        

    def forward(self, x):

        x = self.conv1(x)

        x = self.conv2(x)

        x = self.fc(x)

        return self.out(x)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = LeNet()

model = model.to(device)
opt = optim.Adam(model.parameters(), lr=5e-4)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=7)

criterion = nn.CrossEntropyLoss()

epochs = 40



train_loss = []

valid_loss = []

accuracy   = []

valid_low = np.Inf





for e in range(epochs):

    running_tl = 0

    running_vl = 0

    running_ac = 0

    

    # backprop and and update

    model.train()

    for images, labels in trainload:

        images, labels = images.to(device), labels.to(device)

        opt.zero_grad()

        t_cel = criterion(model(images.float()), labels)

        t_cel.backward()

        opt.step()

        running_tl += t_cel.item()

        

    # validation pass    

    with torch.no_grad():

        model.eval()

        for images, labels in validload:

            images, labels = images.to(device), labels.to(device)

            scores = model(images.float())

            ps = F.softmax(scores, dim=1)

            v_cel = criterion(scores, labels)

            pred = torch.argmax(ps, dim=1)

            running_ac += (pred == labels).cpu().numpy().mean()

            running_vl += v_cel.item()

    

    # Decay Learning Rate:

    print(f'Epoch {e} Learning Rate: {opt.param_groups[0]["lr"]} Validation Loss: {v_cel}')

    scheduler.step(v_cel)



    # get loss metrics for plotting later

    train_loss.append(running_tl/len(trainload))

    valid_loss.append(running_vl/len(validload))

    accuracy.append(running_ac/len(validload))
plt.title('Loss vs Epochs')

plt.xlabel('Epochs')

plt.plot(train_loss, label='Training')

plt.plot(valid_loss, label='Validation')

plt.legend()
plt.title('Accuracy on Validation Set by Epoch')

plt.xlabel('Epochs')

plt.plot(accuracy)
torch.save(model.state_dict(), 'alltrans_50e.pt')

accuracy[-1]
comp_data = pd.read_csv('../input/digit-recognizer/test.csv')

comp_data.head()
comp_set = MNIST(comp_data, labels=False, transform=tfs)

comp_loader = DataLoader(comp_set, batch_size=32, num_workers=0, shuffle=False)
predictions = np.array([])



for images in comp_loader:

    images = images.to(device).float()

    scores = model(images)

    ps = F.softmax(scores, dim=1)

    preds = torch.argmax(ps, dim=1).cpu()

    predictions = np.append(predictions, preds)

    
predictions = predictions.astype(np.int)
sub_df = pd.DataFrame({'ImageId': np.arange(1, len(predictions) + 1),

                       'Label': predictions})



sub_df.head(10)
images = iter(comp_loader).next()



fig = plt.figure(figsize=(25,4))



for i in range(20):

    ax = fig.add_subplot(2, 20/2, i+1, xticks=[], yticks=[])

    ax.set_title(f'Label: {sub_df.loc[i, "Label"]}')

    ax.imshow(images[i].squeeze())



plt.tight_layout()

plt.show()
sub_df.to_csv('submission_LeNet.csv', index=False)
# ! kaggle competitions submit -c digit-recognizer -f submission_LeNet.csv -m "same as last +10 epochs"