%matplotlib inline

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import xarray as xr



from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_curve, auc



import torch.nn as nn

import torch.optim as optim

import torch.nn.functional as F



import torch

import torchvision



torch.__version__, torchvision.__version__
# Check GPU

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)
def predict(net, loader, record_y=False):



    y_pred_list = []

    with torch.no_grad():



        total_loss = 0.0

        total_sample = 0

        for i, data in enumerate(loader, 0):

            inputs, labels = data[0].to(device), data[1].to(device)

            outputs = net(inputs)

            loss = criterion(outputs, labels)

            

            current_sample = len(labels)

            total_sample += current_sample

            total_loss += loss.item() * current_sample



            if record_y:

                y_pred_list.append(outputs.cpu().numpy())

                

    avg_loss = total_loss / total_sample

    print(f"Average loss: {avg_loss}")

    

    if record_y:

        y_pred = np.concatenate(y_pred_list)

        return y_pred

    else:

        return avg_loss

ds = xr.open_dataset('../input/chest-xray-cleaned/chest_xray.nc')

ds
ds['image'].isel(sample=slice(0, 12)).plot(col='sample', col_wrap=4, cmap='gray')
ds['label'].mean(dim='sample').to_pandas().plot.barh()  # proportion
all_labels = ds['feature'].values.astype(str)

all_labels
%%time

X_all = ds['image'].data[:, np.newaxis, :, :]  # pytorch use channel-first, unlike Keras

y_all = ds['label'].data.astype(np.float32)  

# https://discuss.pytorch.org/t/runtimeerror-expected-object-of-scalar-type-long-but-got-scalar-type-float-when-using-crossentropyloss/30542/4

# https://discuss.pytorch.org/t/expected-object-of-scalar-type-float-but-got-scalar-type-long-for-argument-2-target/44109



X_train, X_test, y_train, y_test = train_test_split(

    X_all, y_all, test_size=0.2, random_state=42

)



X_train.shape, X_test.shape, y_train.shape, y_test.shape
trainset = torch.utils.data.TensorDataset(

    torch.from_numpy(X_train), torch.from_numpy(y_train)

)



trainloader = torch.utils.data.DataLoader(

    trainset, batch_size=32, shuffle=True, num_workers=2

)



testset = torch.utils.data.TensorDataset(

    torch.from_numpy(X_test), torch.from_numpy(y_test)

)



testloader = torch.utils.data.DataLoader(

    testset, batch_size=32, shuffle=False, num_workers=2

)
dataiter = iter(trainloader)

data = dataiter.next()

inputs, labels = data[0].to(device), data[1].to(device)

inputs.shape, labels.shape  # batch, channel, x, y
inputs.dtype, labels.dtype
class Net(nn.Module):

    def __init__(self,):

        super(Net, self).__init__()

        

        channel_1 = 16

        channel_2 = 32

        channel_3 = 64

        

        self.conv1 = nn.Conv2d(1, channel_1, 3, padding=1)

        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(channel_1, channel_2, 3, padding=1)

        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(channel_2, channel_3, 3, padding=1)

        

        last_x = 32  # spatial size = /128 / 2 / 2

        self.last_size = channel_3 * last_x * last_x

        self.fc1 = nn.Linear(self.last_size, 14)  # need to flatten to filter * x * y



    def forward(self, x):

        x = F.relu(self.conv1(x))

        x = F.relu(self.pool1(x))

        x = F.relu(self.conv2(x))

        x = F.relu(self.pool2(x))

        x = F.relu(self.conv3(x))

        x = x.view(-1, self.last_size)  # flatten

        x = self.fc1(x)

        x = torch.sigmoid(x)

        # https://discuss.pytorch.org/t/usage-of-cross-entropy-loss/14841/5

        return x



net = Net()
%time net.to(device)
%%time



# criterion = nn.MSELoss()

criterion = nn.BCELoss()



optimizer = optim.Adam(net.parameters())



print_freq = 400  # print loss per that many steps



train_history = []

eval_history = []

for epoch in range(5):  # loop over the dataset multiple times



    running_loss = 0.0

    for i, data in enumerate(trainloader, 0):

        # get the inputs; data is a list of [inputs, labels]

        # inputs, labels = data

        inputs, labels = data[0].to(device), data[1].to(device)



        # zero the parameter gradients

        optimizer.zero_grad()



        # forward + backward + optimize

        outputs = net(inputs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()



        # print statistics

        running_loss += loss.item()

        if i % print_freq == print_freq-1:

            print('[%d, %5d] loss: %.4f' %

                  (epoch + 1, i + 1, running_loss / print_freq))

            running_loss = 0.0

            

    print('Training loss:')

    train_loss = predict(net, trainloader, record_y=False)

    train_history.append(train_loss)

    

    # we shouldn't actually monitor test loss; just for convenience

    print('Validation loss:')

    eval_loss = predict(net, testloader, record_y=False)

    eval_history.append(eval_loss)

    print('-- new epoch --')



print('Finished Training')
df_history = pd.DataFrame(

    np.stack([train_history, eval_history], axis=1), 

    columns=['train_loss', 'val_loss'],

)

df_history.index.name = 'epoch'

df_history
plt.rcParams['font.size'] = 14

df_history.plot(grid=True, marker='o', ylim=[0, None], linewidth=3.0, alpha=0.8)

plt.title('Simple CNN on Chest-xray')
%%time 



# do not shuffle, so that the output order matches true label

y_train_pred = predict(

    net, 

    torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=False, num_workers=2),

    record_y=True

)



y_test_pred = predict(net, testloader, record_y=True)
def plot_ruc(y_true, y_pred):

    fig, c_ax = plt.subplots(1,1, figsize = (9, 9))

    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')

    

    for (idx, c_label) in enumerate(all_labels):

        fpr, tpr, thresholds = roc_curve(y_true[:,idx].astype(int), y_pred[:,idx])

        c_ax.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % (c_label, auc(fpr, tpr)))



    c_ax.legend()

    c_ax.set_xlabel('False Positive Rate')

    c_ax.set_ylabel('True Positive Rate')
plot_ruc(y_test, y_test_pred)

plt.title('ROC test set')
plot_ruc(y_train, y_train_pred)

plt.title('ROC train set')