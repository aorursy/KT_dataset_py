import os

import random



import numpy as np

import pandas as pd

import cv2



import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import accuracy_score



import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.optim import Adam

from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms





%matplotlib inline
def seed_everything(seed=42):

    os.environ['PYTHONHASHSEED'] = str(seed)

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
INPUT_DIR = '../input/ailab-ml-training-1/'



PATH = {

    'train': os.path.join(INPUT_DIR, 'train.csv'),

    'sample_submission': os.path.join(INPUT_DIR, 'sample_submission.csv'),

    'train_image_dir': os.path.join(INPUT_DIR, 'train_images/train_images'),

    'test_image_dir': os.path.join(INPUT_DIR, 'test_images/test_images'),

}



ID = 'fname'

TARGET = 'label'



SEED = 42

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

seed_everything(SEED)



PARAMS = {

    'n_splits': 5,

    'batch_size': 64,

    'epochs': 5,

    'lr': 0.001,

    'valid_batch_size': 256,

    'test_batch_size': 256,

}
train_df = pd.read_csv(PATH['train'])

sample_submission_df = pd.read_csv(PATH['sample_submission'])
print(f'number of train data: {len(train_df)}')

print(f'number of test data: {len(sample_submission_df)}')
print(f'number of unique label: {train_df[TARGET].nunique()}')
sns.countplot(train_df[TARGET])

plt.title('train label distribution')

plt.show()
train_df.head()
sample = train_df.groupby(TARGET).first().reset_index()



fig, ax = plt.subplots(2, 5)

fig.set_size_inches(4 * 5, 4 * 2)



for i, row in sample.iterrows():

    fname, label = row[ID], row[TARGET]

    img = cv2.imread(os.path.join(PATH['train_image_dir'], fname))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ax[i//5,i%5].imshow(img, 'gray')

    ax[i//5,i%5].set_title(f'{fname} - label: {label}')
print(f'shape of image: {img.shape}')
class KMNISTDataset(Dataset):

    def __init__(self, fname_list, label_list, image_dir, transform=None):

        super().__init__()

        self.fname_list = fname_list

        self.label_list = label_list

        self.image_dir = image_dir

        self.transform = transform

    

    def __len__(self):

        return len(self.fname_list)

    

    def __getitem__(self, idx):

        fname = self.fname_list[idx]

        label = self.label_list[idx]

        

        image = cv2.imread(os.path.join(self.image_dir, fname))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if self.transform is not None:

            image = self.transform(image)

        

        return image, label
class MLP(nn.Module):

    def __init__(self, input_dim=28*28, hidden_dim=128, output_dim=10):

        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, output_dim)

        self.activation = nn.ReLU()

    

    def forward(self, x):

        x = x.view(x.size(0), -1)

        x = self.fc1(x)

        x = self.activation(x)

        x = self.fc2(x)

        

        return x
def accuracy_score_torch(y_pred, y):

    y_pred = torch.argmax(y_pred, axis=1).cpu().numpy()

    y = y.cpu().numpy()



    return accuracy_score(y_pred, y)
kf = StratifiedKFold(n_splits=PARAMS['n_splits'], random_state=SEED, shuffle=True)

oof = np.zeros(len(train_df))

state_dicts = []



for i, (dev_idx, val_idx) in enumerate(kf.split(train_df[ID], train_df[TARGET])):

    print(f'fold: {i}')

    

    dev_df = train_df.iloc[dev_idx, :].reset_index(drop=True)

    val_df = train_df.iloc[val_idx, :].reset_index(drop=True)

    

    transform = transforms.Compose([

        transforms.ToTensor(),

    ])

    

    dev_dataset = KMNISTDataset(dev_df[ID], dev_df[TARGET], PATH['train_image_dir'], transform=transform)

    val_dataset = KMNISTDataset(val_df[ID], val_df[TARGET], PATH['train_image_dir'], transform=transform)



    dev_dataloader = DataLoader(dev_dataset, batch_size=PARAMS['batch_size'], shuffle=True)

    val_dataloader = DataLoader(val_dataset, batch_size=PARAMS['valid_batch_size'], shuffle=False)



    model = MLP().to(DEVICE)

    optim = Adam(model.parameters(), lr=PARAMS['lr'])

    criterion = nn.CrossEntropyLoss()

    

    for epoch in range(PARAMS['epochs']):

        model.train()

        dev_loss_list = []

        dev_acc_list = []



        for x, y in dev_dataloader:

            x = x.to(dtype=torch.float32, device=DEVICE)

            y = y.to(dtype=torch.long, device=DEVICE)



            optim.zero_grad()

            y_pred = model(x)

            loss = criterion(y_pred, y)

            loss.backward()

            optim.step()



            dev_loss_list.append(loss.item())

            dev_acc_list.append(accuracy_score_torch(y_pred, y))



        model.eval()

        val_loss_list = []

        val_acc_list = []



        for x, y in val_dataloader:

            x = x.to(dtype=torch.float32, device=DEVICE)

            y = y.to(dtype=torch.long, device=DEVICE)



            with torch.no_grad():

                y_pred = model(x)

                loss = criterion(y_pred, y)



            val_loss_list.append(loss.item())

            val_acc_list.append(accuracy_score_torch(y_pred, y))



        print('epoch: {}/{} - loss: {:.5f} - acc: {:.3f} - val_loss: {:.5f} - val_acc: {:.3f}'.format(

            epoch,

            PARAMS['epochs'], 

            np.mean(dev_loss_list),

            np.mean(dev_acc_list),

            np.mean(val_loss_list),

            np.mean(val_acc_list),

        ))

        

    state_dicts.append(model.state_dict())

    

    model.eval()

    predictions = []



    for x, _ in val_dataloader:

        x = x.to(dtype=torch.float32, device=DEVICE)



        with torch.no_grad():

            y_pred = model(x)

            y_pred = torch.argmax(y_pred, axis=1).cpu().numpy()

            y_pred = y_pred.tolist()



        predictions += y_pred

    

    oof[val_idx] = np.array(predictions)
score = accuracy_score(oof, train_df[TARGET])

print(f'CV: {score:.5f}')
transform = transforms.Compose([

    transforms.ToTensor(),

])



test_dataset = KMNISTDataset(

    sample_submission_df[ID],

    sample_submission_df[TARGET],

    PATH['test_image_dir'],

    transform=transform

)



test_dataloader = DataLoader(test_dataset, batch_size=PARAMS['test_batch_size'], shuffle=False)
predictions = np.zeros((len(sample_submission_df), 10))



for state_dict in state_dicts:

    model = MLP().to(DEVICE)

    model.eval()

    model.load_state_dict(state_dict)

    preds = []

    

    for x, _ in test_dataloader:

        x = x.to(dtype=torch.float32, device=DEVICE)

        

        with torch.no_grad():

            y_pred = model(x)

            preds.append(y_pred.cpu().numpy())

    

    preds = np.concatenate(preds)

    predictions += preds / len(state_dicts)
predictions = np.argmax(predictions, axis=1).tolist()

sample_submission_df[TARGET] = predictions
sample_submission_df.to_csv('submission.csv', index=False)

from IPython.display import FileLink

FileLink('submission.csv')
sns.countplot(sample_submission_df[TARGET])

plt.title('test prediction label distribution')

plt.show()
fig, ax = plt.subplots(2, 5)

fig.set_size_inches(4 * 5, 4 * 2)



for i, row in sample_submission_df.iloc[:10,:].iterrows():

    fname, label = row[ID], row[TARGET]

    img = cv2.imread(os.path.join(PATH['test_image_dir'], fname))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ax[i//5,i%5].imshow(img, 'gray')

    ax[i//5,i%5].set_title(f'{fname} - label: {label}')