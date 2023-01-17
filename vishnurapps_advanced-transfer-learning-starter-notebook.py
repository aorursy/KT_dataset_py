import os

import torch

import pandas as pd

import numpy as np

from torch.utils.data import Dataset, random_split, DataLoader

from PIL import Image

import torchvision.models as models

import matplotlib.pyplot as plt

from tqdm.notebook import tqdm

import torchvision.transforms as T

from sklearn.metrics import f1_score

import torch.nn.functional as F

import torch.nn as nn

from torchvision.utils import make_grid

%matplotlib inline
DATA_DIR = '../input/jovian-pytorch-z2g/Human protein atlas'



TRAIN_DIR = DATA_DIR + '/train'                           

TEST_DIR = DATA_DIR + '/test'                             



TRAIN_CSV = DATA_DIR + '/train.csv'                       

TEST_CSV = '../input/jovian-pytorch-z2g/submission.csv' 
data_df = pd.read_csv(TRAIN_CSV)

data_df.head()
labels = {

    0: 'Mitochondria',

    1: 'Nuclear bodies',

    2: 'Nucleoli',

    3: 'Golgi apparatus',

    4: 'Nucleoplasm',

    5: 'Nucleoli fibrillar center',

    6: 'Cytosol',

    7: 'Plasma membrane',

    8: 'Centrosome',

    9: 'Nuclear speckles'

}
def encode_label(label):

    target = torch.zeros(10)

    for l in str(label).split(' '):

        target[int(l)] = 1.

    return target



def decode_target(target, text_labels=False, threshold=0.5):

    result = []

    for i, x in enumerate(target):

        if (x >= threshold):

            if text_labels:

                result.append(labels[i] + "(" + str(i) + ")")

            else:

                result.append(str(i))

    return ' '.join(result)
class HumanProteinDataset(Dataset):

    def __init__(self, df, root_dir, transform=None):

        self.df = df

        self.transform = transform

        self.root_dir = root_dir

        

    def __len__(self):

        return len(self.df)    

    

    def __getitem__(self, idx):

        row = self.df.loc[idx]

        img_id, img_label = row['Image'], row['Label']

        img_fname = self.root_dir + "/" + str(img_id) + ".png"

        img = Image.open(img_fname)

        if self.transform:

            img = self.transform(img)

        return img, encode_label(img_label)
# imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

imagenet_stats = ([0.05438065, 0.05291743, 0.07920227], [0.39414383, 0.33547948, 0.38544176])



# train_tfms = T.Compose([

#     T.ToTensor(), 

#     T.Normalize(*imagenet_stats,inplace=True), 

# ])



train_tfms1 = T.Compose([

#     T.RandomCrop(512, padding=8, padding_mode='reflect'),

#     T.RandomResizedCrop(256, scale=(0.5,0.9), ratio=(1, 1)), 

#     T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),

    T.RandomRotation(90),

    T.RandomHorizontalFlip(p=0.3),

    T.RandomVerticalFlip(p=0.6),

    T.ToTensor(), 

    T.Normalize(*imagenet_stats,inplace=True), 

    T.RandomErasing(inplace=True)

])



train_tfms2 = T.Compose([

    T.RandomCrop(512, padding=8, padding_mode='reflect'),

#     T.RandomResizedCrop(256, scale=(0.5,0.9), ratio=(1, 1)), 

#     T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),

    T.RandomHorizontalFlip(p=0.9),

    T.RandomVerticalFlip(p=0.9),

    T.RandomAffine(degrees=(-45,45)),

    T.ToTensor(), 

    T.Normalize(*imagenet_stats,inplace=True), 

    T.RandomErasing(inplace=True)

])



valid_tfms = T.Compose([

#     T.Resize(256), 

    T.ToTensor(), 

    T.Normalize(*imagenet_stats)

])
y = data_df['Label'].values

type(y)

print(type(y[1]))
# y.shapee

print(type(y))

X = data_df.copy()
np.random.seed(42)

msk = np.random.rand(len(data_df)) < 0.9



train_df = data_df[msk].reset_index()

val_df = data_df[~msk].reset_index()

# from skmultilearn.model_selection import iterative_train_test_split

# train_df, train_y, val_df, test_y = iterative_train_test_split(X = data_df, y = y, test_size = 0.1)



# !pip install -qq iterative-stratification

# from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

# msss = MultilabelStratifiedShuffleSplit(n_splits=2, test_size=0.1, random_state=0)



# for train_index, test_index in msss.split(X, y):

#    print("TRAIN:", train_index, "TEST:", test_index)

#    X_train, X_test = X[train_index], X[test_index]

#    y_train, y_test = y[train_index], y[test_index]
# from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

# import numpy as np



# X = np.array([[1,2], [3,4], [1,2], [3,4], [1,2], [3,4], [1,2], [3,4]])

# y = np.array([[0,0], [0,0], [0,1], [0,1], [1,1], [1,1], [1,0], [1,0]])



# msss = MultilabelStratifiedShuffleSplit(n_splits=3, test_size=0.5, random_state=0)



# for train_index, test_index in msss.split(X, y):

#     print("TRAIN:", train_index, "TEST:", test_index)

#     X_train, X_test = X[train_index], X[test_index]

#     y_train, y_test = y[train_index], y[test_index]
from torch.utils.data import ChainDataset

# train_ds = HumanProteinDataset(train_df, TRAIN_DIR, transform=train_tfms)

train_ds1 = HumanProteinDataset(train_df, TRAIN_DIR, transform=train_tfms1)

train_ds2 = HumanProteinDataset(train_df, TRAIN_DIR, transform=train_tfms2)

val_ds = HumanProteinDataset(val_df, TRAIN_DIR, transform=valid_tfms)

len(train_ds1), len(val_ds)
def show_sample(img, target, invert=True):

    if invert:

        plt.imshow(1 - img.permute((1, 2, 0)))

    else:

        plt.imshow(img.permute(1, 2, 0))

    print('Labels:', decode_target(target, text_labels=True))
show_sample(*train_ds1[1541])
batch_size = 80
# train_dl = DataLoader(train_ds, batch_size, shuffle=True, 

#                       num_workers=3, pin_memory=True)

train_dl1 = DataLoader(train_ds1, batch_size, shuffle=True, 

                      num_workers=3, pin_memory=True)

train_dl2 = DataLoader(train_ds2, batch_size, shuffle=True, 

                      num_workers=3, pin_memory=True)

val_dl = DataLoader(val_ds, batch_size*2, 

                    num_workers=2, pin_memory=True)
def show_batch(dl, invert=True):

    for images, labels in dl:

        fig, ax = plt.subplots(figsize=(16, 8))

        ax.set_xticks([]); ax.set_yticks([])

        data = 1-images if invert else images

        ax.imshow(make_grid(data, nrow=16).permute(1, 2, 0))

        break
show_batch(train_dl1, invert=True)
def F_score(output, label, threshold=0.5, beta=1):

    prob = output > threshold

    label = label > threshold



    TP = (prob & label).sum(1).float()

    TN = ((~prob) & (~label)).sum(1).float()

    FP = (prob & (~label)).sum(1).float()

    FN = ((~prob) & label).sum(1).float()



    precision = torch.mean(TP / (TP + FP + 1e-12))

    recall = torch.mean(TP / (TP + FN + 1e-12))

    F2 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-12)

    return F2.mean(0)
class EarlyStopping:

    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False):

        """

        Args:

            patience (int): How long to wait after last time validation loss improved.

                            Default: 7

            verbose (bool): If True, prints a message for each validation loss improvement. 

                            Default: False

        """

        self.patience = patience

        self.verbose = verbose

        self.counter = 0

        self.best_score = None

        self.early_stop = False

        self.val_loss_min = np.Inf



    def __call__(self, val_loss, model):



        score = -val_loss



        if self.best_score is None:

            self.best_score = score

            self.save_checkpoint(val_loss, model)

        elif score < self.best_score:

            self.counter += 1

            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')

            if self.counter >= self.patience:

                self.early_stop = True

        else:

            self.best_score = score

            self.save_checkpoint(val_loss, model)

            self.counter = 0



    def save_checkpoint(self, val_loss, model):

        '''Saves model when validation loss decrease.'''

        if self.verbose:

            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        torch.save(model.state_dict(), 'checkpoint.pt')

        self.val_loss_min = val_loss
class MultilabelImageClassificationBase(nn.Module):

    def training_step(self, batch):

        images, targets = batch 

        out = self(images)                      

        loss = F.binary_cross_entropy(out, targets)      

        return loss

    

    def validation_step(self, batch):

        images, targets = batch 

        out = self(images)                           # Generate predictions

        loss = F.binary_cross_entropy(out, targets)  # Calculate loss

        score = F_score(out, targets)

        return {'val_loss': loss.detach(), 'val_score': score.detach() }

        

    def validation_epoch_end(self, outputs):

        batch_losses = [x['val_loss'] for x in outputs]

        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses

        batch_scores = [x['val_score'] for x in outputs]

        epoch_score = torch.stack(batch_scores).mean()      # Combine accuracies

        return {'val_loss': epoch_loss.item(), 'val_score': epoch_score.item()}

    

    def epoch_end(self, epoch, result):

        print("Epoch [{}], last_lr: {:.4f}, train_loss: {:.4f}, val_loss: {:.4f}, val_score: {:.4f}".format(

            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_score']))
# resnet34 = models.resnet34(pretrained=True)

# resnet34
# fc = nn.Sequential(

#     nn.Linear(512, 256),

#     nn.ReLU(),

#     nn.Dropout(0.5),

    

#     nn.Linear(256, 128),

#     nn.ReLU(),

#     nn.Dropout(0.5),

    

#     nn.Linear(128, 64),

#     nn.ReLU(),

#     nn.Dropout(0.5),

    

#     nn.Linear(64,10),

    

# )

from collections import OrderedDict

fc = nn.Sequential(OrderedDict([

('bn1', nn.BatchNorm1d(512)),

('drop1', nn.Dropout(p=0.5)),

('linear1', nn.Linear(512, 256)),

('drop2', nn.Dropout(p=0.5)),

('linear2', nn.Linear(256, 64)),

('drop3', nn.Dropout(p=0.5)),

('linear3', nn.Linear(64, 10))

]))



# resnet34.fc = fc

# resnet34
class ProteinResnet(MultilabelImageClassificationBase):

    def __init__(self, fc):

        super().__init__()

        # Use a pretrained model

        self.network = models.resnet34(pretrained=True)

        # Replace last layer

#         num_ftrs = self.network.fc.in_features

        self.network.fc = fc

    

    def forward(self, xb):

        return torch.sigmoid(self.network(xb))

    

    def freeze(self):

        # To freeze the residual layers

        for param in self.network.parameters():

            param.require_grad = False

        for param in self.network.fc.parameters():

            param.require_grad = True

    

    def unfreeze(self):

        # Unfreeze all layers

        for param in self.network.parameters():

            param.require_grad = True
def get_default_device():

    """Pick GPU if available, else CPU"""

    if torch.cuda.is_available():

        return torch.device('cuda')

    else:

        return torch.device('cpu')

    

def to_device(data, device):

    """Move tensor(s) to chosen device"""

    if isinstance(data, (list,tuple)):

        return [to_device(x, device) for x in data]

    return data.to(device, non_blocking=True)



class DeviceDataLoader():

    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):

        self.dl = dl

        self.device = device

        

    def __iter__(self):

        """Yield a batch of data after moving it to device"""

        for b in self.dl: 

            yield to_device(b, self.device)



    def __len__(self):

        """Number of batches"""

        return len(self.dl)
device = get_default_device()

device
# train_dl = DeviceDataLoader(train_dl, device)

train_dl1 = DeviceDataLoader(train_dl1, device)

train_dl2 = DeviceDataLoader(train_dl2, device)

val_dl = DeviceDataLoader(val_dl, device)
@torch.no_grad()

def evaluate(model, val_loader):

    model.eval()

    outputs = [model.validation_step(batch) for batch in val_loader]

    return model.validation_epoch_end(outputs)



def get_lr(optimizer):

    for param_group in optimizer.param_groups:

        return param_group['lr']



def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader, early_stopping,  

                  weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD):

    torch.cuda.empty_cache()

    history = []

    

    # Set up cutom optimizer with weight decay

    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)

    # Set up one-cycle learning rate scheduler

    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, 

                                                steps_per_epoch=len(train_loader))

    

    for epoch in range(epochs):

        # Training Phase 

        model.train()

        train_losses = []

        lrs = []

        for batch in tqdm(train_loader):

            loss = model.training_step(batch)

            train_losses.append(loss)

            loss.backward()

            

            # Gradient clipping

            if grad_clip: 

                nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            

            optimizer.step()

            optimizer.zero_grad()

            

            # Record & update learning rate

            lrs.append(get_lr(optimizer))

            sched.step()

        

        # Validation phase

        result = evaluate(model, val_loader)

        result['train_loss'] = torch.stack(train_losses).mean().item()

        result['lrs'] = lrs

        model.epoch_end(epoch, result)

        history.append(result)

        

        early_stopping(history[-1]['val_loss'], model)

        

        if early_stopping.early_stop:

            print("Early stopping")

            break

            

    return history
model = to_device(ProteinResnet(fc), device)
history = [evaluate(model, val_dl)]

history
model.freeze()
# early_stopping = EarlyStopping(patience=5, verbose=True)

epochs = 50

max_lr = 0.0001

grad_clip = 0.1

weight_decay = 1e-4

opt_func = torch.optim.Adam
# %%time

# history += fit_one_cycle(epochs, max_lr, model, train_dl, val_dl, early_stopping,

#                          grad_clip=grad_clip, 

#                          weight_decay=weight_decay, 

#                          opt_func=opt_func)
%%time

early_stopping = EarlyStopping(patience=5, verbose=True)

history += fit_one_cycle(epochs, max_lr, model, train_dl1, val_dl, early_stopping,

                         grad_clip=grad_clip, 

                         weight_decay=weight_decay, 

                         opt_func=opt_func)
%%time

early_stopping = EarlyStopping(patience=5, verbose=True)

history += fit_one_cycle(epochs, max_lr, model, train_dl2, val_dl, early_stopping,

                         grad_clip=grad_clip, 

                         weight_decay=weight_decay, 

                         opt_func=opt_func)
model.unfreeze()

early_stopping = EarlyStopping(patience=5, verbose=True)
%%time

history += fit_one_cycle(epochs, max_lr/10, model, train_dl, val_dl, early_stopping,

                         grad_clip=grad_clip, 

                         weight_decay=weight_decay, 

                         opt_func=opt_func)
# train_time='29:17'
# load the last checkpoint with the best model

model.load_state_dict(torch.load('checkpoint.pt'))
def plot_scores(history):

    scores = [x['val_score'] for x in history]

    plt.plot(scores, '-x')

    plt.xlabel('epoch')

    plt.ylabel('score')

    plt.title('F1 score vs. No. of epochs');
plot_scores(history)
def plot_losses(history):

    train_losses = [x.get('train_loss') for x in history]

    val_losses = [x['val_loss'] for x in history]

    plt.plot(train_losses, '-bx')

    plt.plot(val_losses, '-rx')

    plt.xlabel('epoch')

    plt.ylabel('loss')

    plt.legend(['Training', 'Validation'])

    plt.title('Loss vs. No. of epochs');
plot_losses(history)
def plot_lrs(history):

    lrs = np.concatenate([x.get('lrs', []) for x in history])

    plt.plot(lrs)

    plt.xlabel('Batch no.')

    plt.ylabel('Learning rate')

    plt.title('Learning Rate vs. Batch no.');
plot_lrs(history)
def predict_single(image):

    xb = image.unsqueeze(0)

    xb = to_device(xb, device)

    preds = model(xb)

    prediction = preds[0]

    print("Prediction: ", prediction)

    show_sample(image, prediction)
test_df = pd.read_csv(TEST_CSV)

test_dataset = HumanProteinDataset(test_df, TEST_DIR, transform=valid_tfms)
img, target = test_dataset[0]

img.shape
predict_single(test_dataset[100][0])
predict_single(test_dataset[74][0])
test_dl = DeviceDataLoader(DataLoader(test_dataset, batch_size, num_workers=3, pin_memory=True), device)
@torch.no_grad()

def predict_dl(dl, model):

    torch.cuda.empty_cache()

    batch_probs = []

    for xb, _ in tqdm(dl):

        probs = model(xb)

        batch_probs.append(probs.cpu().detach())

    batch_probs = torch.cat(batch_probs)

    return [decode_target(x) for x in batch_probs]
test_preds = predict_dl(test_dl, model)
submission_df = pd.read_csv(TEST_CSV)

submission_df.Label = test_preds

submission_df.sample(20)
sub_fname = 'submission.csv'
submission_df.to_csv(sub_fname, index=False)
weights_fname = 'protein-resnet.pth'

torch.save(model.state_dict(), weights_fname)
# !pip install jovian --upgrade --quiet
# import jovian
# jovian.reset()

# jovian.log_hyperparams(arch='resnet34', 

#                        epochs=2*epochs, 

#                        lr=max_lr, 

#                        scheduler='one-cycle', 

#                        weight_decay=weight_decay, 

#                        grad_clip=grad_clip,

#                        opt=opt_func.__name__)
# jovian.log_metrics(val_loss=history[-1]['val_loss'], 

#                    val_score=history[-1]['val_score'],

#                    train_loss=history[-1]['train_loss'],

#                    time=train_time)
# project_name='protein-advanced'
# jovian.commit(project=project_name, environment=None, outputs=[weights_fname])
# history[]