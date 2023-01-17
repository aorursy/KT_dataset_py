import numpy as np

import pandas as pd



import os

import torch

import torch.nn as nn



from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import roc_auc_score as auc



import warnings

warnings.filterwarnings('ignore')
TRAIN_PATH = '/kaggle/input/cat-in-the-dat-ii/train.csv'

TEST_PATH = '/kaggle/input/cat-in-the-dat-ii/test.csv'



train_df = pd.read_csv(TRAIN_PATH)

test_df = pd.read_csv(TEST_PATH)



#train_df.head()
# add a target column with -1 in each instance.

test_df.loc[:, 'target'] = -1
data = pd.concat([train_df, test_df]).reset_index(drop=True)

data.shape, train_df.shape, test_df.shape
features = [f for f in train_df.columns if f not in ['id', 'target']]

print(features)
# I've the features now. Let's do the Label Encoding and then we would do entity encoding

# for each feature in features we fit and return encoded labels 

def label_encoder(data, features):

    for feat in features:

        le = LabelEncoder()

        data.loc[:, feat] = le.fit_transform(data[feat].astype(str).fillna('-1').values)

        

    return data



data = label_encoder(data, features)

train = data[:500000]

valid = data[500000:600000]

test = data[600000:]
data.shape, train.shape, valid.shape, test.shape
device = 'cuda' if torch.cuda.is_available() else 'cpu'

device
# Always state what you want from a function

# This function will return embedding for all of the categorical features

def entity_embedding(df, col):

    inputs = []

    vector_size = 0

    for c in col:

        unique_values = df[c].nunique()

        vector_size = vector_size + int(min(np.ceil(unique_values / 2), 64))

        #embedding = nn.Embedding(len(df), int(min(np.ceil(unique_values / 2), 10)))

        #embed = embedding(torch.tensor((df[c].values), dtype=torch.long))

        #inputs.append(embed)

    return vector_size #, inputs



vector_size = entity_embedding(train, features)



# concat each vector along it's dim=1

#x = torch.cat([e for e in entEmb], dim=1)



#x.size(), 

vector_size
class Dataset(torch.utils.data.Dataset):

    def __init__(self, df, features):

        self.df = df.drop(['id', 'target'], axis=1).values

        self.target = df.target.values

        self.features = features

        self.unique_values = [int(df[feat].nunique()) for feat in self.features]

        

    def __len__(self):

        return len(self.df)    

    

    def __getitem__(self, idx):

        inputs = self.df[idx]

        targets = self.target[idx]

        unique_vals = self.unique_values

        sample = {'inputs': inputs, 'targets': targets, 'unique': unique_vals}

        

        return sample
dataset = Dataset(train, features)
# In this cell we're trying to create a list of embeddings

unique_vals = dataset[:]['unique']

emb_list = [nn.Embedding(4, int(min(np.ceil(val / 2), 50))) for val in unique_vals]

for idx, emb in zip(list([i for i in range(23)]), emb_list):

    print(f"Index: {idx},    {emb}")
# list of embeddings vector

# [emb(labels_idx) for idx, emb zip(index, emb_List)]

#emb = [emb(torch.tensor((dataset[0:4]['inputs'][:, idx]), dtype=torch.long)) for idx, emb in zip(list([i for i in range(23)]),emb_list)]

#emb = torch.cat([e for e in emb], dim=1)

#emb.size()
# This class will take in the input from dataset and 

# returns the flattened vector



class EmbeddingLayer(nn.Module):

    def __init__(self, unique_vals):

        super(EmbeddingLayer, self).__init__()

        

        self.embed_list = [nn.Embedding(100000, int(min(np.ceil(val / 2), 64))) for val in unique_vals]

            

    def forward(self, inputs):

        emb = [emb(torch.tensor((inputs[:, idx]), dtype=torch.long)) for idx, emb in zip(list([i for i in range(23)]), self.embed_list)]

        out = torch.cat([e for e in emb], dim=1)

        return out



print(dataset[0:4]['inputs'].shape)

embb = EmbeddingLayer(unique_vals)

embb(dataset[0:4]['inputs']).size()       
trainset = Dataset(train, features)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)



validset = Dataset(valid, features)

validloader = torch.utils.data.DataLoader(validset, batch_size=128, shuffle=False)



testset = Dataset(test, features)

testloader = torch.utils.data.DataLoader(testset, batch_size=64)



len(trainloader), len(validloader), len(testloader)
class Model(nn.Module):

    def __init__(self, input_size, unique_vals, device, hidden_size=1024, dropout=0.5):

        super(Model, self).__init__()

        

        self.fc1 = nn.Linear(input_size, hidden_size)

        self.fc2 = nn.Linear(hidden_size, hidden_size)

        self.fc3 = nn.Linear(hidden_size, 1)

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(dropout)

        self.embedding = EmbeddingLayer(unique_vals)

        self.sigmoid = nn.Sigmoid()

        self.device = device

        

        self.bn1 = nn.BatchNorm1d(hidden_size)

        self.bn2 = nn.BatchNorm1d(hidden_size)

        

    def forward(self, le_df):

        embed = self.embedding(le_df)

        embed = embed.to(self.device)

        out = self.bn1(self.relu(self.fc1(embed)))

        

        out = self.dropout(out)

        

        out = self.bn2(self.relu(self.fc2(out)))

        out = self.dropout(out)

        

        out = self.fc3(out)

        #print(out.size(), '------')

        return self.sigmoid(out)

 



model = Model(vector_size, unique_vals, device).to(device)
for data in trainloader:

    inp, trg = data['inputs'], data['targets']

    print(inp.size(), trg.size())

    break
# important series of operations to free tensor from cuda() and convert into simple numpy() array

torch.randn(3, 1).to(device).detach().cpu().numpy()
def train(dataloader, model, criterion, optimizer, auc, device):

    model.train()

    

    prev_auc = []

    

    print('Training: \n')

    

    for i, data in enumerate(dataloader):

        inputs, target = data['inputs'], data['targets']

        

        inputs = inputs.to(device)

        target = torch.tensor(target, dtype=torch.float)

        target = target.to(device)

        #print(inputs.size(), target.size())

        optimizer.zero_grad()

        output = model(inputs)

        #print(f"Output: {output.squeeze(1).size()}")

        #print(f"Target: {target.size()}")

        loss = criterion(output, target)

        auc_score = auc(target.detach().cpu().numpy(), output.detach().cpu().numpy())

        #print(target.detach().cpu().numpy().shape)

        loss.backward()

        optimizer.step()

        

        if i % 400 == 0:

            print(f"bi: {i},  loss: {loss.item():.4f},  auc: {auc_score:.4f}")

        

            if len(prev_auc) == 0:

                prev_auc.append(auc_score)



            if (len(prev_auc) > 0) and (auc_score > max(prev_auc)):

                prev_auc.append(auc_score)

                torch.save(model, f'model{len(prev_auc)}.pth')

        

    return loss.item()





def evaluate(dataloader, model, criterion, optimizer, auc, device):

    model.eval()

    

    scores = []

    print('\n')

    print('Validation: \n')

    for i, data in enumerate(dataloader):

        inputs, target = data['inputs'], data['targets']

        

        inputs = inputs.to(device)

        target = torch.tensor(target, dtype=torch.float)

        target = target.to(device)

        optimizer.zero_grad()

        output = model(inputs)

        loss = criterion(output, target)

        auc_score = auc(target.detach().cpu().numpy(), output.detach().cpu().numpy())

        scores.append(auc_score)

        

        if i % 100 == 0:

            print(f"bi: {i},  loss: {loss.item():.4f},  auc: {auc_score:.4f}")



    return loss.item(), np.mean(scores)
criterion = nn.BCELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
for epoch in range(1):

    train_loss = train(trainloader, model, criterion, optimizer, auc, device)

    val_loss, val_acc = evaluate(validloader, model, criterion, optimizer, auc, device)

    print(f"Epoch: {epoch+1}/10, train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, auc: {val_acc:.4f}")
# load saved model

model = torch.load('model2.pth')



# perform validation

val_loss, val_acc = evaluate(validloader, model, criterion, optimizer, auc, device)



print(f"val_loss: {val_loss:.4f}, auc: {val_acc:.4f}")
test_df.head()
# prediction for test set for submission

def test(dataloader, model, device):

    model.eval()

    

    print('\n')

    print('Validation: \n')

    

    predictions = []

    

    for i, data in enumerate(dataloader):

        inputs, target = data['inputs'], data['targets']

        

        inputs = inputs.to(device)

        target = torch.tensor(target, dtype=torch.float)

        target = target.to(device)

        optimizer.zero_grad()

        

        output = model(inputs)

        

        predictions.append(output.detach().cpu().numpy())

    print('Prediciton complete!')

    return predictions



predictions = test(testloader, model, device)

len(predictions)
# flatten out the list of lists

pred_list = []

for pred in predictions:

    for p in pred:

        pred_list.append(p)



len(pred_list)
SUB_PATH = '/kaggle/input/cat-in-the-dat-ii/sample_submission.csv'

submission_df = pd.read_csv(SUB_PATH)

submission_df.head()
SUB_PATH = '/kaggle/input/cat-in-the-dat-ii/sample_submission.csv'

submission_df = pd.read_csv(SUB_PATH)

submission_df['target'] = pd.DataFrame(pred_list)

submission_df.head()
submission_df.to_csv('submission.csv', index=False)
pd.read_csv('submission.csv')