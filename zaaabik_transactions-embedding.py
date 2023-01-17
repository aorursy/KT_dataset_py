import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.manifold import TSNE
from joblib import dump, load
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, recall_score, precision_score, roc_curve, accuracy_score
import copy
from tqdm import tqdm
MAX_ROWS = 10000 * 30
SEQ_LENGTH = 50
# MAX_ROWS = 3000

number_of_small_groups = 172

data_path = '../input'

transaction_test = 'transactions_test.csv'
transaction_train = 'transactions_train.csv'
embedding = 'embeddings_for_groups_clean.csv'
target = 'train_target.csv'

transaction_train_path = os.path.join(data_path, transaction_train)
transaction_train_df = pd.read_csv(transaction_train_path, nrows=MAX_ROWS)

count_client_transactions = transaction_train_df.groupby('client_id').size()
small_transaction_count_clients = count_client_transactions[count_client_transactions < 500].index
transaction_train_df = transaction_train_df[~transaction_train_df['client_id'].isin(small_transaction_count_clients)]

print(len(np.unique(transaction_train_df.small_group)))
assert len(np.unique(transaction_train_df.small_group)) == number_of_small_groups
# transaction_train_df = pd.merge(
#     transaction_train_df, 
#     embedding_df,
#     left_on='small_group',
#     right_on='small_group_code',
#     how='left'
# ).drop(columns=['small_group_code', 'small_group_y', 'small_group_x'])
transaction_train_df = transaction_train_df.drop(columns=['trans_date'])
count_client_transactions = transaction_train_df.groupby('client_id').size()
small_transaction_count_clients = count_client_transactions[count_client_transactions < 500].index
transaction_train_df = transaction_train_df[~transaction_train_df['client_id'].isin(small_transaction_count_clients)]
columns_to_normalize = ['amount_rur']
scaler = MinMaxScaler()
scaled_df = scaler.fit_transform(transaction_train_df[columns_to_normalize])
transaction_train_df.loc[:, columns_to_normalize] = scaled_df
# transaction_train_df = pd.get_dummies(transaction_train_df, columns=['small_group']) 
# transaction_train_df_dummy.head()
dump(scaler, 'scaler.joblib')
client_ids = transaction_train_df.client_id.unique()
train_client_ids, validation_client_ids = train_test_split(client_ids)
validation_df = transaction_train_df[transaction_train_df.client_id.isin(validation_client_ids)]
transaction_train_df = transaction_train_df[transaction_train_df.client_id.isin(train_client_ids)]
print('train part', len(transaction_train_df))
print('test part', len(validation_df))
validation_df.head()
transaction_train_df['amount_rur'].describe()
avg_amount_transaction = transaction_train_df['client_id'].value_counts()
avg_amount_transaction.describe()
transaction_train_df.client_id.unique()        

class TripletTransactionDatasetFull(Dataset):
    def __init__(self, X, seq_len):
        unique_client_id = X.client_id.unique()        
        self.seq_dict = self.create_seq_dict(X, seq_len, unique_client_id)
        self.unique_client_id = np.array(list(self.seq_dict.keys()))
        self.triplets, self.client_ids = self.create_triplets(self.seq_dict, unique_client_id)
    
    def create_seq_dict(self, X, seq_len, unique_client_id):
        seq_dict = {}
        groups = X.groupby('client_id')
        for key in unique_client_id:
            cur_group = groups.get_group(key).drop(columns='client_id')
            n = len(cur_group)
            
            chunks = n // SEQ_LENGTH
            cur_group = cur_group[:chunks * SEQ_LENGTH]
            splitted_array = np.split(cur_group, chunks)
            seq_dict[key] = splitted_array
            
        return seq_dict
    
    def create_triplets(self, seq_dict, unique_client_id):
        triplets = []
        client_ids = []
        for positive_client_id in unique_client_id:            
            positive_group_len = seq_dict[positive_client_id]
            positive, anchor = np.random.choice(len(positive_group_len), 2, replace=False)
            
            neg_client_ids = unique_client_id[unique_client_id != positive_client_id]
            neg_client_id = np.random.choice(neg_client_ids, 1)[0]
            negative_group = seq_dict[neg_client_id]
            negative = np.random.choice(len(negative_group), 1)[0]

            triplets.append(
                (positive, anchor, negative)
            )
            client_ids.append(
                (positive_client_id, positive_client_id, neg_client_id)
            )
        return triplets, client_ids
    
    def __len__(self):
        return len(self.seq_dict)
    
    def __getitem__(self, idx):
        positive_client_id, anchor_client_id, neg_client_id = self.client_ids[idx]
        positive, anchor, negative = self.triplets[idx]        
        positive = self.seq_dict[positive_client_id][positive]        
        anchor = self.seq_dict[anchor_client_id][anchor]
        negative = self.seq_dict[neg_client_id][negative]
        
        positive_ammount, positive_small_group = positive['amount_rur'].values.reshape(-1,1), positive['small_group'].values.reshape(-1,1)
        anchor_ammount, anchor_small_group = anchor['amount_rur'].values.reshape(-1,1), anchor['small_group'].values.reshape(-1,1)
        negative_ammount, negative_small_group = negative['amount_rur'].values.reshape(-1,1), negative['small_group'].values.reshape(-1,1)
        
        return (positive_ammount, positive_small_group), (anchor_ammount, anchor_small_group), (negative_ammount, negative_small_group)

triplet_dataset = TripletTransactionDatasetFull(transaction_train_df, SEQ_LENGTH)
(positive_ammount, positive_small_group), (anchor_ammount, anchor_small_group), (negative_ammount, negative_small_group) = triplet_dataset[len(triplet_dataset) - 1]

positive_ammount.shape, positive_small_group.shape
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
# device = 'cpu'# 
device = torch.device("cpu")
if torch.cuda.is_available():
    print('CUDA!')
    device = torch.device("cuda:0")
class LSTM(nn.Module):
    def __init__(self, input_size, embedding_input, embedding_out, hidden_dim, batch_size, output_dim=500, num_layers=2, bidir=False):
        super(LSTM, self).__init__()
        self.input_dim = embedding_out + input_size
        self.bidir = bidir
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.dense1_bn = nn.BatchNorm1d(250)
        self.Em
        self.drop = nn.Dropout(0.4)
        self.hidden = None
        self.hidden = self.init_hidden()
        
        
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True, bidirectional=self.bidir)
        l1_size = self.hidden_dim
        if self.bidir:
            l1_size *= 2
        self.linear1 = nn.Linear(l1_size, 250)
        self.linear2 = nn.Linear(250, self.output_dim)
        self.linear3 = nn.Linear(250, 250)
        self.linear = nn.Linear(self.hidden_dim, self.output_dim)
        self.relu = nn.ReLU()        

    def init_hidden(self, batch_size=None):
        layers = self.num_layers
        if self.hidden:
            del self.hidden
        if not batch_size:
            batch_size = self.batch_size
        if self.bidir:
            layers *= 2 
        hidden_state = torch.zeros(layers, batch_size, self.hidden_dim, dtype=torch.double)
        cell_state = torch.zeros(layers, batch_size, self.hidden_dim, dtype=torch.double)
        return (hidden_state.float().to(device), cell_state.float().to(device))

    def forward(self, x):
        x, self.hidden = self.lstm(x, self.hidden)
        x = x[:,-1]
        x = self.linear1(x)
        x = self.relu(x)
        # x = self.dense1_bn(x)
        x = self.linear2(x)
        # return only output after passed all data
        # x = F.normalize(x, p=2, dim=1)
        return x
class TripletLSTM(nn.Module):
    def __init__(self, lstm):
        super(LSTM, self).__init__()
        self.lstm = lstm
        
    def forward(self, x1, x2, x3):
        output1 = self.lstm(x1)
        self.lstm.hidden = self.lstm.init_hidden()
        
        output2 = self.lstm(x2)
        self.lstm.hidden = self.lstm.init_hidden()
        
        self.lstm.hidden = self.lstm.init_hidden()
        output3 = self.lstm(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.lstm(x)
def triplet_loss(anchor, positive, negative):
    margin = 0.5
    distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
    distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
    losses = F.relu(distance_positive - distance_negative + margin)
    return losses.mean()
batch_size = 32
hidden_dim = 150
num_layers = 1
output_dim = 32

epochs = 10
lr = 1e-3

train_dataset = TripletTransactionDatasetFull(transaction_train_df, SEQ_LENGTH)
test_dataset = TripletTransactionDatasetFull(validation_df, SEQ_LENGTH)

train_dataloader = DataLoader(train_dataset, batch_size)
test_dataloader = DataLoader(test_dataset, batch_size)
print('Train batches count', len(train_dataloader))
print(' Test batches count', len(test_dataloader))
for p,a,n in train_dataloader:
    pass

for p,a,n in test_dataloader:
    break

p[0].shape
p[1].unique()
l = nn.Embedding(number_of_small_groups, 128)
# nn.Embedding()
output = l(p[1].long()).squeeze(2)
output = torch.cat((p[0].float(), output.float()), dim=-1)
# torch.cat((p[0][0], ))
# print(p[0].shape, output.shape)
# print(.shape)
target = torch.zeros_like(output)
target = target[:, -1, :]
output = output[:, -1, :]
# print(target.float(), output.float())
criterion = nn.BCEWithLogitsLoss()
loss = criterion(output, target)
loss.backward()


# loss = criterion(output, target)
torch.where(torch.norm(l.weight.grad, dim=1) != 0)
output
for p,a,n in train_dataloader:
    pass

# for p,a,n in test_dataloader:
#     pass
    # print(p.shape, a.shape, n.shape)
def valudate(model, dataloader):
    losses = []
    
    
def validate(model, val_ds):    
    losses = []
    positive_dist = []
    negative_dist = []
    all_predicts = np.empty(0)
    all_true_labels = np.empty(0)    
    model.eval()
    batches = range(len(val_ds))    
    for batch in batches:
        pos, anch, neg = val_ds.get_item(batch)
        pos, anch, neg = pos.to(device), anch.to(device), neg.to(device)
        
        pos_pred, anch_pred, neg_pred = predict_triplet(model, pos, anch, neg)
        loss_val = triplet_loss(pos_pred, anch_pred, neg_pred)
        pos_dist = get_disance(pos_pred, anch_pred)
        neg_dist = get_disance(anch_pred, neg_pred)
        
        cur_positive_dist = pos_dist.data.cpu().numpy()
        cur_negative_dist = neg_dist.data.cpu().numpy()
                
        positive_dist.append(np.mean(cur_positive_dist))
        negative_dist.append(np.mean(cur_negative_dist))
        
        loss = loss_val.mean()
        loss = loss.cpu().detach().numpy()
        losses.append(loss)
    
    return np.mean(losses), np.mean(positive_dist), np.mean(negative_dist)

def get_disance(output1, output2):
    distances = (output2 - output1).pow(2).sum(1)
    return distances

def triplet_loss(anchor, positive, negative):
    margin = 0.5
    distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
    distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
    losses = F.relu(distance_positive - distance_negative + margin)
    return losses.mean()

def get_loss(output1, output2, target):
    margin = 0.5
    distances = get_disance(output1, output2)  # squared distances
    losses = 0.5 * (target.float() * distances +
                    (1 + -1 * target).float() * F.relu(margin - (distances + 1e-5).sqrt()).pow(2))
    return losses

# def contrastive_loss(output, target):
#     margin = 0.1
#     output = target * output + (1 - target) * F.relu(output - margin)
#     bce = nn.BCELoss()
#     loss = bce(output, target)
#     return loss
    

def train(model, train_ds, test_ds, epochs, lr):
    optimizer = optim.Adam(model.parameters(), lr)
    
    best_model_state_dicts = []
    best_model_state_dicts.append(copy.deepcopy(model.state_dict()))

    train_losses = []
    validate_losses = []
    pos_distanses = []
    neg_distanses = []
    
    ## initial score
#     validate_loss, pos_distanse, neg_distanse = validate(model, test_ds)
#     validate_losses.append(validate_loss)
#     pos_distanses.append(pos_distanse)
#     neg_distanses.append(neg_distanse)
#     print(f'Without train loss:{validate_loss:.4f} pos dist:{pos_distanse:.4f} neg dist:{neg_distanse:.4f}')
    
#     train_loss, pos_distanse, neg_distanse  = validate(model, train_ds)
#     train_losses.append(train_loss)
    batches = range(len(train_ds))
    for epoch in range(epochs):
        model.train()
        train_ds.shuffle()
        for batch in tqdm(batches):
            model.zero_grad()
            pos, anch, neg = train_ds.get_item(batch)
            pos, anch, neg = pos.to(device), anch.to(device), neg.to(device)
            
            pos_pred, anch_pred, neg_pred = predict_triplet(model, pos, anch, neg)
                   
            pos_dist = get_disance(pos_pred, anch_pred)
            neg_dist = get_disance(anch_pred, neg_pred)
            
            # triplet_loss
            loss = triplet_loss(pos_pred, anch_pred, neg_pred)
            loss.backward()
            optimizer.step()
            
        validate_loss, pos_distanse, neg_distanse = validate(model, test_ds)
        validate_losses.append(validate_loss)
        pos_distanses.append(pos_distanse)
        neg_distanses.append(neg_distanse)
        
        train_loss, train_pos_distanse, train_neg_distanse = validate(model, train_ds)
        train_losses.append(train_loss)
        
        best_model_state_dicts.append(copy.deepcopy(model.state_dict()))
        
        print(f'Epoch {epoch} train loss: {train_loss:.4f} test loss:{validate_loss:.4f} pos dist:{pos_distanse:.4f} neg_dist:{neg_distanse:.4f}')
    return train_losses, validate_losses, pos_distanses, neg_distanses, best_model_state_dicts

torch.cuda.empty_cache()

input_dim = pos.shape[-1]
batch_size = 256
hidden_dim = 150
num_layers = 1
output_dim = 32

epochs = 10
lr = 1e-3

train_ds = TripletTransactionDatasetFull(transaction_train_df, batch_size)
test_ds = TripletTransactionDatasetFull(validation_df, batch_size)
print('Train batches count', len(train_ds))
print(' Test batches count', len(test_ds))

model = LSTM(input_dim, hidden_dim, batch_size=batch_size, output_dim=output_dim, num_layers=num_layers)
model.to(device)
train_loss, test_loss, pos_distanses, neg_distanses, state_dicts = train(model, train_ds, test_ds, epochs, lr)
plt.figure(figsize=(15, 5))
plot_x = range(0, epochs + 1)
plt.plot(plot_x, train_loss, label='Train loss')
plt.plot(plot_x, test_loss, label='Test loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid()
plt.title('Loss')
plt.legend();
plt.savefig('Loss')
data = {
    'epoch': plot_x,
    'train_loss': train_loss,
    'test_loss': test_loss
}
loss_df = pd.DataFrame(data=data)
loss_df.to_csv('loss.csv', index='epoch')
plt.figure(figsize=(15,5))
plot_x = range(0, epochs + 1)
plt.plot(plot_x, pos_distanses, label='Similarity between positive')
plt.plot(plot_x, neg_distanses, label='Similarity between negative')
plt.xlabel('Epoch')
plt.ylabel('Cosine similarity')
plt.title('Similarity on validation')
plt.grid()
plt.legend();
data = {
    'epoch': plot_x,
    'mean_pos_distanses': pos_distanses,
    'mean_neg_distanses': neg_distanses
}
simularity_df = pd.DataFrame(data=data)
simularity_df.to_csv('simularity.csv', index='epoch')
plt.figure(figsize=(15,5))
plot_x = range(0, epochs + 1)
plt.plot(plot_x, rocs, label='ROC AUC')
plt.xlabel('Epoch')
plt.ylabel('ROC AUC')
plt.title('ROC AUC')
plt.grid()
plt.legend();
plt.savefig('ROC AUC')
best_roc = np.min(test_loss)
best_idx = np.argmin(test_loss)
best_model = LSTM(input_dim, hidden_dim, batch_size=batch_size, 
                  output_dim=output_dim, num_layers=num_layers)

best_model.load_state_dict(state_dicts[best_idx])
# torch.save(best_model.state_dict(), f'model_state_dict ROC: {best_roc:.4f}')
# torch.save(best_model, f'model ROC: {best_roc:.4f}')
best_model.to(device);
plt.title(f'ROC curve AUC:{roc_auc:.3f}')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.plot(fpr, tpr);
plt.savefig('ROC curve')
plt.title(f'PR curve AUC:{pr_auc:.3f}')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.plot(recall, precision);
plt.savefig('PR curve')
# validation_ds = TripletTransactionDataset(test_d, batch_size)
# validation_batches = range(len(validation_ds))
print('Test')
validation_loss, mean_pos_sim, mean_neg_sim = validate(best_model, test_ds)

print('Loss on validation', validation_loss)
print('Mean similarity between positive', mean_pos_sim)
print('Mean similarity between negative', mean_neg_sim)
print('Train')
validation_loss, mean_pos_sim, mean_neg_sim = validate(model, train_ds)

print('Loss on validation', validation_loss)
print('Mean similarity between positive', mean_pos_sim)
print('Mean similarity between negative', mean_neg_sim)
from sklearn.decomposition import PCA
from umap import UMAP
def plot_vis(best_model, df, dim_reduction=False):
    vis_group_df = df.groupby('client_id')
    res = {}
    res = pd.DataFrame()
    for client_id in list(vis_group_df.groups):
        client_trans = vis_group_df.get_group(client_id)
        n = len(client_trans) 
        need_to_remove = n % SEQ_LENGTH
        if need_to_remove != 0:
            client_trans = client_trans[:-need_to_remove]    
        client_trans = np.array_split(client_trans.drop(columns=['client_id']).values, n // SEQ_LENGTH)
        batch_size = np.array(client_trans).shape[0]
        
        best_model.hidden = best_model.init_hidden(batch_size)
        client_trans = torch.Tensor(client_trans).to(device)
        
        preds = best_model(client_trans)
        preds = preds.data.cpu().numpy()
        # res[client_id] = preds
        
        tmp_df = pd.DataFrame(preds)
        tmp_df['client_id'] = f'{client_id}'
        res = pd.concat((res, tmp_df), axis=0)
    
    # n_neighbors = 8
    for n_neighbors in [8]:# range(2, 25):
        if dim_reduction:
            # vis = TSNE(2, perplexity=i).fit_transform(res.iloc[:, :-1])
            vis = UMAP(init='random', n_neighbors=n_neighbors, min_dist=4.5, spread=5).fit_transform(res.iloc[:, :-1])
            # vis = PCA(2).fit_transform(res.iloc[:, :-1])        
            vis = pd.DataFrame(vis)
        else:
            vis = res.iloc[:, :-1]
        vis['client_id'] = res['client_id'].values
        plt.figure(figsize=(25,10))
        s = 0

        clients_count = len(vis_group_df.groups)

        total_pos_dist = 0
        total_neg_dist = 0
        for client_id in list(vis_group_df.groups):
            cur_client_trans = vis[vis['client_id'] == f'{client_id}'].drop(columns=['client_id'])
            all_other = vis[vis['client_id'] != f'{client_id}'].drop(columns=['client_id'])
            x,y = cur_client_trans.iloc[:,0], cur_client_trans.iloc[:,1]
            plt.scatter(x,y, label=client_id, alpha=0.7)

            neg_dist = euclidean_distances(cur_client_trans, all_other)
            pos_dist = euclidean_distances(cur_client_trans, cur_client_trans)
            total_neg_dist += neg_dist.mean() / clients_count
            total_pos_dist += pos_dist.mean() / clients_count
        plt.title(f'UMAP neigbors n_neighbors = {n_neighbors}')
        # plt.legend()
        plt.savefig(f'UMAP_triplet_best')
        print('pos: ', total_pos_dist)
        print('neg: ', total_neg_dist)
        
best_model.eval()
# plot_vis(best_model, transaction_train_df.iloc[:25_000], True)
plot_vis(best_model, validation_df, True)
# mkdir umap
# !ls
# !rm UMAP_tr*.png
def test_dist(best_model, df):
    vis_group_df = df.groupby('client_id')
    res = {}
    res = pd.DataFrame()
    for client_id in list(vis_group_df.groups):
        client_trans = vis_group_df.get_group(client_id)
        n = len(client_trans) 
        need_to_remove = n % SEQ_LENGTH
        if need_to_remove != 0:
            client_trans = client_trans[:-need_to_remove]    
        client_trans = np.array_split(client_trans.drop(columns=['client_id']).values, n // SEQ_LENGTH)
        batch_size = np.array(client_trans).shape[0]
        
        best_model.hidden = best_model.init_hidden(batch_size)
        client_trans = torch.Tensor(client_trans).to(device)
        
        preds = best_model(client_trans)
        preds = preds.data.cpu().numpy()
        # res[client_id] = preds
        
        tmp_df = pd.DataFrame(preds)
        tmp_df['client_id'] = f'{client_id}'
        res = pd.concat((res, tmp_df), axis=0)
        
    
    vis = res.iloc[:, :-1]
    vis['client_id'] = res['client_id']    
    
    clients_count = len(vis_group_df.groups)
    total_pos_dist = 0
    total_neg_dist = 0
    preds = []
    labels = []
    for client_id in list(vis_group_df.groups):
        cur_client_trans = vis[vis['client_id'] == f'{client_id}'].drop(columns=['client_id'])
        all_other = vis[vis['client_id'] != f'{client_id}'].drop(columns=['client_id'])
        neg_dist = euclidean_distances(cur_client_trans, all_other)
        flat_neg_dist = neg_dist.flatten()
        preds.extend(flat_neg_dist)
        labels.extend(np.ones(len(flat_neg_dist)))
        
        pos_dist = euclidean_distances(cur_client_trans, cur_client_trans)
        pos_dist = pos_dist[~np.eye(pos_dist.shape[0],dtype=bool)].reshape(pos_dist.shape[0],-1)
        flat_pos_dist = pos_dist.flatten()
        preds.extend(flat_pos_dist)
        labels.extend(np.zeros(len(flat_pos_dist)))
        
        total_neg_dist += neg_dist.mean() / clients_count
        total_pos_dist += pos_dist.mean() / clients_count
    
    roc = roc_auc_score(labels, preds)
    print(f'ROC {roc}')
    print('pos: ', total_pos_dist)
    print('neg: ', total_neg_dist)
    return labels, preds
        
model.eval()
labels, preds = test_dist(best_model, validation_df);
# labels, preds = test_dist(best_model, transaction_train_df);
# transaction_train_df,
# train_ds.get_item(0)[0]
# labels, preds = test_dist(model, validation_df);
sum(labels)
np.array(labels).shape, np.array(preds).shape
pred_df = pd.DataFrame({
    'label' : labels,
    'pred' : preds
})
# transaction_train_df.describe()
pred_df.to_csv('preds_labels.csv', index=None)
from sklearn.manifold import TSNE
res = {}
res = pd.DataFrame()
for client_id in list(vis_group_df.groups):
    client_trans = vis_group_df.get_group(client_id)
    n = len(client_trans) 
    need_to_remove = n % SEQ_LENGTH
    if need_to_remove != 0:
        client_trans = client_trans[:-need_to_remove]    
    client_trans = np.array_split(client_trans.drop(columns=['client_id']).values, n // SEQ_LENGTH)
    batch_size = np.array(client_trans).shape[0]
    best_model.hidden = best_model.init_hidden(batch_size)
    client_trans = torch.Tensor(client_trans).to(device)
    preds = best_model(client_trans)
    preds = preds.data.cpu().numpy()
    # res[client_id] = preds
    tmp_df = pd.DataFrame(preds)
    tmp_df['client_id'] = f'{client_id}'
    
    res = pd.concat((res, tmp_df), axis=0)
    
    vis = TSNE(2).fit_transform(res.iloc[:, :-1])
    
    for client_id in list(vis_group_df.groups)[1:4]:
        idxs = res[res['client_id'] == f'{client_id}'].index    
        x,y = vis[idxs,0], vis[idxs,1]
        plt.scatter(x,y, label=client_id)
    plt.legend()
    
# best_model.hidden = best_model.init_hidden(10000)
# input_x = vis_df.drop(columns=['client_id'])
# input_x = torch.Tensor(input_x.values)
# vis_pred = best_model(input_x)
