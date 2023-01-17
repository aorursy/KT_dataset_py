import pandas as pd
import numpy as np

path = '../input/stanford-covid-vaccine/'
bbps_folder = f'{path}bpps/'
train = pd.read_json(f'{path}train.json', lines=True)
test = pd.read_json(f'{path}test.json', lines=True)
sample_sub = pd.read_csv(f'{path}sample_submission.csv')
targets = np.stack([train[col].to_list() for col in ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C','deg_pH10', 'deg_50C']], axis=-1)
filter_1 = pd.Series((targets > -0.5).all(axis=(1,2))).astype(int)
filter_2 = (train['signal_to_noise'] > 1).astype(int)
reactivity_bins = pd.qcut(train['reactivity'].map(np.mean), 5, labels=False)
deg_Mg_pH10_bins = pd.qcut(train['deg_Mg_pH10'].map(np.mean), 5, labels=False)
deg_Mg_50C_bins = pd.qcut(train['deg_Mg_50C'].map(np.mean), 5, labels=False)

stratify = pd.concat([filter_1, filter_2, reactivity_bins, deg_Mg_50C_bins, deg_Mg_pH10_bins], axis=1)
stratify = stratify.astype(str).apply(''.join, axis=1)
stratify.value_counts()
from sklearn.model_selection import StratifiedKFold

skf= StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
folds = list(skf.split(train, stratify))
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch import LongTensor, FloatTensor
import torch
import numpy as np
import pandas as pd
from copy import deepcopy


def get_structure_matrix(structure):
    n = len(structure)
    structure_matrix = np.zeros((n,n), int)
    for i in range(n):
        if structure[i] == '(':
            count = 0
            for j in range(i, n):
                if structure[j] == '(':
                    count += 1
                elif structure[j] == ')':
                    count -= 1
                if count==0:
                    structure_matrix[i,j]=1
                    structure_matrix[j,i]=1
                    break
    return structure_matrix.astype(int)

def get_stracture_bpps(structure_matrix, bbps):
    stracture_bbps = np.zeros(structure_matrix.shape[:-1])
    stracture_bbps[structure_matrix.any(axis=0)] = bbps[structure_matrix.astype(bool)]
    return stracture_bbps

def get_distance2pair(structure_matrix):
    self_idx = np.arange(structure_matrix.shape[0])
    idx_pair = structure_matrix.argmax(axis=1)
    idx_pair[~structure_matrix.any(axis=1)] = self_idx[~structure_matrix.any(axis=1)]
    distance2pair = (idx_pair - self_idx)/(structure_matrix.shape[0]-1)
    return distance2pair


def proc_df(df):
    df = deepcopy(df)
    df['bbps'] = df['id'].apply(lambda x: np.load(f'{bbps_folder}{x}.npy'))
    df['bbps_sum'] = df['bbps'].apply(lambda x: x.sum(axis=1))
    df['bbps_max'] = df['bbps'].apply(lambda x: x.max(axis=1))
    df['structure_matrix'] = df['structure'].apply(get_structure_matrix)
    df['structure_bpps'] = df.apply(lambda x: get_stracture_bpps(x['structure_matrix'], x['bbps']), axis=1)
    df['distance2pair'] = df['structure_matrix'].apply(get_distance2pair)
    return df



STRUCTURE_CODE = {'(': 0, '.': 1, ')': 2}
PREDICTED_LOOP_TYPE_CODE = {'H': 0, 'E': 1, 'B': 2, 'M': 3, 'X': 4, 'S': 5, 'I': 6}
SEQUANCE_CODE = {'U': 0, 'C': 1, 'A': 2, 'G': 3}

ERROR_COLUMNS = ['reactivity_error', 'deg_error_Mg_pH10', 'deg_error_Mg_50C', 'deg_error_pH10', 'deg_error_50C']
TARGET_COLUMNS = ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C', 'deg_pH10', 'deg_50C']


class CompDataset(Dataset):
    def __init__(self, data, bbps_folder, device='cpu', train_mode=True):
        data = proc_df(data)
        
        self.data = {}
        self.data['sequence'] = LongTensor(np.array([[SEQUANCE_CODE[t] for t in s] for s in data['sequence']]))
        self.data['structure'] = LongTensor(np.array([[STRUCTURE_CODE[t] for t in s] for s in data['structure']]))
        self.data['predicted_loop_type'] = LongTensor(np.array([[PREDICTED_LOOP_TYPE_CODE[t] for t in lt] for lt in data['predicted_loop_type']]))
        self.data['bbps'] = FloatTensor(np.array(data['bbps'].to_list()))
        self.data['structure_matrix'] = LongTensor(np.array(data['structure_matrix'].to_list()))
        
        
        bbps_sum = FloatTensor(np.array(data['bbps_sum'].to_list()))
        bbps_max = FloatTensor(np.array(data['bbps_max'].to_list()))
        structure_bpps = FloatTensor(np.array(data['structure_bpps'].to_list()))
        distance2pair = FloatTensor(np.array(data['distance2pair'].to_list()))
        self.data['features'] = torch.stack([bbps_sum, bbps_max, structure_bpps, distance2pair], dim=1)
        
        
        
        if train_mode:
            error = np.stack([data[col].tolist() for col in ERROR_COLUMNS], axis=1).swapaxes(1,2)
            target = np.stack([data[col].tolist() for col in TARGET_COLUMNS], axis=1).swapaxes(1,2)
            self.data['targets'] = torch.cat([FloatTensor(target), torch.zeros((target.shape[0],39,5))], dim=1)#FloatTensor(target)
            self.data['error'] = torch.cat([FloatTensor(error), torch.ones((target.shape[0],39,5))*5], dim=1)#FloatTensor(error)
            self.data['signal_to_noise'] = FloatTensor(data['signal_to_noise'].to_numpy())
            self.data['sn_filter'] = LongTensor(data['SN_filter'].to_numpy())
            self.data['filter_1'] = LongTensor((target.min(-1) > -0.5).all(-1))
            self.data['filter_2'] = LongTensor(target.mean((1,2))/error.mean((1,2)) > 1)
            
        for key in self.data.keys():
            self.data[key] = self.data[key].to(device)
    
        
    def __len__(self):
        return self.data['sequence'].shape[0]
    
    def __getitem__(self, idx: int):
        return {k: v[idx] for k,v in self.data.items()}
WSMin = 48
def random_window_collate_fn(batch):
    seq_len = batch[0]['sequence'].shape[0]
    windowed_batch = []
    window_size = np.random.randint(WSMin, seq_len)
    for el in batch:
        window_left = np.random.randint(seq_len-window_size)
        window_right = window_left+window_size
        el['features'] = el['features'][:, window_left:window_right]
        for k in ['sequence', 'structure', 'predicted_loop_type', 'targets', 'error']:
            if k in el.keys():
                el[k] = el[k][window_left:window_right]
        for k in ['bbps', 'structure_matrix']:
            el[k] = el[k][window_left:window_right, window_left:window_right]
        windowed_batch.append(el)
    new_batch = {}
    for k in batch[0].keys():
        new_batch[k] = torch.stack([el[k] for el in windowed_batch])
    return new_batch
import torch
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F


MEDIAN_ERROR = 0.105


class CompLoss(_Loss):
    def forward(self, logits, targets, error, filter_1, filter_2):
        logits, targets, error = logits[:,:,:3], targets[:,:,:3], error[:,:,:3]
        weights = MEDIAN_ERROR/(MEDIAN_ERROR + error)
        
        loss = (logits - targets)**2
        loss = loss*weights
        loss = loss.mean(1)
        loss = torch.sqrt(loss) 
        loss = loss.mean(0)
        loss = loss.mean(0)
        return loss
    
        
def comp_metric_fn(logits, targets, error, filter_1, filter_2):
    logits, targets, error = logits[:,:,:3], targets[:,:,:3], error[:,:,:3]
    logits, targets, error = logits[(filter_1==1)&(filter_2==1)], targets[(filter_1==1)&(filter_2==1)], error[(filter_1==1)&(filter_2==1)]
    weights = (error < 5).float()

    loss = (logits - targets)**2
    loss = loss.sum(1)/weights.sum(1)
    loss = loss.mean(0)
    loss = torch.sqrt(loss) 
    loss = loss.mean(0)
    return loss.item()


class PreTrainLoss(_Loss):
    def forward(self, logits, sequence):
        logits = logits.reshape(-1,4)
        sequence = sequence.reshape(-1)
        return F.cross_entropy(logits, sequence)
import torch

from catalyst.dl import Callback, CallbackOrder


class LoaderMetricCallback(Callback):
    def __init__(self, input_key, output_key, metric_fn, prefix="metric"):
        super().__init__(CallbackOrder.Metric)
        self.input_key = input_key
        self.output_key = output_key
        self.metric_fn = metric_fn
        self.prefix = prefix
        self.data = {k: [] for k in self.output_key+self.input_key}
        
    def on_batch_end(self, state):
        y_hat = state.output['logits'].detach().cpu().numpy()
        y = state.input['targets'].detach().cpu().numpy()
        for k in self.output_key:
            self.data[k].append(state.output[k].detach().cpu())
        for k in self.input_key:
            self.data[k].append(state.input[k].detach().cpu())
            
    def on_loader_end(self, state):
        if state.is_valid_loader:
            for k in self.output_key+self.input_key:
                self.data[k] = torch.cat(self.data[k])

            state.loader_metrics[self.prefix] = self.metric_fn(**self.data)

        self.data = {k: [] for k in self.output_key+self.input_key}
import torch
from torch import nn
import torch.nn.functional as F


class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))
    
    
class Conv1dStack(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, padding=1, dilation=1):
        super(Conv1dStack, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, out_dim, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm1d(out_dim),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
        )
        self.res = nn.Sequential(
            nn.Conv1d(out_dim, out_dim, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm1d(out_dim),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        x = self.conv(x)
        h = self.res(x)
        return x + h

    
class SeqEncoder(nn.Module):
    def __init__(self, in_dim: int):
        super(SeqEncoder, self).__init__()
        self.conv0 = Conv1dStack(in_dim, 128, 3, padding=1)
        self.conv1 = Conv1dStack(128, 64, 6, padding=5, dilation=2)
        self.conv2 = Conv1dStack(64, 32, 15, padding=7, dilation=1)
        self.conv3 = Conv1dStack(32, 32, 30, padding=29, dilation=2)

    def forward(self, x):
        x1 = self.conv0(x)
        x2 = self.conv1(x1)
        x3 = self.conv2(x2)
        x4 = self.conv3(x3)
        x = torch.cat([x1, x2, x3, x4], dim=1)
        return x

    
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pretrain_mode = False
        prenet_hidden = 128
        seq_hidden_size = 1024
        hidden_cls_dim = 128
        feature_prenet_hidden = 32
        n_feature = 4
        
        self.sequence_embed = nn.Embedding(4, 8)
        self.predicted_loop_type_embed = nn.Embedding(7, 16)
        self.structure_embed = nn.Embedding(4, 8)
        
        self.feature_prenet = nn.Sequential(
                nn.Conv1d(n_feature, feature_prenet_hidden, kernel_size=1),
                nn.BatchNorm1d(feature_prenet_hidden),
            )

        self.prenet = nn.Sequential(
            nn.Conv1d((8+16+8) * 2 + 1 + feature_prenet_hidden*2, prenet_hidden, kernel_size=1),
            nn.BatchNorm1d(prenet_hidden),
        )
        self.seq_encoder = SeqEncoder(prenet_hidden)

        self.reccurent_layers = nn.ModuleList()
        self.dense_layers = nn.ModuleList()
        for idx in range(2):
            self.reccurent_layers.append(nn.GRU(
                input_size= 256 if idx==0 else seq_hidden_size*2,
                hidden_size=seq_hidden_size,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            ))
            
            self.dense_layers.append(nn.Sequential(
                nn.Conv1d(seq_hidden_size*2, seq_hidden_size, kernel_size=1),
                nn.BatchNorm1d(seq_hidden_size),
                Mish(),
                nn.Dropout(0.3)
            ))
        
        self.classifier = nn.Sequential(
                nn.Linear(seq_hidden_size*2, hidden_cls_dim), Mish(), nn.Dropout(p=0.3),
                nn.Linear(hidden_cls_dim, hidden_cls_dim), Mish(), nn.Dropout(p=0.2),
                nn.Linear(hidden_cls_dim, 5)
            )
        self.pretrain = nn.Sequential(
                nn.Linear(seq_hidden_size*2, hidden_cls_dim), Mish(), nn.Dropout(p=0.3),
                nn.Linear(hidden_cls_dim, hidden_cls_dim), Mish(), nn.Dropout(p=0.2),
                nn.Linear(hidden_cls_dim, 4)#, nn.Softmax()
            )

    def _attention_head(self, x, bbps):
        return torch.bmm(bbps, x)
    

    def forward(self, sequence, predicted_loop_type, structure, bbps, features):
        sequence = self.sequence_embed(sequence)
        predicted_loop_type = self.predicted_loop_type_embed(predicted_loop_type)
        structure = self.structure_embed(structure)
        
        features = self.feature_prenet(features).permute(0,2,1)
        
        x = torch.cat([sequence, predicted_loop_type, structure, features], dim=-1)
        
        if self.pretrain_mode:
            n_elements_to_mask = 3
            for i in range(x.shape[0]):
                mask = list(np.random.randint(x.shape[1], size=n_elements_to_mask))
                x[i, mask]=0
            
        x_sec = self._attention_head(x, bbps)
        attention_sum_f = bbps.sum(-1)[:,:,None]
        x = torch.cat([x, x_sec, attention_sum_f], dim=-1)
    
        x = self.prenet(x.permute(0,2,1)).permute(0,2,1)
        x = self.seq_encoder(x.permute(0,2,1)).permute(0,2,1)

        for idx, (reccurent_layer, linear_layer) in enumerate(zip(self.reccurent_layers, self.dense_layers)):
            x, _ = reccurent_layer(x)
            x = linear_layer(x.permute(0,2,1)).permute(0,2,1)
            x = torch.cat([x, self._attention_head(x, bbps)], axis=-1)
        
        if self.pretrain_mode:
            out = self.pretrain(x)
        else:
            out = self.classifier(x)
        return out
import os
from os import listdir


def remove_useless_checkpoints(path, usefull_checkpoint):
    for el in listdir(path):
        if el != usefull_checkpoint:
            print(f'rm {path}/{el}')
            !rm '{path}/{el}'
import torch
from torch.utils.data import DataLoader
from catalyst.dl import SupervisedRunner, Runner
from catalyst.contrib.nn.optimizers import RAdam, Lookahead, Adam
from catalyst.contrib.nn.criterion import CrossEntropyLoss
from catalyst.dl.callbacks import OptimizerCallback, CriterionCallback


def dataloader_train_init(data, device='cuda'):
    dataset = CompDataset(data, bbps_folder, device=device, train_mode=False)
    return DataLoader(dataset, 16, shuffle=True, drop_last=True, collate_fn=random_window_collate_fn)


device = 'cuda'

dataloader_107 = dataloader_train_init(train.append(test[test['seq_length']==107]), device)
dataloader_130 = dataloader_train_init(test[test['seq_length']==130], device)

model = Model()
model.pretrain_mode = True

optimizer =  Lookahead(RAdam(model.parameters(), lr=1e-3))

criterion = {"loss": PreTrainLoss()}

callbacks =[CriterionCallback(input_key='sequence', output_key='logits', prefix="loss", criterion_key="loss"),
            OptimizerCallback(metric_key="loss", accumulation_steps=1),]

input_keys = ['sequence', 'predicted_loop_type', 'structure', 'bbps', 'features']
input_citerion_keys = ['sequence']
runner = SupervisedRunner(device=device, input_key=input_keys, input_target_key=input_citerion_keys)
for i in range(5):
    runner.train(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            loaders={'train': dataloader_107},
            logdir='pretrain',
            callbacks=callbacks,
            num_epochs=1,
            verbose= False,
            main_metric="loss",
            minimize_metric=True,
        )      
    runner.train(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            loaders={'train': dataloader_130},
            logdir='pretrain',
            callbacks=callbacks,
            num_epochs=1,
            verbose=False,
            main_metric="loss",
            minimize_metric=True,
        )
    
remove_useless_checkpoints('pretrain/checkpoints', 'last.pth')
import torch
from torch.utils.data import DataLoader
from catalyst.dl import SupervisedRunner, Runner
from catalyst.contrib.nn.optimizers import RAdam, Lookahead, Adam
from catalyst.dl.callbacks import OptimizerCallback, EarlyStoppingCallback, SchedulerCallback, CriterionCallback, MetricAggregationCallback, MetricManagerCallback

device = 'cuda'
input_keys = ['sequence', 'predicted_loop_type', 'structure', 'bbps', 'features']
input_citerion_keys = ['targets', 'error', 'filter_1', 'filter_2']
batch_size = 16

def dataloader_train_init(idx):
    dataset = CompDataset(train.iloc[idx], bbps_folder, device=device)
    return DataLoader(dataset, batch_size, shuffle=True, drop_last=True, collate_fn=random_window_collate_fn)

def dataloader_valid_init(idx):
    dataset = CompDataset(train.iloc[idx], bbps_folder, device=device)
    return DataLoader(dataset, batch_size, shuffle=False, drop_last=False)

def model_init():
    model = Model()
    model.load_state_dict(torch.load(f'pretrain/checkpoints/last.pth')['model_state_dict'])
    return model

optimizer_init = lambda model : Lookahead(RAdam(model.parameters(), lr=1e-3))
scheduler_init = lambda optimizer : torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.25, patience=4, cooldown=5, min_lr=1e-6)
loss_init = lambda : CompLoss()
callbacks_init = lambda : [CriterionCallback(input_key=input_citerion_keys, output_key=['logits'], prefix="loss", criterion_key="loss"),
                            LoaderMetricCallback(input_key=input_citerion_keys, output_key=['logits'], prefix="metric", metric_fn=comp_metric_fn),
                            OptimizerCallback(metric_key="loss", accumulation_steps=1),
                            EarlyStoppingCallback(patience=15, metric='metric', minimize=True),
                            SchedulerCallback(mode='epoch')]
for i, (train_idx, val_idx) in enumerate(folds):
    model = model_init()
    optimizer = optimizer_init(model)
    scheduler = scheduler_init(optimizer)
    criterion = {"loss": loss_init()}
    loaders = {'train': dataloader_train_init(train_idx), 'valid': dataloader_valid_init(val_idx)}
    callbacks = callbacks_init()
    runner = SupervisedRunner(device=device, input_key=input_keys, input_target_key=input_citerion_keys)
    runner.train(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            loaders=loaders,
            logdir=f'fold_{i}',
            callbacks=callbacks,
            num_epochs=9999,
            verbose=True,
            load_best_on_end=True,
            main_metric="metric",
            minimize_metric=True,
        )  
    remove_useless_checkpoints(f'fold_{i}/checkpoints', 'best.pth')
from torch import FloatTensor

cv_scores = []
oof_prediction = torch.zeros((train.shape[0], 107, 5))

for i, (train_idx, val_idx) in enumerate(folds):
    best_model_path = f'fold_{i}/checkpoints/best.pth'
    runner = SupervisedRunner(device='cuda', input_key=input_keys, input_target_key=input_citerion_keys)
    dataloader = dataloader_valid_init(val_idx)
    
    prediction = runner.predict_loader(loader=dataloader, model=Model(), resume=best_model_path)
    prediction = torch.cat([b['logits'] for b in prediction])
    fold_score = comp_metric_fn(prediction, *[dataloader.dataset.data[el] for el in input_citerion_keys])
    cv_scores.append(fold_score)
    
    oof_prediction[val_idx] = prediction.cpu()#.numpy()

cv_scores = np.array(cv_scores)
train_data = CompDataset(train, bbps_folder).data
off_score = comp_metric_fn(oof_prediction, *[train_data[el] for el in input_citerion_keys])

np.save("oof_prediction", oof_prediction[:,:,[0,1,3,2,4]])

final_log = f'Fold scores: {list(cv_scores)}\nOFF score: {off_score}\nFold mean: {cv_scores.mean()}\nFold std : {cv_scores.std()}'
with open(f"final_log.txt", 'w') as file:
    file.write(final_log)
print(final_log)
test_107_pred = []
test_130_pred = []
for i, _ in enumerate(folds):
    best_model_path = f'fold_{i}/checkpoints/best.pth'
    runner = SupervisedRunner(device='cuda', input_key=input_keys, input_target_key=input_citerion_keys)
    
    dataset = CompDataset(test[test['seq_length']==107], bbps_folder, device='cuda', train_mode=False)
    dataloader = DataLoader(dataset, batch_size,)
    
    
    prediction = runner.predict_loader(loader=dataloader, model=model_init(), resume=best_model_path)
    prediction = np.concatenate([b['logits'].cpu().numpy() for b in prediction])[:,:,[0,1,3,2,4]]
    test_107_pred.append(prediction)
    
    
    dataset = CompDataset(test[test['seq_length']==130], bbps_folder, device='cuda', train_mode=False)
    dataloader = DataLoader(dataset, batch_size,)
    
    prediction = runner.predict_loader(loader=dataloader, model=model_init(), resume=best_model_path)
    prediction = np.concatenate([b['logits'].cpu().numpy() for b in prediction])[:,:,[0,1,3,2,4]]
    test_130_pred.append(prediction)
    
test_107_pred_oof = np.mean(test_107_pred, axis=0)
test_130_pred_oof = np.mean(test_130_pred, axis=0)

sample_sub = pd.read_csv(f'{path}sample_submission.csv')

sample_sub_ids = sample_sub['id_seqpos'].map(lambda x: x[:12])
ssi_107 = sample_sub_ids.isin(test[test['seq_length']==107]['id'])
ssi_130 = sample_sub_ids.isin(test[test['seq_length']==130]['id'])
sample_sub.loc[ssi_107, 'reactivity':] = test_107_pred_oof.reshape(-1, 5)
sample_sub.loc[ssi_130, 'reactivity':] = test_130_pred_oof.reshape(-1, 5)

sample_sub.to_csv(f"submission.csv", index=False)
sample_sub