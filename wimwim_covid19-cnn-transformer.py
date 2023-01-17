from shutil import copyfile

copyfile(src = "../usr/lib/modellib/modellib.py", dst = "../working/ModelLib.py")
import numpy as np

import pandas as pd

import torch

from torch.utils.data import TensorDataset, Dataset, DataLoader

from torch.utils.data.sampler import SubsetRandomSampler

from sklearn.model_selection import KFold,StratifiedKFold

from tqdm.auto import tqdm

from ModelLib import Create_model,stratified_group_k_fold

import random

import os

from copy import deepcopy

import math

from glob import glob
random.seed(831)

os.environ['PYTHONHASHSEED'] = str(721)

np.random.seed(1111)

torch.manual_seed(1117)

torch.cuda.manual_seed(1001)

torch.backends.cudnn.deterministic = True

torch.backends.cudnn.benchmark = False

# device = 'cpu'

device = 'cuda'
# train_x = np.load('../input/covid19fe/train_x.npy')

# test_x = np.load('../input/covid19fe/test_x.npy')

train_x = np.load('../input/covid19fe/train_aug_x.npy')

test_x = np.load('../input/covid19fe/test_aug_x.npy')

train_bpps = np.load('../input/covid19fe/train_bpps.npy')

test_bpps = np.load('../input/covid19fe/test_bpps.npy')

train_viennarna_bpps = np.load('../input/covid19extrafeatures/train_viennarna_bpps.npy')

test_viennarna_bpps = np.load('../input/covid19extrafeatures/test_viennarna_bpps.npy')

train_mat = np.load('../input/covid19extrafeatures/train_mat.npy')

test_mat = np.load('../input/covid19extrafeatures/test_mat.npy')

train_aug_mat = np.load('../input/covid19fe/train_aug_mat.npy')

test_aug_mat = np.load('../input/covid19fe/test_aug_mat.npy')

label = np.load('../input/covid19fe/label.npy')

label_error = np.load('../input/covid19fe/label_error.npy')

signal_to_noise = np.load('../input/covid19fe/signal_to_noise.npy')
train_x = train_x[:,:,[0,2,3,4,5]]

test_x = test_x[:,:,[0,2,3,4,5]]
train_bpps = np.concatenate([np.expand_dims(train_bpps,axis=1),np.expand_dims(train_viennarna_bpps,axis=1),np.expand_dims(train_mat,axis=1),np.expand_dims(train_aug_mat,axis=1)],axis=1)

test_bpps = np.concatenate([np.expand_dims(test_bpps,axis=1),np.expand_dims(test_viennarna_bpps,axis=1),np.expand_dims(test_mat,axis=1),np.expand_dims(test_aug_mat,axis=1)],axis=1)
train = pd.read_json('../input/stanford-covid-vaccine/train.json',lines=True).drop('index',axis=1)

test = pd.read_json('../input/stanford-covid-vaccine/test.json',lines=True).drop('index',axis=1)



train_length = train.seq_length.values

test_length = test.seq_length.values



train_scored = train.seq_scored.values

test_scored = test.seq_scored.values



SN_filter_mask = (train.SN_filter==1).values

SN_filter = np.where(SN_filter_mask)[0]

signal_filter = np.where(signal_to_noise > 1)[0]
# from sklearn.preprocessing import OneHotEncoder,scale

# from sklearn.cluster import KMeans

# cluster_features = OneHotEncoder().fit_transform(train_x[:,:107,0].reshape(-1,1)).toarray().reshape([len(train),-1])

# cluster_features = scale(cluster_features,axis=0)

# kmeans_model = KMeans(n_clusters=200, random_state=721).fit(cluster_features)

# cluster = kmeans_model.labels_
class Covid19Dataset(Dataset):

    def __init__(self,X,bpps,mat,seq_length,scored_length,label=None,label_error=None,signal_to_noise=None,SN_filter_mask=None):

        self.X = X.astype(np.int)

#         self.bpps = np.log(bpps + 1e-8).astype(np.float32)

        self.bpps = bpps.astype(np.float32)

        

#         self.bpps = np.log(bpps + 1e-8)

#         self.bpps = np.concatenate([bpps.reshape([-1,130,130,1]),mat.reshape([-1,130,130,1])],axis=-1).astype(np.float32)

        if label is not None:

            self.label = label.astype(np.float32)

            self.signal_to_noise = signal_to_noise.astype(np.float32)

            self.label_error=label_error.astype(np.float32)

            self.SN_filter_mask = SN_filter_mask

        else:

            self.label = None

        self.mask = np.zeros([len(X),130],dtype=bool)

        for i in range(len(seq_length)):

            if seq_length[i] < 130:

                self.mask[i,seq_length[i]:] = True

        self.scored_mask = np.ones([len(X),130],dtype=bool)

        for i in range(len(scored_length)):

            if scored_length[i] < 130:

                self.scored_mask[i,scored_length[i]:] = False

        self.seq_length = seq_length





    def __len__(self):

        return len(self.X)



    def __getitem__(self, idx):

        N = self.seq_length[idx]

        X = self.X[idx,:N]

        bpps = self.bpps[idx,:,:N,:N]

        mask = self.mask[idx,:N]

        scored_mask = self.scored_mask[idx,:N]

        if self.label is not None:

            label = self.label[idx,:N]

            label_error= self.label_error[idx,:N]

            signal_to_noise = self.signal_to_noise[idx]

            SN_filter_mask = self.SN_filter_mask[idx]

            return X,bpps,mask,scored_mask,label,label_error,signal_to_noise,SN_filter_mask

        else:

            return X,bpps,mask,scored_mask
nepochs = 300

n_fold = 5

kf = StratifiedKFold(n_fold,shuffle=True,random_state=831)



dataset = Covid19Dataset(train_x,train_bpps,train_mat,train_length,train_scored,label,label_error,signal_to_noise,SN_filter_mask)

cv_score = []

# cv_score = [0.20199335118134817, 0.19580934941768646, 0.19966551661491394]

loss_weights = torch.Tensor([1.2,1.2,1.2,0.7,0.7]).reshape(1,5).to(device)

oof = np.zeros([len(train_x),68,3])

# oof = np.load('temp_oof.npy')

for fold,(trn_group, test_group) in tqdm(enumerate(kf.split(train_x,SN_filter_mask)),total=n_fold):

#     trn_group = np.intersect1d(trn_group,signal_filter)

    test_group = np.intersect1d(test_group,signal_filter)

    traindataset = Covid19Dataset(train_x[trn_group],

                             train_bpps[trn_group],

                             train_mat[trn_group],

                             train_length[trn_group],

                             train_scored[trn_group],

                             label[trn_group],

                             label_error[trn_group],

                             signal_to_noise[trn_group],

                             SN_filter_mask[trn_group])

    valdataset = Covid19Dataset(train_x[test_group],

                             train_bpps[test_group],

                             train_mat[test_group],

                             train_length[test_group],

                             train_scored[test_group],

                             label[test_group],

                             label_error[test_group],

                             signal_to_noise[test_group],

                             SN_filter_mask[test_group])

    

    args_loader = {'batch_size': 64, 'shuffle': True, 'num_workers': 0, 'pin_memory': True, 'drop_last': True}

    train_loader = DataLoader(traindataset, **args_loader)

    args_loader = {'batch_size': 64, 'shuffle': False, 'num_workers': 0, 'pin_memory': True, 'drop_last': False}

    val_loader = DataLoader(valdataset, **args_loader)

    

    dataloaders = {'train' : train_loader, 'val' : val_loader}

    

    model,optimizer,scheduler = Create_model(device)

    best_model = {'reactivity':None,'deg_Mg_pH10':None,"deg_Mg_50C":None}

    best_loss = {'reactivity': np.inf,'deg_Mg_pH10': np.inf,'deg_Mg_50C': np.inf}

    stop_count = 0

    for epoch in tqdm(range(nepochs)):

        epoch_loss = {'train': 0.0, 'val': 0.0, 'val_clean': 0.0, 'val_aug': 0.0,

                      'reactivity': 0.0,'deg_Mg_pH10': 0.0,'deg_Mg_50C': 0.0

                     }

        MA_loss = []

        test_pred = []

        test_pred_aug = []

        test_y = []

        for phase in ['train', 'val']:

            if phase == 'train':

                model.train()

            else:

                model.eval()

            running_loss = 0.0

            for x_b,bpps_b,mask_b,scored_mask_b,label_b,label_error_b,signal_to_noise_b,SN_filter_mask_b in dataloaders[phase]:

                x_b = x_b.long().to(device)

                bpps_b = bpps_b.to(device)

                mask_b = mask_b.to(device)

                label_b = label_b.to(device)

                label_error_b = label_error_b.to(device)

                signal_to_noise_b = signal_to_noise_b.to(device).unsqueeze(1).unsqueeze(1)

                signal_to_noise_b = torch.clamp(signal_to_noise_b/4.5,0,10)

#                 signal_to_noise_b = torch.clamp(signal_to_noise_b/5,0,10)

#                 signal_to_noise_b = torch.clamp(torch.log(1 + signal_to_noise_b)/1.5,0,10)

#                 signal_to_noise_b = torch.sqrt(torch.clamp(signal_to_noise_b,0,999))/2



                label_error_b = torch.log(1+1.0/label_error_b[:,:68]) / 2.2496114573105803

    

                SN_filter_mask_b = SN_filter_mask_b.to(device)

                if phase=='train':

                    aug_mask = torch.randint(low=0,high=2,size=[len(bpps_b)],dtype=bool).to(device)

                    x_b[aug_mask,:,-4] = x_b[aug_mask,:,-2]

                    x_b[aug_mask,:,-3] = x_b[aug_mask,:,-1]

                    x_b = x_b[:,:,:-2]

                    bpps_b[aug_mask,-2] = bpps_b[aug_mask,-1]

                    bpps_b = bpps_b[:,:-1]

                else:

                    x_b_aug = x_b.clone()

                    x_b_aug[:,:,-4] = x_b_aug[:,:,-2]

                    x_b_aug[:,:,-3] = x_b_aug[:,:,-1]

                    x_b_aug  = x_b_aug[:,:,:-2]

                    bpps_b_aug = bpps_b.clone()

                    bpps_b_aug[:,-2] = bpps_b_aug[:,-1]

                    bpps_b_aug = bpps_b_aug[:,:-1]

                    

                    x_b = x_b[:,:,:-2]

                    bpps_b = bpps_b[:,:-1]

                

#                 if phase=='train':

#                     label_b += torch.normal(torch.zeros_like(label_b),1) * 0.001*label_error_b

                

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase=='train'):

                    preds = model(x_b,bpps_b)

                    if phase=='val':

                        preds_aug = model(x_b_aug,bpps_b_aug)

                        preds2 = 0.5*(preds + preds_aug)

                        test_pred.append(preds[:,:68,:3].detach().cpu().numpy())

                        test_pred_aug.append(preds2[:,:68,:3].detach().cpu().numpy())

                        test_y.append(label_b[:,:68,:3].detach().cpu().numpy())



        

                    loss = (preds[:,:68] - label_b[:,:68])**2

                    loss = torch.sqrt((loss * signal_to_noise_b).reshape(-1,5).mean(0)).mean()

#                     loss = torch.sqrt((loss * signal_to_noise_b).reshape(-1,5)[:,:3].mean(0)).mean()

                

#                     loss = (preds[:,:68] - label_b[:,:68])**2

#                     loss = (label_error_b * loss).mean()



                    if phase=='train':

                        loss.backward()

#                         torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

                        optimizer.step()

#                         scheduler.step()

                if phase == 'train':

                    running_loss += loss.item() / len(dataloaders[phase])

                else:

                    running_loss += loss.item() / len(dataloaders[phase])

            if phase == 'train':

                epoch_loss['train'] = running_loss

            else:

                epoch_loss['val'] = running_loss

                test_pred = np.concatenate(test_pred,axis=0)

                test_pred_aug = np.concatenate(test_pred_aug,axis=0)

                test_y = np.concatenate(test_y,axis=0)

                epoch_loss['val_clean'] = np.sqrt(((test_pred-test_y)**2).reshape(-1,3).mean(0)).mean()

                res = ((test_pred_aug-test_y)**2).reshape(-1,3)

                epoch_loss['val_aug'] = np.sqrt(res.mean(0)).mean()

                epoch_loss['reactivity'] = np.sqrt(res[:,0].mean())

                epoch_loss['deg_Mg_pH10'] = np.sqrt(res[:,1].mean())

                epoch_loss['deg_Mg_50C'] = np.sqrt(res[:,2].mean())

        scheduler.step()

        for i,cat in enumerate(['reactivity','deg_Mg_pH10','deg_Mg_50C']):

            if epoch_loss[cat] < best_loss[cat]:

                best_loss[cat] = epoch_loss[cat]

                torch.save(model.state_dict(), f'fold{fold+1}_{cat}_model.pt')

                oof[test_group,:,i] = test_pred_aug[:,:,i]

                stop_count = 0

        stop_count += 1

        print("Epoch {}/{}   -   loss: {:5.5f} - val_loss: {:5.5f} - val_best_loss: {:5.5f} - val_aug_loss: {:5.5f} - reactivity: {:5.5f} - deg_Mg_pH10: {:5.5f} - deg_Mg_50C: {:5.5f}".format(epoch+1, nepochs, epoch_loss['train'], epoch_loss['val'], sum(best_loss.values())/3, epoch_loss['val_aug'], epoch_loss['reactivity'], epoch_loss['deg_Mg_pH10'], epoch_loss['deg_Mg_50C']))

        if stop_count > 50:

            break

#     cv_score += best_score / 5

    cv_score.append(sum(best_loss.values())/3)
for i in range(n_fold):

    print(f"fold {i+1} score:",cv_score[i])

print()

print("CV score:",np.mean(cv_score))

np.save('oof_{:5.5f}'.format(np.mean(cv_score)),oof)
dataset = Covid19Dataset(test_x,test_bpps,test_mat,test_length,test_scored)

args_loader = {'batch_size': 1, 'shuffle': False, 'num_workers': 0, 'pin_memory': True, 'drop_last': False}

test_loader = DataLoader(dataset, **args_loader)

test_predictions = np.zeros([len(test_x),130,5])

for j,col in enumerate(['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C']):

    paths = glob(f'fold*_{col}_model.pt')

    with torch.no_grad():

        for path in tqdm(paths):

            model.load_state_dict(torch.load(path))

            model.eval().to(device)

            predictions = []

            for x_b,bpps_b,mask_b,scored_mask_b in test_loader:

                x_b = x_b.long().to(device)

                bpps_b = bpps_b.to(device)

                mask_b = mask_b.to(device)

                x_b_aug = x_b.clone()

                x_b_aug[:,:,-4] = x_b_aug[:,:,-2]

                x_b_aug[:,:,-3] = x_b_aug[:,:,-1]

                x_b_aug  = x_b_aug[:,:,:-2]

                bpps_b_aug = bpps_b.clone()

                bpps_b_aug[:,-2] = bpps_b_aug[:,-1]

                bpps_b_aug = bpps_b_aug[:,:-1]

                x_b = x_b[:,:,:-2]

                bpps_b = bpps_b[:,:-1]



                preds = model(x_b,bpps_b)

                preds_aug = model(x_b_aug,bpps_b_aug)

                preds = 0.5*(preds + preds_aug)



                p = torch.zeros([preds.shape[0],130,preds.shape[2]])

                p[:,:preds.shape[1]] = preds.cpu()

                predictions.append(p)

            predictions = torch.cat(predictions,dim=0).numpy()

            test_predictions[:,:,j] += predictions[:,:,j] / len(paths)
ss = pd.read_csv("../input/stanford-covid-vaccine/sample_submission.csv",index_col=0)
for n,row in tqdm(test.iterrows(),total=len(test)):

    test_id = row['id']

    seq_len = row['seq_length']

    for i in range(seq_len):

        for j,col in enumerate(['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C']):

            ss.loc[test_id+'_'+str(i),col] = test_predictions[n,i,j]
ss.to_csv("submission_cnn_{:5.5f}.csv".format(np.mean(cv_score)),index=True)