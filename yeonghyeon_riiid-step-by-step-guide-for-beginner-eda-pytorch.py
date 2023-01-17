import riiideducation



try: env = riiideducation.make_env()

except: pass
# For computation

import os, gc, copy, random

import numpy as np

import pandas as pd



# For visualization

import seaborn as sns

import matplotlib.pyplot as plt



# For data analysis

from sklearn import metrics

from sklearn.preprocessing import LabelEncoder

from  sklearn.model_selection import train_test_split
# For using PyTorch

import torch

import torch.nn as nn

import torch.multiprocessing as mp



from torch import optim

from torchvision import models

from torch.nn import functional as F

from torch.utils.data import DataLoader, Dataset
os.environ['PYTHONHASHSEED'] = str(32)

random.seed(32)

np.random.seed(32)

torch.random.manual_seed(32)
def print_dataframe(df, length=5):

    

    """ -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

    Define the function for checking a dataframe.

    This function shows the shape of the dataframe firstly.

    Then, show the head and tail part of the whole dataframe.

    -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*- """

    

    print("* Key of dataframe")

    print(list(df.keys()))

    print("\n* Shape of dataframe:", df.shape)

    print("\n* Head of dataframe")

    print(df.head(length))

    print("\n* Tail of dataframe")

    print(df.tail(length))
%%time

df_train = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/train.csv', \

    usecols=[1, 2, 3, 4, 7, 8, 9], \

    dtype={'timestamp': 'int64', 'user_id': 'int32' ,'content_id': 'int16','content_type_id': 'int8', \

        'answered_correctly':'int8','prior_question_elapsed_time': 'float32','prior_question_had_explanation': 'boolean'})
print_dataframe(df_train)
# Sub-Step 1. Select the rows where the value of 'content_type_id' is 'False'.

df_train = df_train[df_train.content_type_id == False]



# Sub-Step 2. Sort the dataframe with ascending order.

# The base column for sorting is 'timestamp'.

df_train = df_train.sort_values(['timestamp'], ascending=True)



# Sub-Step 3. Then, drop the two columns named 'timestamp' and 'content_type_id'

# They are not useful anymore.

# If you do not want to use 'inplace=True', you can use another command as follows.

# >>> df_train = df_train.drop(['timestamp', 'content_type_id'], axis=1)

df_train.drop(['timestamp', 'content_type_id'], axis=1, inplace=True)



print_dataframe(df_train)
# Sub-Step 1. Calculate the mean and sum of the column 'answered_correctly' group by 'user_id'.

agg_uid = df_train[['user_id','answered_correctly']].groupby(['user_id']).agg(['mean', 'sum'])

agg_uid.columns = ["mean_by_uid", 'sum_by_uid']



print_dataframe(agg_uid)
# Sub-Step 2. Calculate the sum of the column 'answered_correctly' group by 'content_id'.

agg_cid = df_train[['content_id','answered_correctly']].groupby(['content_id']).agg(['mean'])

agg_cid.columns = ["mean_by_cid"]



print_dataframe(agg_cid)
# Sub-Step 3. Merge the generated informations to training set.

# In this notebook, the merged dataframe is newly allocated to another variable for maintaining the original dataframe.

# If you do not need to maintain the original dataframe, you can use the following command.

# >>> del df_train

# You can also use the above command with the lack of RAM situation.

X_tr = pd.merge(df_train, agg_uid, on=['user_id'], how="left")

X_tr = pd.merge(X_tr, agg_cid, on=['content_id'], how="left")

X_tr = X_tr[X_tr.answered_correctly != -1]

X_tr = X_tr.sort_values(['user_id'])

Y_tr = X_tr[['answered_correctly']]

X_tr = X_tr.drop(['answered_correctly'], axis=1)



del df_train
print("Dataframe: X_tr")

print_dataframe(X_tr)



print("\nDataframe: Y_tr")

print_dataframe(Y_tr)
list_key_tr = list(X_tr.keys())

for idx, name_key in enumerate(list_key_tr):

    if(X_tr[name_key].isna().sum() > 0):

        try: X_tr[name_key] = X_tr[name_key].fillna(X_tr[name_key].median())

        except: pass

    

print_dataframe(X_tr)
# Sub-Step 1. Define and initializing the LabelEncoder class.

label_encoder = LabelEncoder()



# Sub-Step 2. Encode the non-numeric column 'prior_question_had_explanation' via the function 'fit_transform'.

X_tr['prior_question_had_explanation_enc'] = label_encoder.fit_transform(X_tr['prior_question_had_explanation'])



# Sub-Step 3. Drop the original non-numeric column.

X_tr.drop(['prior_question_had_explanation'], axis=1, inplace=True)



print_dataframe(X_tr)
plt.figure(figsize=(8, 5))

plt.title("Correlation Coefficient Matrix")

sns.heatmap(X_tr.corr(), cmap='jet', annot=True, fmt='.3f')

plt.show()

plt.close()
list_key_tr = list(X_tr.keys())

for idx, name_key in enumerate(list_key_tr):

    plt.figure(figsize=(6, 3))

    

    plt.title(name_key.upper())

    X_tr[name_key].hist() # The pandas supports 'hist' function.

    plt.ylabel(name_key)

    

    plt.grid()

    plt.tight_layout()

    plt.show()

    plt.close()

    

    gc.collect()
class neuralnet(nn.Module):

    

    def __init__(self, n_additional_features, n_outputs):

        

        super(neuralnet, self).__init__()

        

        self.fc0_1 = nn.Linear(n_additional_features, 512) 

        self.fc0_1d = nn.Dropout(0.5)

        self.fc0_2 = nn.Linear(512, 128) 

        self.fc0_2d = nn.Dropout(0.5)

        self.fc0_3 = nn.Linear(128, 32) 

        self.fc0_3d = nn.Dropout(0.5)

        

        self.fc1 = nn.Linear(32, 64) 

        self.fc1_d = nn.Dropout(0.5)

        self.fc1_skip = nn.Linear(32, 64) 

        

        self.fc2 = nn.Linear(64, 128) 

        self.fc2_d = nn.Dropout(0.5)

        self.fc2_skip = nn.Linear(64, 128) 

        

        self.fc3 = nn.Linear(192, 256) 

        self.fc3_d = nn.Dropout(0.5)

        self.fc4 = nn.Linear(384, n_outputs) 



    def forward(self, additional_features):

        

        out0 = additional_features

        

        out0_1 = F.elu(

            self.fc0_1d(

                self.fc0_1(out0)

            )

        )

        out0_2 = F.elu(

            self.fc0_2d(

                self.fc0_2(out0_1)

            )

        )

        out0_3 = F.elu(

            self.fc0_3d(

                self.fc0_3(out0_2)

            )

        )

        

        out1 = F.elu(

            self.fc1_d(

                self.fc1(out0_3)

            )

        )

        out1_s = F.elu(

            self.fc1_skip(out0_3)

        )

        

        out2 = F.elu(

            self.fc2_d(

                self.fc2(out1)

            )

        )

        out2_s = F.elu(

            self.fc2_skip(out1)

        )

        

        out3 = F.elu(

            self.fc3_d(

                self.fc3(

                    torch.cat((out2, out1_s), 1)

                )

            )

        )

        out4 = F.relu(

            self.fc4(

                torch.cat((out3, out2_s), 1)

            )

        )

        

        return out4

    

    def loss_mse(self, pred, true):

        

        return  torch.mean(torch.sum((pred - true)**2, dim=1))
ratio_val = 0.1

batch_size = 32

num_workers = 4



epochs = 10

learning_rate = 1e-3

early_stopper = False

patience = 3
X_tr, X_val, Y_tr, Y_val = train_test_split(X_tr, Y_tr, test_size=ratio_val, shuffle=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



model = neuralnet(X_tr.shape[1], 1).to(device)

print("Number of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
def measure_auroc(label, logit):

    

    """ -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

    Define the function for measuring the Area Under the Receiver Operating Characteristic Curve (AUROC).

    -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*- """

    

    fpr, tpr, thresholds = metrics.roc_curve(label, logit, pos_label=0)

    auroc = metrics.auc(fpr, tpr)

    if(auroc < 0.5):

        fpr, tpr, thresholds = metrics.roc_curve(label, logit, pos_label=1)

        auroc = metrics.auc(fpr, tpr)

    return auroc
optimizer = optim.Adam(model.parameters(), \

    lr=learning_rate, betas=(0.5, 0.999), eps=1e-08, \

    weight_decay=learning_rate/10, amsgrad=True)



epoch, loss_min = 0, 1e+100

epoch_loss_tr, epoch_loss_val = [], []

epoch_metric_tr, epoch_metric_val = [], []



while((epoch<epochs) and not(early_stopper)):

    epoch += 1



    loss_tr, metric_tr, label, logit = 0, 0, None, None

    model.train()

    terminator, idx_s, idx_e, amount = False, 0, batch_size, X_tr.shape[0]

    cnt_tr = 0

    while(True):

        cnt_tr += 1

        if(idx_s % 65536 == 0): print("Run | Epoch [%d/%d], Batch-index [%d~%d / %d]" %(epoch, epochs, idx_s, idx_e, amount))

        tmp_x, tmp_y = torch.from_numpy(np.asarray(X_tr[idx_s:idx_e])), torch.from_numpy(np.asarray(Y_tr[idx_s:idx_e]))

        optimizer.zero_grad()

        predictions = model(tmp_x.float().to(device))

        loss_tmp_tr = model.loss_mse(predictions, tmp_y.to(device))

        loss_tmp_tr.backward()

        loss_tr += loss_tmp_tr.item()

        

        label_tmp = list(np.squeeze(tmp_y.detach().numpy()))

        logit_tmp = list(np.squeeze(predictions.detach().numpy()))

        if(label is None): label, logit = label_tmp, logit_tmp

        else:

            label.extend(label_tmp)

            logit.extend(logit_tmp)

        

        optimizer.step()

        gc.collect()

        

        break # !!! Remove this break command when you use this notebook in practice. 

        if(terminator): break

        idx_s, idx_e = idx_e, idx_e+batch_size

        if(idx_e >= amount): terminator = True

    

    print(type(label), type(logit))

    metric_tr = measure_auroc(label, logit)

    

    

    """ -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*- """

    

    loss_val, metric_val, label, logit = 0, 0, None, None

    model.eval()

    terminator, idx_s, idx_e, amount = False, 0, batch_size, X_val.shape[0]

    cnt_val = 0

    with torch.no_grad():

        while(True):

            cnt_val += 1

            if(idx_s % 16384 == 0): print("Run | Epoch [%d/%d], Batch-index [%d~%d / %d]" %(epoch, epochs, idx_s, idx_e, amount))

            tmp_x, tmp_y = torch.from_numpy(np.asarray(X_val[idx_s:idx_e])), torch.from_numpy(np.asarray(Y_val[idx_s:idx_e]))

            predictions = model(tmp_x.float().to(device))

            loss_tmp_val = model.loss_mse(predictions, tmp_y.to(device)).mean()

            loss_val += loss_tmp_val.item()

            

            label_tmp = list(np.squeeze(tmp_y.detach().numpy()))

            logit_tmp = list(np.squeeze(predictions.detach().numpy()))

            if(label is None): label, logit = label_tmp, logit_tmp

            else:

                label.extend(label_tmp)

                logit.extend(logit_tmp)

            

            break # !!! Remove this break command when you use this notebook in practice. 

            if(terminator): break

            idx_s, idx_e = idx_e, idx_e+batch_size

            if(idx_e >= amount): terminator = True

            

            gc.collect()



    loss_tr = loss_tr / cnt_tr

    loss_val = loss_val / cnt_val

    print("Epoch [%d/%d]" %(epoch, epochs))

    print(" Training   | MSE: %.4f   AUROC: %.5f" %(loss_tr, metric_tr))

    print(" Validation | MSE: %.4f   AUROC: %.5f" %(loss_val, metric_val))

    

    epoch_loss_tr.append(loss_tr)

    epoch_loss_val.append(loss_val)

    epoch_metric_tr.append(metric_tr)

    epoch_metric_val.append(metric_val)

    

    if(loss_val <= loss_min):

        loss_min = loss_val

        best_model = copy.deepcopy(model.state_dict())

        epochs_no_improve = 0



    else:

        epochs_no_improve += 1

        if(epochs_no_improve == patience):

            print("Early stopping!\n")

            early_stopper = True

            model.load_state_dict(best_model)

    

    break
plt.figure(figsize=(12, 5))



plt.subplot(1, 2, 1)

plt.title('Loss Curve')

plt.plot(epoch_loss_tr, label='Training')

plt.plot(epoch_loss_val, label='Validation')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.grid()

plt.legend(loc='upper right')



plt.subplot(1, 2, 2)

plt.title('AUROC Curve')

plt.plot(epoch_metric_tr, label='Training')

plt.plot(epoch_metric_val, label='Validation')

plt.ylabel('AUROC')

plt.xlabel('Epoch')

plt.grid()

plt.legend(loc='lower right')



plt.show()

plt.close()
del X_tr, X_val, Y_tr, Y_val
df_test = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/example_test.csv')
print_dataframe(df_test)
X_te = pd.merge(df_test, agg_uid, on=['user_id'], how="left")

X_te = pd.merge(X_te, agg_cid, on=['content_id'], how="left")

X_te = X_te.sort_values(['user_id'])



del df_test



X_te['prior_question_had_explanation_enc'] = label_encoder.fit_transform(X_te['prior_question_had_explanation'])

X_te.drop(['prior_question_had_explanation'], axis=1, inplace=True)



list_key_te = list(X_te.keys())

for idx, name_key in enumerate(list_key_te):

    if(X_te[name_key].isna().sum() > 0):

        try: X_te[name_key] = X_te[name_key].fillna(X_te[name_key].median())

        except: pass

    

print_dataframe(X_te)
list_key_te = list(X_te.keys())

for name_key in list_key_tr:

    print(name_key)

    try: idx_key = list_key_te.index(name_key)

    except: print(name_key, "not included")

    else: list_key_te.pop(idx_key)



key4drop = list_key_te

print("Key for drop")

print(key4drop)



X_te.drop(key4drop, axis=1, inplace=True)

print_dataframe(X_te)
model.eval()

logit = []

terminator, idx_s, idx_e, amount = False, 0, batch_size, X_te.shape[0]

with torch.no_grad():

    while(True):

        tmp_x = torch.from_numpy(np.asarray(X_te[idx_s:idx_e]))

        predictions = model(tmp_x.float().to(device))

        

        logit_tmp = list(np.squeeze(predictions.detach().numpy()))

        if(logit is None): logit = logit_tmp

        else: logit.extend(logit_tmp)

            

        if(terminator): break

        idx_s, idx_e = idx_e, idx_e+batch_size

        if(idx_e >= amount): terminator = True



        gc.collect()
df_sb = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/example_sample_submission.csv')

print_dataframe(df_sb)
for idx in range(df_sb.shape[0]):

    df_sb.loc[idx, 'answered_correctly'] = logit[idx]

print(logit[:5])

print_dataframe(df_sb)
df_sb.to_csv('submission.csv', index=False, float_format='%.3f')