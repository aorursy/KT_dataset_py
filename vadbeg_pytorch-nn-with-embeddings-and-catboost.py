import warnings

warnings.filterwarnings('ignore')



import os



import pandas as pd

import numpy as np

from tqdm import tqdm_notebook as tqdm



import matplotlib.pyplot as plt

plt.style.use('ggplot')

import seaborn as sns



from sklearn.model_selection import train_test_split

from sklearn import metrics, preprocessing
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

train = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/train.csv')

test = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/test.csv')

sample_submission = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/sample_submission.csv')
print(f'Train shape: {train.shape}',

      f'Test shape: {test.shape}',

      f'Submission shape: {sample_submission.shape}', sep=' | ')
train.head()
sns.countplot(train.target)
all_df = pd.concat([train, test], axis=0, ignore_index=True)
all_df.index.nunique(), len(all_df.index)
all_df = all_df.drop('id', axis=1)
all_df.head()
nunique_vals = list()



for column in all_df:

    nunique_vals.append(all_df[column].nunique())

    

pd.DataFrame({'columns': all_df.columns,

              'num_of_unique': nunique_vals})
for column in all_df.columns:



    unique_values = all_df[column].unique()

    

    print(f'Statistics fot column: {column}')

    print(f'Column unique values:\n {unique_values}')

    print(f'Number of unique values: {len(unique_values)}')

    print(f'Number of NAN values: {all_df[column].isna().sum()}')

    print('_' * 50)
month_temperature = all_df.groupby(['month', 'ord_2'])['ord_2'].count().to_frame()

month_temperature = month_temperature.rename(columns={'ord_2': 'num_of_days'})

month_temperature = month_temperature.reset_index()
month_temperature.head()
plt.rcParams.update({'font.size': 25})



month = 1



for i in range(6):

    fig, ax = plt.subplots(1, 2, figsize=(60, 20))

    

    for j in range(2):

        

        mt_part = month_temperature.loc[month_temperature.month == month]



        ax[j].set_title(month)

        sns.barplot(x=mt_part['ord_2'], y=mt_part['num_of_days'], ax=ax[j])

        

        month += 1

    

    plt.show()

    plt.pause(0.1)
all_df['ord_5_1'] = all_df['ord_5'].str[0]

all_df['ord_5_2'] = all_df['ord_5'].str[1]



all_df = all_df.drop('ord_5', axis=1)
all_df['nan_features'] = all_df.isna().sum(axis=1)
all_df['month_sin'] = np.sin((all_df['month'] - 1) * (2.0 * np.pi / 12))

all_df['month_cos'] = np.cos((all_df['month'] - 1) * (2.0 * np.pi / 12))



all_df['day_sin'] = np.sin((all_df['day'] - 1) * (2.0 * np.pi / 7))

all_df['day_cos'] = np.cos((all_df['day'] - 1) * (2.0 * np.pi / 7))
categorical = ['bin_0', 'bin_1', 'bin_2', 'bin_3',

               'bin_4', 'day', 'month', 'nom_0',

               'nom_1', 'nom_2', 'nom_3', 'nom_4',

               'nom_5', 'nom_6', 'nom_7', 'nom_8',

               'nom_9', 'ord_0', 'ord_1', 'ord_2', 

               'ord_3', 'ord_4', 'ord_5_1', 'ord_5_2']



nom_5_9 = ['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']



continuous = ['month_sin', 'month_cos',

              'day_sin', 'day_cos']
features = [x for x in all_df.columns 

            if x not in ['id', 'target'] + continuous]
for feat in tqdm(features):

    lbl_enc = preprocessing.LabelEncoder()

    

    all_df[feat] = lbl_enc.fit_transform(all_df[feat]. \

                                         fillna('-1'). \

                                         astype(str).values)

    

all_df['target'] = all_df['target'].fillna(-1)

all_df[continuous] = all_df[continuous].fillna(-2)
from scipy.stats import chi2_contingency, entropy

from collections import Counter





def cramers_v(x, y):

    """

        Calculates Cramer's V statistic for categorical-categorical association.

        

        :param x: pd.Series or np.array

        :param y: pd.Series or np.array 

        

        :return: Cramer's V statistic, float in range of [0, 1]

    """

    

    confusion_matrix = pd.crosstab(x, y)

    

    chi2 = chi2_contingency(confusion_matrix)[0]

    

    n = confusion_matrix.sum().sum()

    

    r, k = confusion_matrix.shape

    phi_2 = chi2 / n

    

    phi2corr = max(0, phi_2 - ((k - 1) * (r - 1)) / (n - 1))

    

    rcorr = r - ((r - 1) ** 2) / (n - 1)

    kcorr = k - ((k - 1) ** 2) / (n - 1)

    

    res = np.sqrt(phi2corr / min(kcorr - 1, rcorr - 1))

    

    return res



def conditional_entropy(x, y):

    """

        Calculates the conditional entropy of x given y: S(x|y)

    

        :param x: pd.Series or np.array

        :param y: pd.Series or np.array 

        

        :return: float

    """

    

    y_counter = Counter(y)

    xy_counter = Counter((list(zip(x, y))))

    

    total_occurrences = sum(y_counter.values())

    entropy = 0.0

    

    for xy in xy_counter.keys():

        p_xy = xy_counter[xy] / total_occurrences

        p_y = y_counter[xy[1]] / total_occurrences

        

        entropy += p_xy * np.log(p_y / p_xy)

        

    return entropy



def theils_u(x, y):

    """

        Calculates Theil's U statistic (Uncertainty coefficient) for categorical-categorical association.

    

        :param x: pd.Series or np.array

        :param y: pd.Series or np.array 

        

        :return: Theil's U statistic, float in range of [0, 1]

    """

    

    s_xy = conditional_entropy(x, y)

    x_counter = Counter(x)

    

    total_occurrences = sum(x_counter.values())

    

    p_x = list(map(lambda n: n / total_occurrences, x_counter.values()))

    s_x = entropy(p_x)

    

    if s_x == 0:

        return 1

    

    else:

        return (s_x - s_xy) / s_x
plt.rcParams.update({'font.size': 10})
%%time



plt.subplots(figsize=(18, 18))

plt.title('Cramers V')



corr_res = round(all_df.corr(method=cramers_v), 2)

sns.heatmap(corr_res, annot=True)
%%time



plt.subplots(figsize=(18, 18))

plt.title('Pearson')





corr_simple_res = round(all_df.corr(), 2)

sns.heatmap(corr_simple_res, annot=True)
%%time



plt.subplots(figsize=(18, 18))

plt.title('Uncertainty coefficient')



corr_res = round(all_df.corr(method=theils_u), 2)

sns.heatmap(corr_res, annot=True)
to_dummies = ['day', 'month', 'nom_0',

              'nom_1', 'nom_2', 'nom_3', 'nom_4']
%%time



all_df = pd.get_dummies(all_df,

                        columns=to_dummies,

                        sparse=True,

                        dtype=np.int8)
all_df.shape
all_df.isna().sum().sum()
%%time



train = all_df[:train.shape[0]]

test = all_df[train.shape[0]:]
print(f'Train shape: {train.shape}',

      f'Test shape: {test.shape}', sep=' | ')
train.isna().sum().sum(), test.isna().sum().sum() 
%%time



train_data = train.drop('target', axis=1).to_numpy()

train_target = train['target'].to_numpy()



test_data = test.drop('target', axis=1).to_numpy()
%%time



categorical = all_df.drop(['target'] + continuous,

                          axis=1).columns



cat_cols_idx, cont_cols_idx = list(), list()



for idx, column in enumerate(all_df.drop('target',

                                         axis=1).columns):

    if column in categorical:

        cat_cols_idx.append(idx)

    elif column in continuous:

        cont_cols_idx.append(idx)
train_data, train_target, test_data
import torch

from torch.utils.data import Dataset, DataLoader

from torch.utils.data.sampler import SubsetRandomSampler
import random



def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    

    torch.manual_seed(seed)

    torch.backends.cudnn.deterministick = True

    torch.backends.cudnn.benchmark = False 

    

set_seed(27)
class ClassificationDataset(Dataset):

    def __init__(self, data, targets=None,

                 is_train=True, cat_cols_idx=None,

                 cont_cols_idx=None):

        self.data = data

        self.targets = targets

        self.is_train = is_train

        self.cat_cols_idx = cat_cols_idx

        self.cont_cols_idx = cont_cols_idx

    

    def __getitem__(self, idx):

        row = self.data[idx].astype('float32')

        

        data_cat = []

        data_cont = []

        

        result = None

        

        if self.cat_cols_idx:

            data_cat = torch.tensor(row[self.cat_cols_idx])

            

        if self.cont_cols_idx:

            data_cont = torch.tensor(row[self.cont_cols_idx])

                

        data = [data_cat, data_cont]

                

        if self.is_train:

            result = {'data': data,

                      'target': torch.tensor(self.targets[idx])}

        else:

            result = {'data': data}

            

        return result

            

    

    def __len__(self):

        return(len(self.data))
train_dataset = ClassificationDataset(train_data, 

                                      targets=train_target,

                                      cat_cols_idx=cat_cols_idx,

                                      cont_cols_idx=cont_cols_idx)

test_dataset = ClassificationDataset(test_data,

                                     cat_cols_idx=cat_cols_idx,

                                     cont_cols_idx=cont_cols_idx,

                                     is_train=False)
len(test_dataset)
print(f'First element of train_dataset: {train_dataset[1]}',

      f'First element of test_dataset: {test_dataset[1]}', sep='\n')
def split_dataset(trainset, valid_size=0.2, batch_size=64):

    num_train = len(trainset)

    

    indices = list(range(num_train))

    np.random.shuffle(indices)

    

    split = int(np.floor(valid_size * num_train))

    

    valid_idx, train_idx = indices[:split], indices[split:]

    

    valid_sampler = SubsetRandomSampler(valid_idx)

    train_sampler = SubsetRandomSampler(train_idx)

    

    valid_loader = DataLoader(trainset, 

                              batch_size=batch_size, 

                              sampler=valid_sampler)

    train_loader = DataLoader(trainset, 

                              batch_size=batch_size, 

                              sampler=train_sampler)

    

    return train_loader, valid_loader
train_loader, valid_loader = split_dataset(train_dataset, 

                                           valid_size=0.2, 

                                           batch_size=2000)
next(iter(train_loader))
len(train_loader)
class ClassificationEmbdNN(torch.nn.Module):

    

    def __init__(self, emb_dims, no_of_cont=None):

        super(ClassificationEmbdNN, self).__init__()

        

        self.emb_layers = torch.nn.ModuleList([torch.nn.Embedding(x, y)

                                               for x, y in emb_dims])

        

        no_of_embs = sum([y for x, y in emb_dims])

        self.no_of_embs = no_of_embs

        self.emb_dropout = torch.nn.Dropout(0.2)

        

        self.no_of_cont = 0

        if no_of_cont:

            self.no_of_cont = no_of_cont

            self.bn_cont = torch.nn.BatchNorm1d(no_of_cont)

        

        self.fc1 = torch.nn.Linear(in_features=self.no_of_embs + self.no_of_cont, 

                                   out_features=256)

        self.dropout1 = torch.nn.Dropout(0.2)

        self.bn1 = torch.nn.BatchNorm1d(256)

        self.act1 = torch.nn.ReLU()

        

        self.fc2 = torch.nn.Linear(in_features=256, 

                                   out_features=256)

        self.dropout2 = torch.nn.Dropout(0.2)

        self.bn2 = torch.nn.BatchNorm1d(256)

        self.act2 = torch.nn.ReLU()

        

        self.fc3 = torch.nn.Linear(in_features=256, 

                                   out_features=64)

        self.dropout3 = torch.nn.Dropout(0.2)

        self.bn3 = torch.nn.BatchNorm1d(64)

        self.act3 = torch.nn.ReLU()

        

        self.fc4 = torch.nn.Linear(in_features=64, 

                                   out_features=1)

        self.act4 = torch.nn.Sigmoid()

        

    def forward(self, x_cat, x_cont=None):

        if self.no_of_embs != 0:

            x = [emb_layer(x_cat[:, i])

                 for i, emb_layer in enumerate(self.emb_layers)]

        

            x = torch.cat(x, 1)

            x = self.emb_dropout(x)

            

        if self.no_of_cont != 0:

            x_cont = self.bn_cont(x_cont)

            

            if self.no_of_embs != 0:

                x = torch.cat([x, x_cont], 1)

            else:

                x = x_cont

        

        x = self.fc1(x)

        x = self.dropout1(x)

        x = self.bn1(x)

        x = self.act1(x)

        

        x = self.fc2(x)

        x = self.dropout2(x)

        x = self.bn2(x)

        x = self.act2(x)

        

        x = self.fc3(x)

        x = self.dropout3(x)

        x = self.bn3(x)

        x = self.act3(x)

        

        x = self.fc4(x)

        x = self.act4(x)

        

        return x
def train_network(model, train_loader, valid_loader,

                  loss_func, optimizer, n_epochs=20,

                  saved_model='model.pt'):

    

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model.to(device)

    

    train_losses = list()

    valid_losses = list()

    

    valid_loss_min = np.Inf

    

    for epoch in range(n_epochs):

        train_loss = 0.0

        valid_loss = 0.0

        

        train_auc = 0.0

        valid_auc = 0.0

        

        model.train()

        for batch in tqdm(train_loader):

            optimizer.zero_grad()

            

            output = model(batch['data'][0].to(device, 

                                               dtype=torch.long),

                           batch['data'][1].to(device, 

                                               dtype=torch.float))

            

            

            loss = loss_func(output, batch['target'].to(device, 

                                                        dtype=torch.float))

            

            loss.backward()

            optimizer.step()

            

            train_auc += metrics.roc_auc_score(batch['target'].cpu().numpy(),

                                               output.detach().cpu().numpy())



            train_loss += loss.item() * batch['data'][0].size(0)  #!!!

    



        model.eval()

        for batch in tqdm(valid_loader):

            output = model(batch['data'][0].to(device, 

                                               dtype=torch.long),

                           batch['data'][1].to(device, 

                                               dtype=torch.float))

            

            

            loss = loss_func(output, batch['target'].to(device, 

                                                        dtype=torch.float))

            

            valid_auc += metrics.roc_auc_score(batch['target'].cpu().numpy(),

                                               output.detach().cpu().numpy())

            valid_loss += loss.item() * batch['data'][0].size(0)  #!!!

           

        

        train_loss = np.sqrt(train_loss / len(train_loader.sampler.indices))

        valid_loss = np.sqrt(valid_loss / len(valid_loader.sampler.indices))



        train_auc = train_auc / len(train_loader)

        valid_auc = valid_auc / len(valid_loader)

        

        train_losses.append(train_loss)

        valid_losses.append(valid_loss)



        print('Epoch: {}. Training loss: {:.6f}. Validation loss: {:.6f}'

              .format(epoch, train_loss, valid_loss))

        print('Training AUC: {:.6f}. Validation AUC: {:.6f}'

              .format(train_auc, valid_auc))

        

        if valid_loss < valid_loss_min:  # let's save the best weights to use them in prediction

            print('Validation loss decreased ({:.6f} --> {:.6f}). Saving model...'

                  .format(valid_loss_min, valid_loss))

            

            torch.save(model.state_dict(), saved_model)

            valid_loss_min = valid_loss

            

    

    return train_losses, valid_losses

        
cat_dim = [int(all_df[col].nunique()) for col in categorical]

cat_dim = [[x, min(200, (x + 1) // 2)] for x in cat_dim]



for el in cat_dim:

    if el[0] < 10:

        el[1] = el[0]



cat_dim
model = ClassificationEmbdNN(emb_dims=cat_dim, 

                             no_of_cont=len(continuous))



loss_func = torch.nn.BCELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.1)



train_losses, valid_losses = train_network(model=model, 

                                           train_loader=train_loader, 

                                           valid_loader=valid_loader, 

                                           loss_func=loss_func, 

                                           optimizer=optimizer,

                                           n_epochs=3, 

                                           saved_model='simple_nn.pt')
def predict(data_loader, model):

    model.eval()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    

    model.to(device)

    

    with torch.no_grad():

        predictions = None

        

        for i, batch in enumerate(tqdm(data_loader)):   

            

            output = model(batch['data'][0].to(device, 

                                               dtype=torch.long), 

                           batch['data'][1].to(device, 

                                               dtype=torch.float)).cpu().numpy()

            

            if i == 0:

                predictions = output

                

            else: 

                

                predictions = np.vstack((predictions, output))

                

    return predictions
model.load_state_dict(torch.load('simple_nn.pt'))



test_loader = DataLoader(test_dataset, 

                         batch_size=1000)



nn_predictions = predict(test_loader, model)
nn_predictions
nn_predictions_df = pd.DataFrame({'id': sample_submission['id'], 'target': nn_predictions.squeeze()})
nn_predictions_df.head()
from catboost import CatBoostClassifier
X = train.drop('target', axis=1)

y = train['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=27)



test_data = test.drop('target', axis=1)
cat_features = ['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4', 'nom_5', 'nom_6', 'nom_7',

                'nom_8', 'nom_9', 'ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4',

                'ord_5_1', 'ord_5_2', 'nan_features', 'day_0', 'day_1', 'day_2',

                'day_3', 'day_4', 'day_5', 'day_6', 'day_7', 'month_0', 'month_1',

                'month_2', 'month_3', 'month_4', 'month_5', 'month_6', 'month_7',

                'month_8', 'month_9', 'month_10', 'month_11', 'month_12', 'nom_0_0',

                'nom_0_1', 'nom_0_2', 'nom_0_3', 'nom_1_0', 'nom_1_1', 'nom_1_2',

                'nom_1_3', 'nom_1_4', 'nom_1_5', 'nom_1_6', 'nom_2_0', 'nom_2_1', 

                'nom_2_2', 'nom_2_3','nom_2_4', 'nom_2_5', 'nom_2_6', 'nom_3_0',

                'nom_3_1', 'nom_3_2','nom_3_3', 'nom_3_4', 'nom_3_5', 'nom_3_6',

                'nom_4_0', 'nom_4_1', 'nom_4_2', 'nom_4_3', 'nom_4_4']
best_params = {

    'bagging_temperature': 0.8, 

    'depth': 5, 

    'iterations': 1000,

    'l2_leaf_reg': 30,

    'learning_rate': 0.05,

    'random_strength': 0.8

}



model_cat = CatBoostClassifier(**best_params,

                               loss_function='Logloss',

                               eval_metric='AUC', 

                               nan_mode='Min',

                               thread_count=4,

                               task_type='GPU', 

                               verbose=True)



model_cat.fit(X_train, y_train,

              eval_set=(X_test, y_test), 

              cat_features=cat_features,

              verbose_eval=300, 

              early_stopping_rounds=500, 

              use_best_model=True,

              plot=False)



cat_predictions = model_cat.predict_proba(test_data)[:, 1]
cat_predictions_df = pd.DataFrame({'id': sample_submission['id'], 

                                   'target': cat_predictions})
cat_predictions_df.head()
nn_predictions_df.head()
res_sub = pd.DataFrame({'id': sample_submission['id']})

res_sub.head()
res_sub['target'] = round((cat_predictions_df['target'] + nn_predictions_df['target']) / 2, 2)
res_sub.head(5)
res_sub.to_csv('res_sub.csv', index=False)