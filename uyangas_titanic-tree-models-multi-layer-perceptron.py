import pandas as pd

import numpy

from datetime import datetime

from sklearn.metrics import accuracy_score



gender_submission = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")

test = pd.read_csv("/kaggle/input/titanic/test.csv")

train = pd.read_csv("/kaggle/input/titanic/train.csv")
print(f"Training set dimension: {train.shape}")

print(f"Testing set dimension: {test.shape}")
import pandas as pd

import numpy as np

import pickle

import os

from sklearn.preprocessing import OneHotEncoder

from sklearn.base import TransformerMixin, BaseEstimator

from sklearn.model_selection import GridSearchCV



print("Fill na and set index: FillSet")

def FillNA(X, ind, cat, num, col='Cabin'):



    X = X.set_index(ind)

    X[cat] = X[cat].fillna('Missing')

    X[num] = X[num].fillna(X[num].median())

    X[col] = X[col].apply(lambda x: x[0])



    return X



print("Title function: Title")

def Title(df, col):

    title = []

    for i in df[col]:

        if 'Miss.' in i:

            title.append('Miss')

        elif 'Mrs.' in i:

            title.append('Mrs')

        elif 'Mr.' in i:

            title.append('Mr')

        else:

            title.append('No')



    df[col] = title



    return df





print("One hot decoder imported: Decoder")

class Decoder(TransformerMixin, BaseEstimator):

    def __init__(self, columns):

        super().__init__()



        self.columns = columns

        self.onehot = OneHotEncoder(drop='first')



    def fit(self, X, y=None):

        return self



    def fit_transform(self, X):

        one = self.onehot.fit_transform(X[self.columns]).toarray()

        col_names = self.onehot.get_feature_names()



        return pd.concat([X.drop(self.columns, axis=1), pd.DataFrame(one, index=X.index, columns=col_names)], axis=1)





print("Select imported: SelectCol")

def SelectCol(X, drop_cols, target, col_str):

    if col_str is not None:

        for i in col_str:

            X.apply(lambda i: str(i))



    X = X.drop(drop_cols, axis=1)



    train = X[~X.Survived.isna()]

    test = X[X.Survived.isna()]



    return train.drop(target, axis=1), train[target], test.drop(target, axis=1)







print("Grid Search train: GSCV")

def GSCV(pipe, params, X, y, test, submission, m, scoring, cv=5):

    grid = GridSearchCV(estimator=pipe,

                        param_grid=params,

                        cv=cv,

                        iid=False,

                        return_train_score=False,

                        refit=True,

                        scoring=scoring

                       )

    grid.fit(X, y)

    pd.DataFrame(grid.predict(test), index=submission.index, columns=['Survived']).to_csv("/kaggle/working/" + m + "_submission.csv")

    return grid.best_score_, grid.best_params_, grid.best_estimator_.predict(test)







print("Ensemble function: EnsemblePropensity")

def EnsemblePropensity(directory, folder):

    d = directory + folder

    count = 0

    df = pd.DataFrame()

    for i in os.listdir(d):

        if 'csv' in i:

            score = pd.read_csv(d + i)

            df = pd.concat([df, score.iloc[:, 1]], axis=1)

            count += 1

            index = score.iloc[:, 0]



    df = pd.DataFrame(np.sum(df, axis=1), columns=['Probability'])

    df['Survived'] = [1 if i >= 2 else 0 for i in df.Probability]

    df = df.iloc[:, 1]

    df.to_csv(d + "Ensemble_" + str(count) + ".csv")



    print("Ensemble complete")

test.insert(test.shape[1], "Survived", [np.nan]*len(test))

print(f"Train dim: {train.shape}, Test dim: {test.shape}")



df = pd.concat([train, test], sort=True)

print(f"DF dimension: {df.shape}")
print("<<<<< Original dataset >>>>>>")

print(f"Missing: {df.isna().sum()}")

df = FillNA(df, 'PassengerId',['Cabin', 'Embarked'], ['Age','Fare'])

print("<<<<< Filled NA >>>>>>")

print(f"Missing: {df.isna().sum()}")
df = Title(df, 'Name')

df.head()
df = Decoder(['Cabin', 'Embarked', 'Pclass', 'Sex', 'Name']).fit_transform(df)

print(f"New dimension: {df.shape}")
X, y, test_data = SelectCol(df, ['Ticket'], 'Survived', None)

X.shape, y.shape, test_data.shape
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb

import lightgbm as lgbm

import catboost as cb



model = []

accuracy = []

best_params = []

runtime = []

scores = []



scoring = 'f1_micro'

params = [ {'criterion': ['gini','entropy'],

            'max_depth':np.arange(1,20).tolist()},

          

          {'n_estimators': np.arange(5,30).tolist(),

            'criterion': ['gini','entropy'],

            'max_depth':np.arange(1,10).tolist()},

          

          {'learning_rate': [0.001, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3],

            'n_estimator': np.arange(5,25).tolist(),

            'max_depth':np.arange(3,15).tolist(),

             'alpha': [1, 2, 3]},

          

          {'learning_rate': [0.001, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3],

            'n_estimator': np.arange(5,25).tolist(),

            'num_leaves':np.arange(3,15).tolist(),

             'reg_alpha': [0.01, 0.02, 0.03]},

           

           {'depth': np.arange(5,25).tolist(),

             'learning_rate' : [0.01, 0.03, 0.1, 0.15, 0.3],

             'l2_leaf_reg': [1,3,4,9],

             'iterations': [100]} ]



models = ["dt", "rf", "xgb", "lgbm", "cb"]



estimators = [DecisionTreeClassifier(random_state=234),

              RandomForestClassifier(random_state=123),

              xgb.XGBClassifier(n_jobs=3),

              lgbm.LGBMClassifier(objective = 'binary'),

              cb.CatBoostClassifier(silent=True)]



print("Parameters and estimators are created")
# for i in range(len(models)):

#     est = estimators[i]

#     param = params[i]

#     m = models[i]

#     start = datetime.now()

    

#     score, parm, estimation = GSCV(pipe=est, 

#                                   params=param, 

#                                   X=X, 

#                                   y=y, 

#                                   test=test_data,

#                                   submission=gender_submission,

#                                   m = m,

#                                   scoring=scoring

#                                   )

#     end = datetime.now()

    

#     model.append(m + "_" + scoring)

#     accuracy.append(score)

#     best_params.append(parm)

#     runtime.append(end-start)

#     scores.append(scoring)

    

#     print(f"<<<< Model: {m} >>>>")

#     print(f"Train score: {score}")

#     print(f"Best parameters: {parm}")

#     print(f"Train Runtime: {end-start}")
# pd.DataFrame({'model': model,

#             'score': accuracy,

#             'best parameters': best_params,

#             'runtime': runtime,

#             'metric': scores})
estimators = [DecisionTreeClassifier(random_state=234, criterion='entropy', max_depth=4),

              RandomForestClassifier(random_state=123, criterion='entropy', max_depth=6, n_estimators=5),

              xgb.XGBClassifier(n_jobs=3, alpha=1, learning_rate=0.03, max_depth=8, n_estimators=5),

              lgbm.LGBMClassifier(objective = 'binary', learning_rate=0.1, n_estimator=5, num_leaves=11, reg_alpha=0.01)]



models = ["dt", "rf","xgb","lgbm"]

accuracy=[]



for i in range(len(estimators)):

    est = estimators[i]

    

    est.fit(X, y)

    y_pred = est.predict(X)

    

    accuracy.append(accuracy_score(y, y_pred))

    pd.DataFrame(est.predict(test_data), columns=['Survived']).to_csv("/kaggle/working/" + "1_" + models[i] + "_submission.csv")

    

    print(f"{models[i]} done")

    print(f"Accuracy {accuracy_score(y, y_pred)}")
pd.DataFrame({'model': models,

              "accuracy":accuracy

            })
import torch

from torch.autograd import Variable

from keras.utils import to_categorical



X = Variable(torch.from_numpy(X.values))

y = Variable(torch.from_numpy(y.values))

test_data = Variable(torch.from_numpy(test_data.values))
from torch import nn





class MLP(nn.Module):

    def __init__(self):

        super(MLP, self).__init__()

        

        self.fc1 = nn.Linear(21, 12)

        self.fc2 = nn.Linear(12, 6)

        self.fc3 = nn.Linear(6, 2)

        

        self.dropout = nn.Dropout(p=0.02)

        

        self.relu = nn.ReLU()

        self.logsoftmax = nn.LogSoftmax(dim=1)

        

    def forward(self, x):

        x = self.dropout(self.relu(self.fc1(x)))

        x = self.dropout(self.relu(self.fc2(x)))

        x = self.logsoftmax(self.fc3(x))

        

        return x



model = MLP()

model
from torch import optim



optimizer = optim.SGD(model.parameters(), lr=0.01)

criterion = nn.NLLLoss()

epoch = 100



permutation = torch.randperm(X.size()[0])

batch_size = 33



for e in range(epoch):

    train_loss = 0

    train_accuracy = 0

    

    for i in range(0,X.size()[0],batch_size):

        indices = permutation[i:i+batch_size]

        batch_x, batch_y = X[indices], y[indices]

        

        optimizer.zero_grad()



        logps = model(batch_x.float())

        loss = criterion(logps, batch_y.long())

        train_loss = loss



        ps = torch.exp(logps)

        top_ps, top_class = ps.topk(1, dim=1)

        equals = top_class == batch_y.view(*top_class.shape)



        train_accuracy += torch.mean(equals.type(torch.FloatTensor))



        loss.backward()

        optimizer.step()

    

    else:

        

        with torch.no_grad():



            model.eval()



            logps = model(test_data.float())

            ps = torch.exp(logps)

            top_ps, top_class = ps.topk(1, dim=1)



            pd.DataFrame(np.array(top_class), columns=['Survived']).to_csv("/kaggle/working/" + str((e+1)) + "_mlp_submission.csv")



            model.train()

    

    print("<<< Epoch: {} >>>".format(e+1), "Train loss: {:3f}".format(train_loss), "Train accuracy: {:3f}".format(train_accuracy*100/(len(X)/batch_size)))