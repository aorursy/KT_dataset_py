import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import StandardScaler



from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from sklearn import neighbors

from sklearn.grid_search import GridSearchCV
train_df = pd.read_csv('../input/train.csv')

train_df.info()

copy_df = train_df.copy()

species = train_df['species'].unique()

species.sort()

spe_dict = dict(enumerate(species))

inv_spe_dict = {v: k for k,v in spe_dict.items()}

train_df['species_index'] = train_df['species'].map(inv_spe_dict).astype(int)

train_df = train_df.drop(['species','id'],axis = 1)
train_data = train_df.values

train_df
X,y = train_data[:700,:-1],train_data[:700,-1]

scaler = StandardScaler().fit(X)

X = scaler.transform(X)

model = MLPClassifier(hidden_layer_sizes=(150,),activation='logistic',solver='lbfgs',alpha=0.003

                      ,max_iter=200,early_stopping=True,validation_fraction=0.2,

                      learning_rate='adaptive',tol=1e-8,random_state=1).fit(X,y)
def cv(a,b,model):

    cv_X = train_data[a:b,:-1]

    cv_X = scaler.transform(cv_X)

    cv_y = train_data[a:b,-1]

    cv_p1_y = model.predict(cv_X)

    cv_p2_y = model.predict_log_proba(cv_X)

    print(accuracy_score(cv_y,cv_p1_y))

    print(cv_y)

    print(cv_p1_y)

#print(cv_p2_y)
cv(701,990,model)
test_df = pd.read_csv('../input/test.csv')

index = test_df.pop('id')

test_data = test_df.values



test_X = test_data

test_X = scaler.transform(test_X)
predict = model.predict(test_X)

predict_proba = model.predict_proba(test_X)

predict_proba
#print(species.tolist())

species_list = species.tolist()

result = pd.DataFrame(predict_proba,index = index, columns = species_list)

result
