import numpy as np

import pandas as pd

data = pd.read_csv('/kaggle/input/congressional-voting-records/house-votes-84.csv')

data.head()
ynmap = {'y':1,'n':0,'?':np.nan}

partymap = {'republican':1,'democrat':0}

data['republican'] = data['Class Name'].map(partymap)

data.drop('Class Name',axis=1,inplace=True)

for column in data.columns.drop('republican'):

    data[column+'1'] = data[column].map(ynmap)

    data.drop(column,axis=1,inplace=True)

data_col = data.columns

data.head()
#impute missing values

from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)

data = pd.DataFrame(imputer.fit_transform(data),columns=data_col)
data.head()
import sklearn

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(data.drop('republican',axis=1),data['republican'],test_size=0.35)
from sklearn.linear_model import LogisticRegression

log = LogisticRegression()

log.fit(X_train,y_train)
(log.predict(X_test) - y_test).apply(abs).mean()
for k_value in range(1,30):

    import numpy as np

    import pandas as pd

    data = pd.read_csv('/kaggle/input/congressional-voting-records/house-votes-84.csv')

    

    ynmap = {'y':1,'n':0,'?':np.nan}

    data['republican'] = data['Class Name'].map(partymap)

    data.drop('Class Name',axis=1,inplace=True)

    for column in data.columns.drop('republican'):

        data[column+'1'] = data[column].map(ynmap)

        data.drop(column,axis=1,inplace=True)

    partymap = {'republican':1,'democrat':0}

    data_col = data.columns

    

    #impute missing values

    from sklearn.impute import KNNImputer

    imputer = KNNImputer(n_neighbors=k_value)

    data = pd.DataFrame(imputer.fit_transform(data),columns=data_col)

    

    from sklearn.linear_model import LogisticRegression

    log = LogisticRegression()

    log.fit(X_train,y_train)

    print('K-Value: '+str(k_value)+" | Absolute Mean Error: "+str((log.predict(X_test) - y_test).apply(abs).mean()))