import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.utils import resample

from IPython.display import HTML

import base64
test_df = pd.read_csv("../input/data-mining-assignment-2/test.csv", sep=',')
train_df = pd.read_csv("../input/data-mining-assignment-2/train.csv", sep=',')
train_test_df=pd.concat([train_df,test_df])
new_df=pd.DataFrame(pd.get_dummies(train_test_df))
train=new_df.loc[:699, ].drop(['ID'],axis=1)

test=new_df[np.isnan(new_df['Class'])].drop(['ID','Class'], axis=1)
y=train['Class']

X=train.drop(['Class'],axis=1)

X.head()
np.random.seed(42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state = 42)



print('Training Features Shape:', X_train.shape)

print('Training Labels Shape:', y_train.shape)

print('Testing Features Shape:', X_test.shape)

print('Testing Labels Shape:', y_test.shape)
rf = RandomForestClassifier(n_estimators=200, max_depth = 9, random_state = 42, n_jobs = -1)

rf.fit(X_train, y_train)

rf.score(X_test,y_test)
features = test.columns

prediction = rf.predict(test[features])
output = pd.DataFrame(test_df['ID'])

output['Class'] = prediction.astype(int)

output
def create_download_link(df, title = "Download CSV file", filename = "data.csv"):

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)

create_download_link(output)