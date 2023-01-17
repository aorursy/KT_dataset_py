import numpy as np

import pandas as pd

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.metrics import roc_auc_score

mal=pd.read_csv('../input/opcode_frequency_malware.csv')

benign=pd.read_csv('../input/opcode_frequency_benign.csv')

test = pd.read_csv('../input/Test_data.csv')
test_filenames = test['FileName']

test = test.drop(['Unnamed: 1809','FileName','1808'],axis=1)

benign['1808']=0

mal['1808']=1

data=[mal,benign]

data_final=pd.concat(data)
data_final=data_final.drop_duplicates()
data_final=data_final.drop(columns=['FileName'],axis=1)
X=data_final.drop(columns=['1808'],axis=1)

y=data_final['1808']
ada_1 = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=13)).fit(X,y)

print(roc_auc_score(ada_1.predict(X),y))
preds = ada_1.predict(test)

final_df = pd.DataFrame(columns=['FileName','Class'])

final_df['FileName'] = test_filenames

final_df['Class'] = preds

final_df.head()

# final_df.to_csv('2015B4A70481G_ada.csv',index=False)
from IPython.display import HTML

import base64

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}"target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)

create_download_link(final_df)