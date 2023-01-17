import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

pd.options.mode.use_inf_as_na = True
data = pd.read_csv("../input/dataset.csv", sep=',')

X_final=data[['id']].iloc[175:1031]

y=data['Class'].iloc[0:175]

data.head()
data.replace('?',np.NaN,inplace=True)

data['Account1'].fillna('ad',inplace=True)

data['History'].fillna('c2',inplace=True)

data['Age'].fillna(0,inplace=True)

data['Motive'].fillna('p3',inplace=True)

data['Credit1'].fillna(0,inplace=True)

data['Monthly Period'].fillna(24,inplace=True)

data['InstallmentRate'].fillna(4,inplace=True)

data['Tenancy Period'].fillna(4,inplace=True)

data['InstallmentCredit'] = data['InstallmentCredit'].astype('float')

data['Yearly Period'] = data['Yearly Period'].astype('float')

data['InstallmentCredit'].fillna(data['InstallmentCredit'].mean(),inplace=True)

data['Yearly Period'].fillna(data['Yearly Period'].mean(),inplace=True)

# data['Monthly Period'].value_counts()
for i, row in data.iterrows():

    if row['Plotsize']=='la':

        data.at[i,'Plotsize'] = 'LA'

    if row['Plotsize']=='sm':

        data.at[i,'Plotsize'] = 'SM'

    if row['Plotsize']=='me' or row['Plotsize']=='M.E.':

        data.at[i,'Plotsize'] = 'ME'



for i, row in data.iterrows():

    if row['Housing']=='H2':

        data.at[i,'Housing'] = '1'

    if row['Housing']=='H1':

        data.at[i,'Housing'] = '2'

    if row['Housing']=='H3':

        data.at[i,'Housing'] = '3'

        

for i, row in data.iterrows():

    if row['Phone']=='yes':

        data.at[i,'Phone'] = '1'

    if row['Phone']=='no':

        data.at[i,'Phone'] = '0'

        

for i, row in data.iterrows():

    if row['Account2']=='Sacc4':

        data.at[i,'Account2'] = 'sacc4'

        

for i, row in data.iterrows():

    if row['Sponsors']=='g1':

        data.at[i,'Sponsors'] = 'G1'
data['#Credits']=data['#Credits'].astype(str)

data['Expatriate']=data['Expatriate'].astype(str)
# datax = data.drop(['id','Credit1','Age','Yearly Period','InstallmentCredit','Class'],axis=1)

datax = data.drop(['id','Class','Gender&Type','Monthly Period'],axis=1)

datax.columns

X=pd.get_dummies(datax,['Account1','Motive', 'History', 'Account2', 'Employment Period',

       'Sponsors', 'Plotsize', 'Plan','#Credits', 'Post', 'Phone', 'Expatriate'],columns=['Account1','Motive','History', 'Account2', 'Employment Period',

       'Sponsors', 'Plotsize', 'Plan','#Credits', 'Post', 'Phone', 'Expatriate'])

# X = pd.get_dummies(datax,datax.columns)

X.head()
from sklearn import preprocessing



x = X.values #returns a numpy array

X_normalized = np.zeros_like(x)



X_T = np.transpose(x)

min_max_scaler = preprocessing.Normalizer()

X_normalized = min_max_scaler.fit_transform(X_T).T

# print(x_scaled_column.mean(), x_scaled_column.var())

# X_normalized[:,column_index] = x_scaled_column[:,0]

    

print(X_normalized.shape)

from sklearn.decomposition import PCA

pca = PCA(n_components=8)

X_new = pca.fit_transform(X_normalized)
from sklearn.cluster import KMeans, SpectralClustering, Birch

from sklearn.metrics import accuracy_score

preds1=[]

spec = SpectralClustering(n_clusters = 3, affinity='sigmoid')

spec.fit(X_new)



pred = spec.labels_



preds1.append(pred)



print(np.unique(preds1, return_counts=True))
print(np.unique(preds1, return_counts=True))
maxa = 0

ans = []

stra = ''
y_pred1=[]

for i in range(len(preds1[0])):

    y_pred1.append(preds1[0][i])

unique, counts=np.unique(y_pred1, return_counts=True)

counts
if accuracy_score(y,y_pred1[0:175])>maxa:

    ans = y_pred1

    maxa = accuracy_score(y,y_pred1[0:175])

    stra='1'
y_pred2=[]

for i in range(len(y_pred1)):

    if y_pred1[i]==0:

        y_pred2.append(1)

    elif y_pred1[i]==1:

        y_pred2.append(0)

    else:

        y_pred2.append(2)
len(y_pred2)

unique, counts=np.unique(y_pred2, return_counts=True)

counts



if accuracy_score(y,y_pred2[0:175])>maxa:

    ans = y_pred2

    maxa = accuracy_score(y,y_pred2[0:175])

    stra='2'
y_pred3=[]

for i in range(len(y_pred1)):

    if y_pred1[i]==0:

        y_pred3.append(2)

    elif y_pred1[i]==1:

        y_pred3.append(1)

    else:

        y_pred3.append(0)
len(y_pred3)

unique, counts=np.unique(y_pred3, return_counts=True)

counts

if accuracy_score(y,y_pred3[0:175])>maxa:

    ans = y_pred3

    maxa = accuracy_score(y,y_pred3[0:175])

    stra='3'
y_pred4=[]

for i in range(len(y_pred1)):

    if y_pred1[i]==0:

        y_pred4.append(0)

    elif y_pred1[i]==1:

        y_pred4.append(2)

    else:

        y_pred4.append(1)
len(y_pred4)

unique, counts=np.unique(y_pred4, return_counts=True)

counts

if accuracy_score(y,y_pred4[0:175])>maxa:

    ans = y_pred4

    maxa = accuracy_score(y,y_pred4[0:175])

    stra='4'
y_pred5=[]

for i in range(len(y_pred1)):

    if y_pred1[i]==0:

        y_pred5.append(1)

    elif y_pred1[i]==1:

        y_pred5.append(2)

    else:

        y_pred5.append(0)
len(y_pred5)

unique, counts=np.unique(y_pred5, return_counts=True)

counts

if accuracy_score(y,y_pred5[0:175])>maxa:

    ans = y_pred5

    maxa = accuracy_score(y,y_pred5[0:175])

    stra='5'
y_pred6=[]

for i in range(len(y_pred1)):

    if y_pred1[i]==0:

        y_pred6.append(2)

    elif y_pred1[i]==1:

        y_pred6.append(0)

    else:

        y_pred6.append(1)

        
len(y_pred6)

unique, counts=np.unique(y_pred6, return_counts=True)

counts

if accuracy_score(y,y_pred6[0:175])>maxa:

    ans = y_pred6

    maxa = accuracy_score(y,y_pred6[0:175])

    stra='6'
X_final.head()
X_final['Class']=ans[175:1031]
X_final.to_csv("5.csv",index=False)

maxa
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



create_download_link(X_final)