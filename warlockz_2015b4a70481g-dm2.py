import pandas as pd

import numpy as np

from sklearn.metrics import roc_auc_score

from sklearn.utils.class_weight import compute_class_weight

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score

from sklearn.decomposition import PCA

pd.options.mode.use_inf_as_na = True
data = pd.read_csv('../input/train.csv')

data_test = pd.read_csv('../input/test.csv')
df = data

df_test = data_test
df['Hispanic'].replace('?','HA',inplace=True)

df['COB FATHER'].replace('?','c24',inplace=True)

df['COB MOTHER'].replace('?','c24',inplace=True)

df['COB SELF'].replace('?','c24',inplace=True)



df_test['Hispanic'].replace('?','HA',inplace=True)

df_test['COB FATHER'].replace('?','c24',inplace=True)

df_test['COB MOTHER'].replace('?','c24',inplace=True)

df_test['COB SELF'].replace('?','c24',inplace=True)



bins = [0, 20, 40, 60, 80, 100]

labels = [0,1,2,3,4]

df['binned_age'] = pd.cut(df['Age'], bins=bins, labels=labels)

df_test['binned_age'] = pd.cut(df_test['Age'], bins=bins, labels=labels)
to_drop = []

to_drop.append('Worker Class')

to_drop.append('Enrolled')

to_drop.append('MIC')

to_drop.append('MOC')

to_drop.append('MLU')

to_drop.append('Reason')

to_drop.append('Area')

to_drop.append('State')

to_drop.append('MSA')

to_drop.append('REG')

to_drop.append('MOVE')

to_drop.append('Live')

to_drop.append('PREV')

to_drop.append('Teen')

to_drop.append('Fill')

to_drop.append('Age')



# to_drop.append('Weight')

to_drop.append('ID')

to_drop.append('Detailed')



df = df.drop(to_drop,axis=1)

df_test = df_test.drop(to_drop,axis=1)


DAT_test = pd.get_dummies(df_test,['Schooling','Married_Life','Cast','Hispanic','Sex','Full/Part','Tax Status','Summary','COB FATHER','COB MOTHER','COB SELF','Citizen'],columns=['Schooling','Married_Life','Cast','Hispanic','Sex','Full/Part','Tax Status','Summary','COB FATHER','COB MOTHER','COB SELF','Citizen'])

DAT = pd.get_dummies(df,['Schooling','Married_Life','Cast','Hispanic','Sex','Full/Part','Tax Status','Summary','COB FATHER','COB MOTHER','COB SELF','Citizen'],columns=['Schooling','Married_Life','Cast','Hispanic','Sex','Full/Part','Tax Status','Summary','COB FATHER','COB MOTHER','COB SELF','Citizen'])

DAT['binned_age'] = DAT['binned_age'].cat.codes

DAT_test['binned_age'] = DAT_test['binned_age'].cat.codes

df = DAT

df_test = DAT_test





X = df.drop('Class',axis=1)

Y = df['Class']



from sklearn.ensemble import AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier





ada = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1,class_weight='balanced'),n_estimators=300,random_state=47)

ada.fit(X=X,y=Y)
preds = ada.predict(df_test)

indexes = data_test['ID']

final_df = pd.DataFrame(columns=['ID','Class'])

final_df['ID'] = data_test['ID']

final_df['Class']=preds

final_df.head()

from IPython.display import HTML

import base64



def create_download_link(df, title = "Download CSV file", filename = "dataset.csv"):

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}"target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



create_download_link(final_df)