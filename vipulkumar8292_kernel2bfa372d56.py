import pandas as pd

import numpy as np
corona = pd.read_csv('../input/novel-corona-virus-2019-dataset/COVID19_line_list_data.csv')
corona.tail()
corona.drop(columns=['id','reporting date','location','country','Unnamed: 3','summary','hosp_visit_date','exposure_start','exposure_end','source','If_onset_approximated','symptom_onset','reporting date','symptom','link','Unnamed: 21','Unnamed: 22','Unnamed: 23','Unnamed: 24','Unnamed: 25','Unnamed: 26','recovered'],inplace=True)
corona.tail()
corona.shape
corona.isnull().sum()
corona.dropna(inplace=True)
corona.shape
corona.isnull().sum()
corona[['gender','age']].replace('NaN',0.0, inplace=True)
corona.tail()
corona.shape
from sklearn.preprocessing import LabelEncoder

label_enc = LabelEncoder()

df = pd.DataFrame(corona['gender'])
corona['gender']=df.apply(label_enc.fit_transform)
corona.loc[corona['death']!= '0'] = 1
corona['death'] = corona['death'].astype('float64')
x = corona[['case_in_country','gender','age','visiting Wuhan','from Wuhan']]

y = corona['death']
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.4)
from sklearn.preprocessing import StandardScaler

scale = StandardScaler()

scale.fit(x_train)

x_train_1=scale.transform(x_train)

x_test_1=scale.transform(x_test)
from sklearn.neural_network import MLPClassifier

model_mlp = MLPClassifier(hidden_layer_sizes=(5,4),max_iter = 1000,activation='relu')

model_mlp.fit(x_train,y_train)
prediction=model_mlp.predict(x_test)
from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test,prediction))

print(classification_report(y_test,prediction))