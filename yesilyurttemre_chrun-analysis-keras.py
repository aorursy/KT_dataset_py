import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import pickle

import sys

from sklearn import metrics



if not sys.warnoptions:

    import warnings

    warnings.simplefilter("ignore")

      

        

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import plotly.figure_factory as ff

dataset = pd.read_csv('/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')

dataset.head()
dataset.info()
lab = dataset["Churn"].value_counts().keys().tolist()

val = dataset["Churn"].value_counts().values.tolist()



trace = go.Pie(labels = lab ,

               values = val ,

               marker = dict(colors =  [ 'royalblue' ,'lime'],

                             line = dict(color = "white",

                                         width =  1.3)

                            ),

               rotation = 90,

               hoverinfo = "label+value+text",

               hole = .5

              )

layout = go.Layout(dict(title = "Customer attrition in data",

                        plot_bgcolor  = "rgb(243,243,243)",

                        paper_bgcolor = "rgb(243,243,243)",

                       )

                  )



data = [trace]

fig = go.Figure(data = data, layout = layout)

py.iplot(fig)
sns.countplot(x= 'Churn', data = dataset)

for i in range(len(dataset['TotalCharges'])):

    if dataset.iloc[i,19] == ' ':

        print(i)
dataset['TotalCharges'] = dataset['TotalCharges'].replace(' ', np.nan)



dataset = dataset[dataset['TotalCharges'].notnull()]

dataset = dataset.reset_index()[dataset.columns]



dataset['TotalCharges'] = dataset['TotalCharges'].astype(float)



replace_columns = [ 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection','TechSupport','StreamingTV', 'StreamingMovies']



for i in replace_columns:

    dataset[i] = dataset[i].replace({'No internet service' : 'No'})

    

dataset['MultipleLines'] = dataset['MultipleLines'].replace({'No phone service' : 'No'})
print ("Rows     : " ,dataset.shape[0])

print ("Columns  : " ,dataset.shape[1])

print ("\nFeatures : \n" ,dataset.columns.tolist())

print ("\nMissing values :  ", dataset.isnull().sum().values.sum())

print ("\nUnique values :  \n",dataset.nunique())





print(dataset['Contract'].unique())

print(dataset['PaymentMethod'].unique())

print(dataset['InternetService'].unique())
slice_df = pd.concat([dataset.iloc[:,1:8],dataset.iloc[:,9:15], dataset.iloc[:,16:17], dataset.iloc[:,18:]], axis = 1)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()



encode_columns = [ 'gender', 'Partner', 'Dependents','PhoneService','MultipleLines','OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'Churn']



for i in encode_columns:

    slice_df[i] = le.fit_transform(slice_df[i])



churn = slice_df.iloc[:,-1:]

slice_df = slice_df.iloc[:,:-1]





internet_service = dataset.iloc[:,8:9]

contract = dataset.iloc[:,-6:-5]

payment_method = dataset.iloc[:,-4:-3]

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder()

internet_service = ohe.fit_transform(internet_service).toarray()

contract = ohe.fit_transform(contract).toarray()

payment_method = ohe.fit_transform(payment_method).toarray()
internet_service = pd.DataFrame(data = internet_service, index = range(len(internet_service)), columns = ['DSL','Fiber optic', 'No internet service'])

contract = pd.DataFrame(data = contract, index = range(len(contract)), columns = ['Month-to-month', 'One year', 'Two year'])

payment_method = pd.DataFrame(data = payment_method, index = range(len(payment_method)), columns = ['Bank transfer (automatic)', 'Credit card (automatic)', 'Electronic check', 'Mailed check'])



X = pd.concat([slice_df, internet_service, contract, payment_method], axis = 1)

X = X.values

Y = churn.values
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(X, Y,test_size=0.2, random_state=0)
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler()

X_train = sc.fit_transform(x_train)

X_test = sc.fit_transform(x_test)
import keras



from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout







classifier = Sequential(layers = None, name = None)



classifier.add(Dense(128, activation = 'tanh')) 

classifier.add(Dense(256, activation = 'tanh'))  

classifier.add(Dense(512, activation = 'tanh'))

classifier.add(Dense(1024, activation = 'tanh'))

classifier.add(Dense(1, activation = 'sigmoid'))



classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

            

            

classifier.fit(X_train, y_train, epochs = 50)





y_pred = classifier.predict(X_test, use_multiprocessing=True, max_queue_size=1)

y_pred = (y_pred > 0.5)
from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report



print(classification_report(y_test, y_pred))

categories = ['Churn', 'Not Churn']

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm/np.sum(cm), cmap = 'Blues', fmt = '.2%', annot = True, xticklabels = categories, yticklabels = categories)