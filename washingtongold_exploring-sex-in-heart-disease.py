import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import pandas as pd

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
print(os.listdir("../input"))

data = pd.read_csv('../input/heart-disease-uci/heart.csv')
data = data.rename(columns={'cp':'chest pain type','trestbps':'resting blood pressure',

                    'chol':'serum cholestrol','fbs':'fasting blood sugar',

                    'restecg':'resting electro','thalach':'max heart rate',

                    'exang':'exercise induced angina','oldpeak':'ST depression',

                    'ca':'colored vessels'})
data.head()
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(data.drop('target',axis=1),data['target']

                                                 ,test_size=0.3)

from sklearn.linear_model import LogisticRegression

log = LogisticRegression()

log.fit(X_train,y_train)
fig = plt.figure(figsize=(10,4))

sns.barplot(x = abs(log.coef_[0]),y=data.columns.drop('target'),palette = 'viridis')

plt.tight_layout()
from sklearn import preprocessing

scaler = preprocessing.StandardScaler()

standard = pd.DataFrame(scaler.fit_transform(data))

standard.head()
create_dic = {}

columns = list(data.columns)

for i in range(len(columns)):

    create_dic[columns[i]] = standard.loc[i]
sdata = pd.DataFrame(create_dic)

sdata.head()
sX_train,sX_test,sy_train,sy_test = train_test_split(data.drop('target',axis=1),

                                                 data['target'],test_size=0.3)

slog = LogisticRegression()

slog.fit(sX_train,sy_train)
fig = plt.figure(figsize=(10,4))

sns.barplot(x = abs(slog.coef_[0]),y=data.columns.drop('target'),palette = 'viridis')

plt.tight_layout()
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test,log.predict(X_test)))

print(confusion_matrix(y_test,log.predict(X_test)))
coeffs = pd.Series(log.coef_[0],list(data.columns[:-1]))

coeffs['sex']
def run_interface():

    age = float(input('Age: '))

    cpt = float(input('Chest pain type: '))

    rbp = float(input('Resting blood pressure: '))

    sc = float(input("Serum cholestrol: "))

    fbs = float(input("Fasting blood sugar > 120 md/dl: "))

    res = float(input("Resting electrocardiographic results: "))

    mhra = float(input("Maximum heart rate achieved: "))

    eia = float(input("Exercise induced agina: "))

    op = float(input("ST depression induced by exercise: "))

    spe = float(input("Slope of peak exercise ST segment: "))

    mv = float(input("# of major vesels colored by fluoroscopy: "))

    thal = float(input("Thal value: "))

    

    sex_vals = [1.0,0.0]

    pred_vals = []

    

    for sex in sex_vals:

        pred = log.predict_proba(np.array([age,sex,cpt,rbp,sc,fbs,res,mhra,eia,

                                          op,spe,mv,thal]).reshape(1,-1))

        pred_vals.append(pred[0][1])

        

    print("_____________________________________________________________________\n")

    print("Probabilities indicate the chance of having heart disease.")

    print("All variables were kept the same, except for sex.")

    print("Male: {}".format(pred_vals[0]))

    print("Female: {}".format(pred_vals[1]))

    print("\nHolding all variables fixed, females have a {} times higher chance of having heart disease than males.".format(pred_vals[1]/pred_vals[0]))
X_train.iloc[34]
run_interface()
X_train.iloc[41]
run_interface()
values = []

sex_vals = [1.0,0.0]



for index in range(len(data)):

    

    pred_vals = []

    

    for sex in sex_vals:

        

        array = []

        array.append(data.iloc[index]['age'])

        array.append(sex)

        for column in data.iloc[index].drop(['age','sex','target']):

            array.append(column)

        

        pred = log.predict_proba(np.array(array).reshape(1,-1))

        pred_vals.append(pred[0][1])

    

    values.append(pred_vals[1]/pred_vals[0])
sns.distplot(values,bins=20)

plt.title("Distribution of X Value Across UCI Heart Disease Dataset")

plt.xlabel('X Value (Female chance divided by Male chance of heart disease)')