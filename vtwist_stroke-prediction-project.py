import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import MaxAbsScaler, LabelEncoder 

from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.metrics import accuracy_score



data = pd.read_csv("../input/stroke_prediction.csv") 

data.head(20)
data.columns = map(str.lower, data.columns)      
data.info()
gender = {"Female":0, "Male":1, "Other":2}

ever_married = {"No": 0, "Yes":1}

type_of_work = {"children":1, "Never_worked":2, "Govt_job":3, "Private":4, "Self-employed":5}

residence = {"Rural":0, "Urban":1}

smoking_status = {"never smoked":0, "formerly smoked":1, "smokes":2}





data_n = data.replace({"gender": gender, 

                      "ever_married": ever_married,

                      "type_of_work": type_of_work,

                      "residence": residence,

                      "smoking_status": smoking_status

                     })



data_n = data_n.astype('float64')

data_n.head(20)

data_n.describe()
np.sum(data_n < 0)
data_n = data_n.abs()
def days_to_years(days):

    return np.float64(round(days/365.2422))



data_s = data_n.rename(columns={"age_in_days": "age"})



data_s['age'] = data_n['age_in_days'].apply(days_to_years)



data_s.head()
data_s.isnull().sum()/len(data)*100
data_s = data_s.interpolate(method='linear', limit_direction='both')

data_s.smoking_status = data_s.smoking_status.round()



data_s.head(20)

data_s.info()
sns.countplot(data_s['stroke'])

data_s['stroke'].value_counts()

data_s.shape
count_class_0, count_class_1 = data_s['stroke'].value_counts()



data_class_0 = data_s[data_s['stroke'] == 0]

data_class_1 = data_s[data_s['stroke'] == 1]



data_class_0_under = data_class_0.sample(count_class_1)

data_under = pd.concat([data_class_0_under, data_class_1], axis=0)

data_u = data_under
sns.countplot(data_u['stroke'])

data_u['stroke'].value_counts()
X_train,X_test,y_train,y_test = train_test_split(data_u.drop('stroke', axis=1), data_u['stroke'], test_size = 0.3, random_state = 35)



ms = MaxAbsScaler()



X_train = ms.fit_transform(X_train)

X_test = ms.transform(X_test)



rfc = RandomForestClassifier(n_estimators=200)

rfc.fit(X_train,y_train)



pred_rfc = rfc.predict(X_test)
print(classification_report(y_test, pred_rfc))

print(confusion_matrix(y_test, pred_rfc))
rfc_as = accuracy_score(y_test, pred_rfc)

rfc_as
importances = rfc.feature_importances_

std = np.std([tree.feature_importances_ for tree in rfc.estimators_],

             axis=0)

indices = np.argsort(importances)[::-1]

col_name = data_u.drop('stroke', axis=1).columns.values

# Plot the feature importances of the forest



fig = plt.figure()

plt.title("Feature importances")

plt.bar(range(X_train.shape[1]),

        importances[indices],

        yerr=std[indices],

        align="center")

plt.xticks(range(X_train.shape[1]), np.array(col_name)[indices])

plt.xlim([-1, X_train.shape[1]])

fig.autofmt_xdate()

plt.show()
mlpc = MLPClassifier(hidden_layer_sizes=(13,13,13), max_iter=1000)

mlpc.fit(X_train, y_train)

pred_mlpc = mlpc.predict(X_test)
print(classification_report(y_test, pred_mlpc))

print(confusion_matrix(y_test, pred_mlpc))
nn_as = accuracy_score(y_test, pred_mlpc)

nn_as
data_t = ms.fit_transform(data_s.drop('stroke', axis=1))

pred = mlpc.predict(data_t)
prediction = pd.DataFrame(pred,columns=['prediction'])
prediction['prediction'].value_counts()
print(f"Current algorithm predicted {int(prediction.prediction.sum())} strokes out of {prediction.shape[0]} patients")