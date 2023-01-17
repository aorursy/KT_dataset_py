#Importing libraries

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
#Read and check the data

df = pd.read_csv('../input/heart-disease-uci/heart.csv')
df.head()
print('Number of rows and columns :',df.shape)

df.describe()
df.isna().sum()
df.info()
#Feature correlation

plt.figure(figsize=(20,10))

sns.heatmap(df.corr(),annot = True,cmap="RdYlGn")

plt.title('Heatmap for the Dataset')
#calculate how many are having heart disease and not heart having disease

target_value = df.target.value_counts()

print('Number of patients have heart disease:{}'.format(target_value[1]))

print("Number of patients haven't heart disease:{}".format(target_value[0]))
sns.countplot(x='target',data=df)
#checking the age of patients

sns.kdeplot(df.age)

plt.xlabel('Age')
#Age vs target

df.groupby(df['age']).target.value_counts().unstack().plot(kind = 'bar',figsize=(20,6))

plt.title('Heart Disease Frequency for Ages')

plt.xlabel('Age')

plt.ylabel('Frequency')

plt.legend(["Haven't Disease",'Have Disease'])
LABELS=['Female','Male']

sns.countplot(x = 'sex',data=df)

plt.xticks(range(2), LABELS)
LABELS=['Female','Male']

df.groupby(df['sex']).target.value_counts().unstack().plot(kind = 'bar')

plt.title('Heart Disease Frequency for Sex')

plt.xticks(range(2), LABELS,rotation=360)

plt.legend(["Haven't Disease",'Have Disease'])
df.groupby(df['cp']).target.value_counts().unstack().plot(kind = 'bar')

plt.legend(["Haven't Disease",'Have Disease'])

plt.xlabel('Chest pain')

plt.ylabel('Frequency')
sns.boxplot(df['target'],df['trestbps'],palette = 'viridis')

plt.title('Relation between trestbps and target')
sns.boxplot(df['target'],df['chol'],palette = 'viridis')

plt.title('Relation between Cholestrol and Target')
df.groupby(df['target']).restecg.value_counts().unstack().plot(kind = 'bar',color = plt.cm.rainbow(np.linspace(0, 1, 3)))

plt.title('Relation between ECG and Target')
sns.boxplot(df['target'],df['slope'],palette = 'viridis')

plt.title('Relation between Slope and Target')
sns.boxenplot(df['target'], df['ca'], palette = 'Reds')

plt.title('Relation between number of major vessels and target')
sns.boxplot(df['target'], df['thalach'], palette = 'Reds')

plt.title('Relation between heart rate and target')
# let's change the names of the  columns for better understanding



df.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate_achieved',

       'exercise_induced_angina', 'st_depression', 'st_slope', 'num_major_vessels', 'thalassemia', 'target']



df.columns
df['sex']=df['sex'].map({0:'female',1:'male'})

df['chest_pain_type']=df['chest_pain_type'].map({0:'typical angina',1:'atypical angina',2:'non-anginal pain',3:'asymptomatic'})

df['fasting_blood_sugar']=df['fasting_blood_sugar'].map({0:'lower than 120mg/ml',1:'greater than 120mg/ml'})

df['rest_ecg']=df['rest_ecg'].map({0:'normal',1:'ST-T wave abnormality',2:'left ventricular hypertrophy'})

df['exercise_induced_angina']=df['exercise_induced_angina'].map({0:'no',1:'yes'})

df['st_slope']=df['st_slope'].map({0:'upsloping',1:'flat',2:'downsloping'})

df['thalassemia']=df['thalassemia'].map({0:'0',1:'normal',2:'fixed defect',3:'reversable defect'})
x = df.drop('target',axis=1)

y= df['target']
df.isna().sum()
df.head()
x = pd.get_dummies(x,drop_first=True)
x.head()
x.shape
# splitting the sets into training and test sets



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
#Modelling

#Randomforest Classifier

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators = 50, max_depth = 5)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

# confusion matrix

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot = True,  cmap = 'Blues',cbar=None)

plt.title('Confusion Matrix')

plt.ylabel('True Class')

plt.xlabel('Predicted Class')



# classification report

cr = classification_report(y_test, y_pred)

print(cr)
#Logistic regression

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(x_train,y_train)

pred_lr= lr.predict(x_test)

cm_lr = confusion_matrix(y_test, pred_lr)

sns.heatmap(cm_lr, annot = True,  cmap = 'Blues',cbar=None)

plt.title('Confusion Matrix')

plt.ylabel('True Class')

plt.xlabel('Predicted Class')



# classification report

cr_lr = classification_report(y_test, pred_lr)

print(cr_lr)
#Gaussian naive bayes

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(x_train,y_train)

pred_nb = nb.predict(x_test)

cm_nb = confusion_matrix(y_test, pred_nb)

sns.heatmap(cm_nb, annot = True,  cmap = 'Blues',cbar=None)

plt.title('Confusion Matrix')

plt.ylabel('True Class')

plt.xlabel('Predicted Class')



# classification report

cr_nb = classification_report(y_test, pred_nb)

print(cr_nb)



from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

knn_score = []

for i in range(1,21):

    dt= KNeighborsClassifier(n_neighbors=i)

    dt.fit(x_train,y_train)

    pred = dt.predict(x_test)

    knn_score.append(accuracy_score(y_test,pred))

plt.plot(range(1,21),knn_score)

plt.xticks(np.arange(1,20,1))    
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=13)

knn.fit(x_train,y_train)

pred_knn = knn.predict(x_test)

cm_knn = confusion_matrix(y_test, pred_knn)

sns.heatmap(cm_knn, annot = True,  cmap = 'Blues',cbar=None)

plt.title('Confusion Matrix')

plt.ylabel('True Class')

plt.xlabel('Predicted Class')



# classification report

cr_knn = classification_report(y_test, pred_knn)

print(cr_knn)