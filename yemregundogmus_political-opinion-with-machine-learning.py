import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
data = pd.read_csv('../input/turkey-political-opinions/data.csv')
data = data.drop(['Timestamp'], axis = 1)
names = ["Sex","Age","Region","Education","Economy_Good","Education_Reform",
         "Against_Privatization","support_death_penalty","obejctive_journalists","Alcohol_prohibition",
         "More_Secular","abortion_ban","Restricted_Freedoms","support_new_party"]
data = data.rename(columns=dict(zip(data.columns, names)))
data.head()
dataCHP = data[(data.parti == 'CHP')]
dataAKP = data[(data.parti == 'AKP')]
dataIYI = data[(data.parti == 'IYI PARTI')]
len(dataCHP), len(dataAKP), len(dataIYI)
data = pd.concat([dataCHP, dataAKP, dataIYI])
data.head()
data.info()
data.describe
fig = plt.figure(figsize=(5,5))
data['Sex'].value_counts().plot(kind = 'pie', autopct='%.1f%%')
plt.ylabel(" ", fontsize = 15)
plt.title("Gender Distribution for the Sample")
print("")
fig = plt.figure(figsize=(5,5))
data['Age'].value_counts().plot(kind = 'pie', autopct='%.1f%%')
plt.ylabel(" ", fontsize = 15)
plt.title("Age Distribution for the Sample")
print("")
fig = plt.figure(figsize=(5,5))
data['Region'].value_counts().plot(kind = 'pie', autopct='%.1f%%')
plt.ylabel(" ", fontsize = 15)
plt.title("City Distribution for the Sample")
print("")
fig = plt.figure(figsize=(5,5))
data['Education'].value_counts().plot(kind = 'pie', autopct='%.1f%%')
plt.ylabel(" ", fontsize = 15)
plt.title("Distribution of Educational Levels for the Sample")
print("")
def plotData(data):
    i = 4
    fig = plt.figure(figsize=(25,25))
    p1 = fig.add_subplot(4,4,1)
    data[names[i]].value_counts().plot(kind = 'pie', autopct='%.1f%%'); i = i + 1
    plt.ylabel(" ", fontsize = 15)
    plt.title("Economic Status is good?")
    plt.grid()
    p2 = fig.add_subplot(4,4,2)
    data[names[i]].value_counts().plot(kind = 'pie', autopct='%.1f%%'); i = i + 1
    plt.title("Need Reform in Education?")
    plt.ylabel(" ", fontsize = 15)
    plt.grid()
    p3 = fig.add_subplot(4,4,3)
    data[names[i]].value_counts().plot(kind = 'pie', autopct='%.1f%%'); i = i + 1
    plt.title("Resolve Privatization Are You?")
    plt.ylabel(" ", fontsize = 15)
    plt.grid()
    p4 = fig.add_subplot(4,4,4)
    data[names[i]].value_counts().plot(kind = 'pie', autopct='%.1f%%'); i = i + 1
    plt.ylabel(" ", fontsize = 15)
    plt.title("penalty like death penalty?")
    p5 = fig.add_subplot(4,4,5)
    data[names[i]].value_counts().plot(kind = 'pie', autopct='%.1f%%'); i = i + 1
    plt.ylabel(" ", fontsize = 15)
    plt.title("journalists objective enough?")
    plt.grid()
    p6 = fig.add_subplot(4,4,6)
    data[names[i]].value_counts().plot(kind = 'pie', autopct='%.1f%%'); i = i + 1
    plt.title("Support prohibition of alcohol after 22.00?")
    plt.ylabel(" ", fontsize = 15)
    plt.grid()
    p7 = fig.add_subplot(4,4,7)
    data[names[i]].value_counts().plot(kind = 'pie', autopct='%.1f%%'); i = i + 1
    plt.title("Live in a Secular State?")
    plt.ylabel(" ", fontsize = 15)
    plt.grid()
    p8 = fig.add_subplot(4,4,8)
    data[names[i]].value_counts().plot(kind = 'pie', autopct='%.1f%%'); i = i + 1
    plt.ylabel(" ", fontsize = 15)
    plt.title("Are you supporting the abortion ban?")
    p9 = fig.add_subplot(4,4,9)
    data[names[i]].value_counts().plot(kind = 'pie', autopct='%.1f%%'); i = i + 1
    plt.title("extraordinary state (Ohal) restricts Freedoms?")
    plt.ylabel(" ", fontsize = 15)
    plt.grid()
    p10 = fig.add_subplot(4,4,10)
    data[names[i]].value_counts().plot(kind = 'pie', autopct='%.1f%%'); i = i + 1
    plt.ylabel(" ", fontsize = 15)
    plt.title("New party should enter to the parliament?")
    print(" ")
print("Opinion of AKP")
plotData(dataAKP)
print("Opinion of CHP")
plotData(dataCHP)
print("Opinion of IYI PARTI")
plotData(dataIYI)
fig = plt.figure(figsize=(5,5))
data['parti'].value_counts().plot(kind = 'pie', autopct='%.1f%%')
plt.ylabel(" ", fontsize = 15)
plt.title("Political opinion distribution for the sample")
print("")
combine = [data]
age_mapping = {"0-18": 1, "18-30": 2, "30-50": 3, "50-60": 4, "60+": 5}
for dataset in combine:
    dataset['Age'] = dataset['Age'].map(age_mapping)
    dataset['Age'] = dataset['Age'].fillna(0)

data.head()
sex_mapping = {"Erkek": 0, "Kadın": 1}
data['Sex'] = data['Sex'].map(sex_mapping)

data.head()
combine = [data]
city_mapping = {"Marmara": 1, "Ege": 2, "Karadeniz": 3, "Akdeniz": 4, "İç Anadolu": 5, "Doğu Anadolu": 6, "Güneydoğu": 7}
for dataset in combine:
    dataset['Region'] = dataset['Region'].map(city_mapping)
    dataset['Region'] = dataset['Region'].fillna(0)

data.head()
combine = [data]
education_mapping = {"İlkokul": 1, "Ortaokul": 2, "Lise": 3, "Ön Lisans": 4, "Lisans": 5, "Lisans Üstü": 6}
for dataset in combine:
    dataset['Education'] = dataset['Education'].map(education_mapping)
    dataset['Education'] = dataset['Education'].fillna(0)

data.head()
question_mapping = {"Evet": 1, "Hayır": 0}
for i in range(4,14):
    data[names[i]] = data[names[i]].map(question_mapping)

data.head()
combine = [data]
opinion_mapping = {"AKP": 0, "CHP": 1, "IYI PARTI": 3}
for dataset in combine:
    dataset['parti'] = dataset['parti'].map(opinion_mapping)
    dataset['parti'] = dataset['parti'].fillna(0)

data.head()
k = 14 
corrmat = data.corr()
cols = corrmat.nlargest(k, 'parti')['parti'].index
f, ax = plt.subplots(figsize=(15, 10))
cm = np.corrcoef(data[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', 
                 annot_kws={'size': 10}, yticklabels=names, xticklabels=names)
plt.show()
from sklearn.model_selection import train_test_split

predictors = data.drop(['parti'], axis=1)
target = data["parti"]
x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.03, random_state= 0)
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz

decisiontree = DecisionTreeClassifier(max_depth = 6)
decisiontree.fit(x_train, y_train)
y_pred = decisiontree.predict(x_val)
acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_decisiontree)
export_graphviz(decisiontree,
                                feature_names=names,
                                out_file='AKPvsCHPvsIYI.dot',
                                filled=True,
                                rounded=True)
from sklearn.ensemble import RandomForestClassifier

randomforest = RandomForestClassifier()
randomforest.fit(x_train, y_train)
y_pred = randomforest.predict(x_val)
acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_randomforest)
from sklearn.svm import SVC

svm = SVC()
svm.fit(x_train, y_train)
y_pred = svm.predict(x_val)
acc_svm = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_svm)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
y_pred = knn.predict(x_val)
acc_knn = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_knn)
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_val)
acc_logreg = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_logreg)
models = pd.DataFrame({
    'Model': ['SVC', 
              'Random Forest', 'KNeighbors', 
              'Decision Tree', 'Logistic Regression'],
    'Score': [ acc_svm, 
              acc_randomforest, acc_knn, acc_decisiontree, acc_logreg
             ]})
models.sort_values(by='Score', ascending=False)
data = pd.concat([dataCHP, dataAKP])
len(dataCHP), len(dataAKP)
data.head()
data.info()
data.describe
fig = plt.figure(figsize=(5,5))
data['Sex'].value_counts().plot(kind = 'pie', autopct='%.1f%%')
plt.ylabel(" ", fontsize = 15)
plt.title("Gender Distribution for the Sample2")
print("")
fig = plt.figure(figsize=(5,5))
data['Age'].value_counts().plot(kind = 'pie', autopct='%.1f%%')
plt.ylabel(" ", fontsize = 15)
plt.title("Age Distribution for the Sample2")
print("")
fig = plt.figure(figsize=(5,5))
data['Region'].value_counts().plot(kind = 'pie', autopct='%.1f%%')
plt.ylabel(" ", fontsize = 15)
plt.title("Region Distribution for the Sample2")
print("")
fig = plt.figure(figsize=(5,5))
data['Education'].value_counts().plot(kind = 'pie', autopct='%.1f%%')
plt.ylabel(" ", fontsize = 15)
plt.title("Distribution of Educational Levels for the Sample2")
print("")
fig = plt.figure(figsize=(5,5))
data['parti'].value_counts().plot(kind = 'pie', autopct='%.1f%%')
plt.ylabel(" ", fontsize = 15)
plt.title("Political opinion distribution for the Sample2")
print("")
data.head()
combine = [data]
age_mapping = {"0-18": 1, "18-30": 2, "30-50": 3, "50-60": 4, "60+": 5}
for dataset in combine:
    dataset['Age'] = dataset['Age'].map(age_mapping)
    dataset['Age'] = dataset['Age'].fillna(0)

sex_mapping = {"Erkek": 0, "Kadın": 1}
data['Sex'] = data['Sex'].map(sex_mapping)

combine = [data]
city_mapping = {"Marmara": 1, "Ege": 2, "Karadeniz": 3, "Akdeniz": 4, "İç Anadolu": 5, "Doğu Anadolu": 6, "Güneydoğu": 7}
for dataset in combine:
    dataset['Region'] = dataset['Region'].map(city_mapping)
    dataset['Region'] = dataset['Region'].fillna(0)
    
combine = [data]
education_mapping = {"İlkokul": 1, "Ortaokul": 2, "Lise": 3, "Ön Lisans": 4, "Lisans": 5, "Lisans Üstü": 6}
for dataset in combine:
    dataset['Education'] = dataset['Education'].map(education_mapping)
    dataset['Education'] = dataset['Education'].fillna(0)

question_mapping = {"Evet": 1, "Hayır": 0}
for i in range(4,14):
    data[names[i]] = data[names[i]].map(question_mapping)
    
combine = [data]
opinion_mapping = {"AKP": 0, "CHP": 1}
for dataset in combine:
    dataset['parti'] = dataset['parti'].map(opinion_mapping)
    dataset['parti'] = dataset['parti'].fillna(0)
    
data.head()
k = 14 #number of variables for heatmap
corrmat = data.corr()
cols = corrmat.nlargest(k, 'parti')['parti'].index
f, ax = plt.subplots(figsize=(15, 10))
cm = np.corrcoef(data[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', 
                 annot_kws={'size': 10}, yticklabels=names, xticklabels=names)
plt.show()
from sklearn.model_selection import train_test_split

predictors = data.drop(['parti'], axis=1)
target = data["parti"]
x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.25, random_state = 0)
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz

decisiontree = DecisionTreeClassifier(max_depth = 6)
decisiontree.fit(x_train, y_train)
y_pred = decisiontree.predict(x_val)
acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_decisiontree)
export_graphviz(decisiontree,
                                feature_names=names,
                                out_file='AKPvsCHP.dot',
                                filled=True,
                                rounded=True)
 
from sklearn.ensemble import RandomForestClassifier

randomforest = RandomForestClassifier()
randomforest.fit(x_train, y_train)
y_pred = randomforest.predict(x_val)
acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_randomforest)
from sklearn.svm import SVC

svm = SVC()
svm.fit(x_train, y_train)
y_pred = svm.predict(x_val)
acc_svm = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_svm)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
y_pred = knn.predict(x_val)
acc_knn = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_knn)
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_val)
acc_logreg = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_logreg)
models = pd.DataFrame({
    'Model': ['SVC', 
              'Random Forest', 'KNeighbors', 
              'Decision Tree', 'Logistic Regression'],
    'Score': [ acc_svm, 
              acc_randomforest, acc_knn, acc_decisiontree, acc_logreg
             ]})
models.sort_values(by='Score', ascending=False)
data = pd.concat([dataIYI, dataAKP])
len(dataIYI), len(dataAKP)
data.head()
data.info()
data.describe
fig = plt.figure(figsize=(5,5))
data['Sex'].value_counts().plot(kind = 'pie', autopct='%.1f%%')
plt.ylabel(" ", fontsize = 15)
plt.title("Gender Distribution for the Sample3")
print("")
fig = plt.figure(figsize=(5,5))
data['Age'].value_counts().plot(kind = 'pie', autopct='%.1f%%')
plt.ylabel(" ", fontsize = 15)
plt.title("Age Distribution for the Sample3")
print("")
fig = plt.figure(figsize=(5,5))
data['Region'].value_counts().plot(kind = 'pie', autopct='%.1f%%')
plt.ylabel(" ", fontsize = 15)
plt.title("Region Distribution for the Sample3")
print("")
fig = plt.figure(figsize=(5,5))
data['parti'].value_counts().plot(kind = 'pie', autopct='%.1f%%')
plt.ylabel(" ", fontsize = 15)
plt.title("Political opinion distribution for the Sample3")
print("")
data.head()
combine = [data]
age_mapping = {"0-18": 1, "18-30": 2, "30-50": 3, "50-60": 4, "60+": 5}
for dataset in combine:
    dataset['Age'] = dataset['Age'].map(age_mapping)
    dataset['Age'] = dataset['Age'].fillna(0)

sex_mapping = {"Erkek": 0, "Kadın": 1}
data['Sex'] = data['Sex'].map(sex_mapping)

combine = [data]
city_mapping = {"Marmara": 1, "Ege": 2, "Karadeniz": 3, "Akdeniz": 4, "İç Anadolu": 5, "Doğu Anadolu": 6, "Güneydoğu": 7}
for dataset in combine:
    dataset['Region'] = dataset['Region'].map(city_mapping)
    dataset['Region'] = dataset['Region'].fillna(0)
    
combine = [data]
education_mapping = {"İlkokul": 1, "Ortaokul": 2, "Lise": 3, "Ön Lisans": 4, "Lisans": 5, "Lisans Üstü": 6}
for dataset in combine:
    dataset['Education'] = dataset['Education'].map(education_mapping)
    dataset['Education'] = dataset['Education'].fillna(0)

question_mapping = {"Evet": 1, "Hayır": 0}
for i in range(4,14):
    data[names[i]] = data[names[i]].map(question_mapping)
    
combine = [data]
opinion_mapping = {"AKP": 0, "IYI PARTI": 1}
for dataset in combine:
    dataset['parti'] = dataset['parti'].map(opinion_mapping)
    dataset['parti'] = dataset['parti'].fillna(0)
    
data.head()
k = 14 #number of variables for heatmap
corrmat = data.corr()
cols = corrmat.nlargest(k, 'parti')['parti'].index
f, ax = plt.subplots(figsize=(15, 10))
cm = np.corrcoef(data[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', 
                 annot_kws={'size': 10}, yticklabels=names, xticklabels=names)
plt.show()
from sklearn.model_selection import train_test_split

predictors = data.drop(['parti'], axis=1)
target = data["parti"]
x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.053, random_state = 0)
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz

decisiontree = DecisionTreeClassifier(max_depth = 6)
decisiontree.fit(x_train, y_train)
y_pred = decisiontree.predict(x_val)
acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_decisiontree)
export_graphviz(decisiontree,
                                feature_names=names,
                                out_file='AKPvsIYI.dot',
                                filled=True,
                                rounded=True)
 
from sklearn.ensemble import RandomForestClassifier

randomforest = RandomForestClassifier()
randomforest.fit(x_train, y_train)
y_pred = randomforest.predict(x_val)
acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_randomforest)
from sklearn.svm import SVC

svm = SVC()
svm.fit(x_train, y_train)
y_pred = svm.predict(x_val)
acc_svm = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_svm)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
y_pred = knn.predict(x_val)
acc_knn = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_knn)
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_val)
acc_logreg = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_logreg)
models = pd.DataFrame({
    'Model': ['SVC', 
              'Random Forest', 'KNeighbors', 
              'Decision Tree', 'Logistic Regression'],
    'Score': [ acc_svm, 
              acc_randomforest, acc_knn, acc_decisiontree, acc_logreg
             ]})
models.sort_values(by='Score', ascending=False)
data = pd.concat([dataCHP, dataIYI])
data.head()
data.info()
data.describe
fig = plt.figure(figsize=(5,5))
data['Sex'].value_counts().plot(kind = 'pie', autopct='%.1f%%')
plt.ylabel(" ", fontsize = 15)
plt.title("Gender Distribution for the Sample4")
print("")
fig = plt.figure(figsize=(5,5))
data['Age'].value_counts().plot(kind = 'pie', autopct='%.1f%%')
plt.ylabel(" ", fontsize = 15)
plt.title("Age Distribution for the Sample4")
print("")
fig = plt.figure(figsize=(5,5))
data['Region'].value_counts().plot(kind = 'pie', autopct='%.1f%%')
plt.ylabel(" ", fontsize = 15)
plt.title("Region Distribution for the Sample4")
print("")
fig = plt.figure(figsize=(5,5))
data['Education'].value_counts().plot(kind = 'pie', autopct='%.1f%%')
plt.ylabel(" ", fontsize = 15)
plt.title("Distribution of Educational Levels for the Sample4")
print("")
fig = plt.figure(figsize=(5,5))
data['parti'].value_counts().plot(kind = 'pie', autopct='%.1f%%')
plt.ylabel(" ", fontsize = 15)
plt.title("Political opinion distribution for the Sample4")
print("")
data.head()
combine = [data]
age_mapping = {"0-18": 1, "18-30": 2, "30-50": 3, "50-60": 4, "60+": 5}
for dataset in combine:
    dataset['Age'] = dataset['Age'].map(age_mapping)
    dataset['Age'] = dataset['Age'].fillna(0)

sex_mapping = {"Erkek": 0, "Kadın": 1}
data['Sex'] = data['Sex'].map(sex_mapping)

combine = [data]
city_mapping = {"Marmara": 1, "Ege": 2, "Karadeniz": 3, "Akdeniz": 4, "İç Anadolu": 5, "Doğu Anadolu": 6, "Güneydoğu": 7}
for dataset in combine:
    dataset['Region'] = dataset['Region'].map(city_mapping)
    dataset['Region'] = dataset['Region'].fillna(0)
    
combine = [data]
education_mapping = {"İlkokul": 1, "Ortaokul": 2, "Lise": 3, "Ön Lisans": 4, "Lisans": 5, "Lisans Üstü": 6}
for dataset in combine:
    dataset['Education'] = dataset['Education'].map(education_mapping)
    dataset['Education'] = dataset['Education'].fillna(0)

question_mapping = {"Evet": 1, "Hayır": 0}
for i in range(4,14):
    data[names[i]] = data[names[i]].map(question_mapping)
    
combine = [data]
opinion_mapping = {"IYI PARTI": 0, "CHP": 1}
for dataset in combine:
    dataset['parti'] = dataset['parti'].map(opinion_mapping)
    dataset['parti'] = dataset['parti'].fillna(0)
    
data.head()
k = 14 #number of variables for heatmap
corrmat = data.corr()
cols = corrmat.nlargest(k, 'parti')['parti'].index
f, ax = plt.subplots(figsize=(15, 10))
cm = np.corrcoef(data[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', 
                 annot_kws={'size': 10}, yticklabels=names, xticklabels=names)
plt.show()
from sklearn.model_selection import train_test_split

predictors = data.drop(['parti'], axis=1)
target = data["parti"]
x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.29, random_state = 0)
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz

decisiontree = DecisionTreeClassifier(max_depth = 6)
decisiontree.fit(x_train, y_train)
y_pred = decisiontree.predict(x_val)
acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_decisiontree)
export_graphviz(decisiontree,
                                feature_names=names,
                                out_file='IYIvsCHP.dot',
                                filled=True,
                                rounded=True)
 
from sklearn.ensemble import RandomForestClassifier

randomforest = RandomForestClassifier()
randomforest.fit(x_train, y_train)
y_pred = randomforest.predict(x_val)
acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_randomforest)
from sklearn.svm import SVC

svm = SVC()
svm.fit(x_train, y_train)
y_pred = svm.predict(x_val)
acc_svm = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_svm)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
y_pred = knn.predict(x_val)
acc_knn = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_knn)
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_val)
acc_logreg = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_logreg)
models = pd.DataFrame({
    'Model': ['SVC', 
              'Random Forest', 'KNeighbors', 
              'Decision Tree', 'Logistic Regression'],
    'Score': [ acc_svm, 
              acc_randomforest, acc_knn, acc_decisiontree, acc_logreg
             ]})
models.sort_values(by='Score', ascending=False)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

Sample1 = mpimg.imread('../input/decision-tree-images/AKPvsCHPvsIYI.jpeg')
Sample2 = mpimg.imread('../input/decision-tree-images/AKPvsCHP.jpeg')
Sample3 = mpimg.imread('../input/decision-tree-images/AKPvsIYI.jpeg')
Sample4 = mpimg.imread('../input/decision-tree-images/IYIvsCHP.jpeg')
fig = plt.figure(figsize=(75,75))
p1 = fig.add_subplot(3,3,1)
plt.title("AKP-CHP-IYI TREE", fontsize = 72)
imgplot = plt.imshow(Sample1)
p2 = fig.add_subplot(3,3,2)
plt.title("AKP-CHP TREE", fontsize = 72)
imgplot = plt.imshow(Sample2)
p3 = fig.add_subplot(3,3,4)
plt.title("AKP-IYI TREE", fontsize = 72)
imgplot = plt.imshow(Sample3)
p4 = fig.add_subplot(3,3,5)
plt.title("CHP-IYI TREE", fontsize = 72)
imgplot = plt.imshow(Sample4)