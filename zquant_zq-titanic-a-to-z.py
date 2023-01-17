# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings("ignore")
#Preprocessing
from sklearn.preprocessing import Imputer, StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import scikitplot as skplt
#Models & Metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold

import keras
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

from sklearn.model_selection import GridSearchCV
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
dataset = pd.read_csv('../input/train.csv').set_index('PassengerId', drop=True)
testset = pd.read_csv('../input/test.csv').set_index('PassengerId', drop=True)
dataset.head()
testset.head()
dataset.info()
dataset.Sex = dataset.Sex.astype('category')
testset.Sex = testset.Sex.astype('category')

dataset.Embarked = dataset.Embarked.astype('category')
testset.Embarked = testset.Embarked.astype('category')

dataset.info()
f, ax = plt.subplots(1, 2, figsize=(14, 8))
sns.heatmap(dataset.isnull(), cbar=False, cmap='viridis', ax=ax[0])
sns.heatmap(testset.isnull(), cbar=False, cmap='viridis', ax=ax[1])
print('NaN Value Counts\n\nTraining Set:\n%s\n\nTesting Set:\n%s' % \
      (dataset.isnull().sum(), testset.isnull().sum()))
train_deceased = dataset[dataset.Survived==0]
train_survived = dataset[dataset.Survived==1]
labels = 'Deceased', 'Survived'
sizes = [train_deceased.shape[0], train_survived.shape[0]]
colors = ['lightcoral', 'lightskyblue']
explode = (0.05, 0)  # explode 1st slice
 
fig = plt.figure(figsize=(14, 8))
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
plt.title('Deceased vs Survived Class Imbalance')
 
plt.axis('equal')
plt.show()
fig = plt.figure(figsize=(24, 15))

plt.subplot(331)
sns.distplot(train_deceased.Age.dropna(), kde=False, color='Red')
sns.distplot(train_survived.Age.dropna(), kde=False, color='Blue')
plt.title('Survival Age Distribution')
plt.ylabel('Frequency')

plt.subplot(332)
sns.distplot(train_survived.Fare, kde=False, color='Blue')
sns.distplot(train_deceased.Fare, kde=False, color='Red')
plt.title('Survival Fare Distribution')
plt.ylabel('Frequency')
plt.xlim(0, 200)

fig, ax = plt.subplots(1, 3, figsize=(14, 8))

g = sns.factorplot(x='Embarked', kind='count', data=dataset, hue='Survived', ax=ax[0], legend=False)
g.ax.set_axis_off()
ax[0].set_title('Embarkment Location Survival')

g = sns.factorplot(x='Pclass', kind='count', data=dataset, hue='Survived', ax=ax[1], legend=False)
g.ax.set_axis_off()
ax[1].set_title('Ticket Class Survival')

g = sns.factorplot(x='Sex', kind='count', data=dataset, hue='Survived', ax=ax[2], legend=False)
g.ax.set_axis_off()
ax[2].set_title('Sex Survival')
sns.factorplot(x='Parch', kind='count', hue='Survived', data=dataset, orient='v')
plt.title('Parent/Children Survival')

sns.factorplot(x='SibSp', kind='count', hue='Survived', data=dataset, orient='v')
plt.title('Sibling/Spouse Survival')
#Look at the Fare feature's distribution.
f, ax = plt.subplots(1, 2, figsize=(14, 8))

sns.kdeplot(dataset.Fare[dataset.Pclass==1], ax=ax[0], legend=False)
sns.kdeplot(dataset.Fare[dataset.Pclass==2], ax=ax[0], legend=False)
sns.kdeplot(dataset.Fare[dataset.Pclass==3], ax=ax[0], legend=False)
ax[0].set_title('Fare per Ticket Class')
ax[0].legend(['1', '2', '3'])
ax[0].set_xlim(-50, 300)

sns.kdeplot(dataset.Fare[dataset.Embarked=='C'], ax=ax[1], legend=False)
sns.kdeplot(dataset.Fare[dataset.Embarked=='S'], ax=ax[1], legend=False)
sns.kdeplot(dataset.Fare[dataset.Embarked=='Q'], ax=ax[1], legend=False)
ax[1].set_title('Fare per Embarkment Location')
ax[0].legend(['C', 'S', 'Q'])
ax[1].set_xlim(-50, 300)
print('Cabins')

print('We only know %.2f%% of the cabin numbers in the training set.' % ((len(dataset[dataset.Cabin.notnull()])/\
                                                                        len(dataset))*100))
print('We only know %.2f%% of the cabin numbers in the testing set.' % ((len(testset[testset.Cabin.notnull()])/\
                                                                        len(testset))*100))

print('\nTickets')
print('There are %s unique versus %s duplicate tickets.' % (dataset.Ticket.nunique(), len(dataset)-dataset.Ticket.nunique()))
fig = plt.figure(figsize=(10, 10))
font = {'size' : 22}
plt.rc('font', **font)
sns.heatmap(data=dataset.loc[:, dataset.columns != 'Survived'].corr(), \
cmap='plasma',annot=True, fmt='.2f', linewidth=0.5)

fig, ax = plt.subplots(1, 2, figsize=(14, 8))

sns.kdeplot(dataset.Age[dataset.Pclass==1], ax=ax[0], legend=False)
sns.kdeplot(dataset.Age[dataset.Pclass==2], ax=ax[0], legend=False)
sns.kdeplot(dataset.Age[dataset.Pclass==3], ax=ax[0], legend=False)

sns.kdeplot(testset.Age[testset.Pclass==1], ax=ax[1], legend=False)
sns.kdeplot(testset.Age[testset.Pclass==2], ax=ax[1], legend=False)
sns.kdeplot(testset.Age[testset.Pclass==3], ax=ax[1], legend=False)

#fig.legend(['1', '2', '3'], loc=5)
fig.suptitle('Training vs Testing Age per Class Distribution')

fig, ax = plt.subplots(1, 2, figsize=(14, 8))

sns.kdeplot(dataset.Age[dataset.Sex=='male'], ax=ax[0], legend=False)
sns.kdeplot(dataset.Age[dataset.Sex=='female'], ax=ax[0], legend=False)

sns.kdeplot(testset.Age[testset.Sex=='male'], ax=ax[1], legend=False)
sns.kdeplot(testset.Age[testset.Sex=='female'], ax=ax[1], legend=False)

#fig.legend(['male', 'female'], loc=5)
fig.suptitle('Training vs Testing Age per Sex Distribution')

fig, ax = plt.subplots(1, 2, figsize=(14, 8), sharex=True)
ax[0].set_xlim(-50, 300)

sns.kdeplot(dataset.Fare[dataset.Embarked=='C'], ax=ax[0])
sns.kdeplot(dataset.Fare[dataset.Embarked=='S'], ax=ax[0])
sns.kdeplot(dataset.Fare[dataset.Embarked=='Q'], ax=ax[0])

sns.kdeplot(testset.Fare[testset.Embarked=='C'], ax=ax[1])
sns.kdeplot(testset.Fare[testset.Embarked=='S'], ax=ax[1])
sns.kdeplot(testset.Fare[testset.Embarked=='Q'], ax=ax[1])


#fig.legend(['C', 'S', 'Q'], loc=5)
fig.suptitle('Training vs Testing Fare per Embarkment Location')
fig, ax = plt.subplots(1, 2, figsize=(14, 8), sharex=True, sharey=True)
sns.pointplot(x='Parch', y='SibSp', data=dataset, ax=ax[0])
sns.pointplot(x='Parch', y='SibSp', data=testset, ax=ax[1])

fig.suptitle('Training v Testing Parch to SibSp Samples')
#Convert Sex column to encoded labels
le = LabelEncoder()
dataset.Sex = le.fit_transform(dataset.Sex)
testset.Sex = le.fit_transform(testset.Sex)

#Since there are only 2 missing values in the embarked row, we can just drop these for now
dataset = dataset[dataset.Embarked.notnull()]

#Get dummy variables for Embarked column & drop one of the dummy variables
dataset = pd.concat([dataset.drop('Embarked', axis=1), pd.get_dummies(dataset.Embarked, drop_first=True)],\
         axis=1)
testset = pd.concat([testset.drop('Embarked', axis=1), pd.get_dummies(testset.Embarked, drop_first=True)],\
         axis=1)

#Due to the overwhelmindly large amount of missing cabin values, we will drop the row all together
dataset.drop('Cabin', inplace=True, axis=1)
testset.drop('Cabin', inplace=True, axis=1)

dataset.head()
avg_1 = dataset.Age[dataset.Pclass==1].dropna().mean()
avg_2 = dataset.Age[dataset.Pclass==2].dropna().mean()
avg_3 = dataset.Age[dataset.Pclass==1].dropna().mean()

def fill_na(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if(np.isnan(Age)):
        if(Pclass==1):
            return avg_1
        elif(Pclass==2):
            return avg_2
        elif(Pclass==3):
            return avg_3
    else:
        return Age
        
dataset['Age'] = dataset[['Age', 'Pclass']].apply(fill_na, axis=1)
testset['Age'] = testset[['Age', 'Pclass']].apply(fill_na, axis=1)
imp = Imputer(missing_values ='NaN', \
             strategy='mean', \
             axis=0)
#print(dataset.columns)
#print(testset.columns)

imp.fit(dataset.iloc[:, 8:9])
testset.Fare = imp.transform(testset.iloc[:, 7:8])
#Scale the features
ss = StandardScaler()

i = np.argwhere('Age'==dataset.columns)[0][0]

dataset.Age = ss.fit_transform(dataset.iloc[:, i:i+1])
testset.Age = ss.transform(testset.iloc[:, i-1:i])
f, ax = plt.subplots(1, 2, figsize=(14, 8))
sns.heatmap(dataset.isnull(), cbar=False, cmap='viridis', ax=ax[0])
sns.heatmap(testset.isnull(), cbar=False, cmap='viridis', ax=ax[1])
def str_freq(cols):
    
    ticket = cols
    return sum(bytearray(ticket, 'utf8'))


dataset.Ticket = dataset.Ticket.apply(str_freq)
testset.Ticket = testset.Ticket.apply(str_freq)
'''
dataset['Family'] = dataset.SibSp + dataset.Parch
testset['Family'] = testset.SibSp + testset.Parch

dataset.drop(['SibSp', 'Parch', 'Name', 'Ticket'], inplace=True, axis=1)
testset.drop(['SibSp', 'Parch', 'Name', 'Ticket'], inplace=True, axis=1)
'''
dataset.drop(['Name', 'Ticket'], inplace=True, axis=1)
testset.drop(['Name', 'Ticket'], inplace=True, axis=1)
def is_neginf(cols):
    Fare = cols[0]
    
    if(np.isneginf(Fare)):
        return 0
    else:
        return Fare

dataset.Fare = np.log(dataset.Fare)
testset.Fare = np.log(testset.Fare)

dataset.Fare = dataset[['Fare', 'Age']].apply(is_neginf, axis=1)
testset.Fare = testset[['Fare', 'Age']].apply(is_neginf, axis=1)

i = np.argwhere('Age'==dataset.columns)[0][0]

dataset.Fare = ss.fit_transform(dataset.iloc[:, i:i+1])
testset.Fare = ss.transform(testset.iloc[:, i-1:i])

fig, ax = plt.subplots(1, 2, figsize=(14, 8), sharex=True)

sns.kdeplot(dataset.Fare, ax=ax[0])
sns.kdeplot(testset.Fare, ax=ax[1])

fig.suptitle('Training vs Testing Log Fare Distribution')
#Make sure all features are in the same order
dataset.info()
print('\n\n')
testset.info()
y = dataset.Survived
X = dataset.loc[:, dataset.columns != 'Survived']

'''
clf = RandomForestClassifier(n_estimators=25, max_depth=4, \
                             min_samples_split = 25, \
                             min_samples_leaf = 7, \
                             random_state=48, 
                            class_weight={0:0.5, 1:0.5})
'''
clf = RandomForestClassifier(n_estimators=25, max_depth=7, \
                             min_samples_split = 30, \
                             min_samples_leaf = 14, \
                             random_state=6, 
                             class_weight={0:0.5, 1:0.5})
clf.fit(X, y)

scoring = {'acc': 'accuracy',
                   'prec_macro': 'precision_macro',
                   'rec_micro': 'recall_macro'}
        
score = cross_validate(clf, X, y,
                       scoring=scoring, 
                       cv=KFold(n_splits=3, \
                               shuffle=True, random_state=0))
        
print('Accuracy: %.2f STD: %.3f' % (score['test_acc'].mean(), score['test_acc'].std()))
print('Precision: %.2f STD: %.3f' % (score['test_prec_macro'].mean(), score['test_prec_macro'].std()))
print('Recall: %.2f STD: %.3f' % (score['test_rec_micro'].mean(), score['test_rec_micro'].std()))
def build_classifier():
    clf = keras.Sequential()
    
    clf.add(Dense(output_dim = 8, input_dim = 8 , init = 'uniform', activation = 'relu'))
    
    clf.add(Dense(output_dim = 8, init = 'uniform', activation = 'tanh'))
    
    clf.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
    
    clf.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])
    
    return clf

clf = KerasClassifier(build_fn = build_classifier, batch_size=10, epochs = 100)
scoring = {'acc': 'accuracy',
                   'prec_macro': 'precision_macro',
                   'rec_micro': 'recall_macro'}

score = cross_validate(clf, X, y, scoring=scoring, \
                      cv=KFold(n_splits=3, \
                               shuffle=True, random_state=0))
print('Accuracy: %.2f STD: %.3f' % (score['test_acc'].mean(), score['test_acc'].std()))
print('Precision: %.2f STD: %.3f' % (score['test_prec_macro'].mean(), score['test_prec_macro'].std()))
print('Recall: %.2f STD: %.3f' % (score['test_rec_micro'].mean(), score['test_rec_micro'].std()))
knn = KNeighborsClassifier()

params = {'n_neighbors' : [2**i for i in range(6)], \
         'weights' : ['uniform', 'distance']}
clf = GridSearchCV(knn, param_grid=params, scoring='accuracy', cv = KFold(n_splits=3, \
                                                                      shuffle=True, \
                                                                      random_state=0))
clf.fit(X, y)
clf.best_params_
clf = KNeighborsClassifier(n_neighbors=8, weights='uniform')

scoring = {'acc': 'accuracy',
                   'prec_macro': 'precision_macro',
                   'rec_micro': 'recall_macro'}
        
score = cross_validate(clf, X, y,
                       scoring=scoring, 
                       cv=KFold(n_splits=3, \
                               shuffle=True, random_state=0))
        
print('Accuracy: %.2f STD: %.3f' % (score['test_acc'].mean(), score['test_acc'].std()))
print('Precision: %.2f STD: %.3f' % (score['test_prec_macro'].mean(), score['test_prec_macro'].std()))
print('Recall: %.2f STD: %.3f' % (score['test_rec_micro'].mean(), score['test_rec_micro'].std()))
'''my_submission = pd.DataFrame({'PassengerId': testset.index.values, 'Survived': \
                              clf.predict_classes(testset).reshape(1, -1)[0]})
my_submission.to_csv('submission.csv', index=False)'''