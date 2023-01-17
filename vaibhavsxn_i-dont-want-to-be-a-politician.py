# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import matplotlib.pyplot as plt

import seaborn as sns

# Any results you write to the current directory are saved as output.
# data = pd.read_csv('/kaggle/input/congressional-voting-records/house-votes-84.names', error_bad_lines=False)

data = pd.read_csv('/kaggle/input/congressional-voting-records/house-votes-84.csv')
data.head()
data.shape
data.info()
data.dtypes
data['Target'] = np.where(data['Class Name'] == 'democrat', 1, 0)
data.Target.value_counts()
plt.figure(figsize = (50,20))

sns.set(style="darkgrid")

ax = sns.countplot(x='Class Name',data=data)

ax.set_xticklabels(ax.get_xticklabels(), fontsize = 30)

plt.tight_layout()
data[' handicapped-infants'].value_counts()
plt.figure(figsize = (20,10))

sns.set(style="darkgrid")

ax = sns.countplot(x=' handicapped-infants',data=data)

ax.set_xticklabels(ax.get_xticklabels(), fontsize = 30)

plt.tight_layout()
data.groupby(' handicapped-infants')['Target'].mean().sort_values(ascending = False)
data['handicapped_infants_n'] = np.where(data[' handicapped-infants'] == 'n', 1, 0)

data['handicapped_infants_y'] = np.where(data[' handicapped-infants'] == 'y', 1, 0)
data[' water-project-cost-sharing'].value_counts()
data.groupby(' water-project-cost-sharing')['Target'].mean().sort_values(ascending = False)
data['water-project-cost-sharing_n'] = np.where(data[' water-project-cost-sharing'] == 'n', 1, 0)

data['water-project-cost-sharing_y'] = np.where(data[' water-project-cost-sharing'] == 'y', 1, 0)
data[' adoption-of-the-budget-resolution'].value_counts()
data.groupby(' adoption-of-the-budget-resolution')['Target'].mean().sort_values(ascending = False)
data['adoption-of-the-budget-resolution_n'] = np.where(data[' adoption-of-the-budget-resolution'] == 'n', 1, 0)

data['adoption-of-the-budget-resolution_y'] = np.where(data[' adoption-of-the-budget-resolution'] == 'y', 1, 0)
data[' physician-fee-freeze'].value_counts()
data.groupby(' physician-fee-freeze')['Target'].mean().sort_values(ascending = False)
data['physician-fee-freeze_n'] = np.where(data[' physician-fee-freeze'] == 'n', 1, 0)

data['physician-fee-freeze_y'] = np.where(data[' physician-fee-freeze'] == 'y', 1, 0)
data.columns
cols_to_be_dropped = ['Class Name', ' handicapped-infants', ' water-project-cost-sharing',

       ' adoption-of-the-budget-resolution', ' physician-fee-freeze',

       ' el-salvador-aid', ' religious-groups-in-schools',

       ' anti-satellite-test-ban', ' aid-to-nicaraguan-contras', ' mx-missile',

       ' immigration', ' synfuels-corporation-cutback', ' education-spending',

       ' superfund-right-to-sue', ' crime', ' duty-free-exports',

       ' export-administration-act-south-africa']



#We will drop these cols at the end of our analysis
data[' el-salvador-aid'].value_counts()
data.groupby(' el-salvador-aid')['Target'].mean()
data['el-salvador-aid_n'] = np.where(data[' el-salvador-aid'] == 'n', 1, 0)

data['el-salvador-aid_y'] = np.where(data[' el-salvador-aid'] == 'y', 1, 0)
data[' religious-groups-in-schools'].value_counts()
data.groupby(' religious-groups-in-schools')['Target'].mean()
data['religious-groups-in-schools_n'] = np.where(data[' religious-groups-in-schools'] == 'n', 1, 0)

data['religious-groups-in-schools_y'] = np.where(data[' religious-groups-in-schools'] == 'y', 1, 0)
data[' anti-satellite-test-ban'].value_counts()
data.groupby(' anti-satellite-test-ban')['Target'].mean()
data['anti-satellite-test-ban_n'] = np.where(data[' anti-satellite-test-ban'] == 'n', 1, 0)

data['anti-satellite-test-ban_y'] = np.where(data[' anti-satellite-test-ban'] == 'y', 1, 0)
data[' aid-to-nicaraguan-contras'].value_counts()
data.groupby(' aid-to-nicaraguan-contras')['Target'].mean()
data['aid-to-nicaraguan-contras_n'] = np.where(data[' aid-to-nicaraguan-contras'] == 'n', 1, 0)

data['aid-to-nicaraguan-contras_y'] = np.where(data[' aid-to-nicaraguan-contras'] == 'y', 1, 0)
data[' mx-missile'].value_counts()
data.groupby(' mx-missile')['Target'].mean()
data['mx-missile_n'] = np.where(data[' mx-missile'] == 'n', 1, 0)

data['mx-missile_y'] = np.where(data[' mx-missile'] == 'y', 1, 0)
data[' immigration'].value_counts()
data.groupby(' immigration')['Target'].mean()
data['immigration_n'] = np.where(data[' immigration'] == 'n', 1, 0)

data['immigration_y'] = np.where(data[' immigration'] == 'y', 1, 0)
data[' synfuels-corporation-cutback'].value_counts()
data.groupby(' synfuels-corporation-cutback')['Target'].mean()
data['synfuels-corporation-cutback_n'] = np.where(data[' synfuels-corporation-cutback'] == 'n', 1, 0)

data['synfuels-corporation-cutback_y'] = np.where(data[' synfuels-corporation-cutback'] == 'y', 1, 0)
data[' education-spending'].value_counts()
data.groupby(' education-spending')['Target'].mean()
data['education-spending_n'] = np.where(data[' education-spending'] == 'n', 1, 0)

data['education-spending_y'] = np.where(data[' education-spending'] == 'y', 1, 0)
data[' superfund-right-to-sue'].value_counts()
data.groupby(' superfund-right-to-sue')['Target'].mean()
data['superfund-right-to-sue_n'] = np.where(data[' superfund-right-to-sue'] == 'n', 1, 0)

data['superfund-right-to-sue_y'] = np.where(data[' superfund-right-to-sue'] == 'y', 1, 0)
data[' crime'].value_counts()
data.groupby(' crime')['Target'].mean()
data['crime_n'] = np.where(data[' crime'] == 'n', 1, 0)

data['crime_y'] = np.where(data[' crime'] == 'y', 1, 0)
data[' duty-free-exports'].value_counts()
data.groupby(' duty-free-exports')['Target'].mean()
data['duty-free-exports_n'] = np.where(data[' duty-free-exports'] == 'n', 1, 0)

data['duty-free-exports_y'] = np.where(data[' duty-free-exports'] == 'y', 1, 0)
data[' export-administration-act-south-africa'].value_counts()
data.groupby(' export-administration-act-south-africa')['Target'].mean()
data['export-administration-act-south-africa_n'] = np.where(data[' export-administration-act-south-africa'] == 'n', 1, 0)

data['export-administration-act-south-africa_y'] = np.where(data[' export-administration-act-south-africa'] == 'y', 1, 0)
data = data.drop(cols_to_be_dropped, axis = 1)
data.head()
data.shape
data.columns
from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size = 0.35, random_state = 42)
train.shape, test.shape
X_train = train.drop('Target', axis = 1)

X_test = test.drop('Target', axis = 1)

y_train = train['Target']

y_test = test['Target']
X_train.shape, X_test.shape, y_train.shape, y_test.shape
from sklearn.ensemble import RandomForestClassifier
def gini(list_of_values):

    sorted_list = sorted(list_of_values)

    height, area = 0, 0

    for value in sorted_list:

        height += value

        area += height - value / 2.

    fair_area = height * len(list_of_values) / 2.

    return (fair_area - area) / fair_area
def plot_confusion_matrix(y_true, y_pred, title = 'Confusion matrix', cmap=plt.cm.Blues):

    from sklearn.metrics import classification_report

    from sklearn.metrics import confusion_matrix

    print ('Classification Report:\n')

    print (classification_report(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred)

    def plot_confusion_matrix_plot(cm, title = 'Confusion matrix', cmap=plt.cm.Blues):

        plt.imshow(cm, interpolation='nearest', cmap=cmap)

        plt.title(title)

        plt.colorbar()

        tick_marks = np.arange(len(y_test.unique()))

        plt.xticks(tick_marks, rotation=45)

        plt.yticks(tick_marks)

        plt.tight_layout()

        plt.ylabel('True label')

        plt.xlabel('Predicted label')

    print (cm)

    plot_confusion_matrix_plot(cm=cm)
rf = RandomForestClassifier(criterion = 'gini', 

                            max_depth = 8,

                            max_features = 'auto',

                            min_samples_leaf = 0.01, 

                            min_samples_split = 0.01,

                            min_weight_fraction_leaf = 0.0632, 

                            n_estimators = 1000,

                            random_state = 50, 

                            warm_start = False)
rf.fit(X_train, y_train)
pred = rf.predict(X_test)
from sklearn.metrics import accuracy_score, confusion_matrix
accuracy_score(pred, y_test)*100
plot_confusion_matrix(y_test, pred)
predicted_probs = rf.predict_proba(X_test)
gini(predicted_probs[:,1])
predicted_probs_train = rf.predict_proba(X_train)
gini(predicted_probs_train[:,1])
predicted_train = rf.predict(X_train)
accuracy_score(predicted_train, y_train)
plot_confusion_matrix(y_train, predicted_train)