# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Mute the sklearn warning

import warnings

warnings.filterwarnings('ignore', module='sklearn')

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import numpy as np

from sklearn.metrics import confusion_matrix, roc_auc_score ,roc_curve,auc

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import LabelBinarizer

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import train_test_split

from sklearn.model_selection import learning_curve

from sklearn.model_selection import GridSearchCV

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/Orange_Telecom_Churn_Data.csv")
data.head().T
# Check columns data type 

data.dtypes.value_counts()
data.state.value_counts().plot(kind = 'bar', figsize = [14,6], color = 'indianred')

plt.title('Checking State Level')

plt.xlabel('State')

plt.ylabel('Count');
print("State Level: {}".format(len(data.state.value_counts())))

print("Area Code Level: {}".format(len(data.area_code.value_counts())))

print("Phone Number Level: {}".format(len(data.phone_number.value_counts())))
# Remove extraneous columns

data.drop(['state', 'area_code', 'phone_number'], axis=1, inplace=True)
data.columns
# Checking shape 

data.shape
plt.figure(figsize=(10,6))

ax = sns.countplot(x = 'churned', data = data)

ax.set_ylim(top=5000)

for p in ax.patches:

    ax.annotate('{:.2f}%'.format(100*p.get_height()/len(data.churned)), (p.get_x()+ 0.3, p.get_height()+200))



plt.title('Distribution of Churned (Target)')

plt.xlabel('Churned')

plt.ylabel('Count');
plt.figure(figsize=(10,6))

data.total_day_calls.plot(kind = 'hist', bins = 50, color = 'indianred')

plt.title('Total Calls in Day', fontsize = 16)

plt.xlabel('Number of calls');
# Subplot 

plt.figure(figsize=(12,8))



plt.subplot(221)

sns.scatterplot(x='total_day_charge', y= 'total_eve_charge', data = data, hue = 'churned' )



plt.subplot(222)

sns.scatterplot(x='total_night_calls', y= 'total_day_calls', data = data, hue = 'churned' )



plt.subplot(223)

sns.scatterplot(x='total_intl_calls', y= 'total_intl_charge', data = data, hue = 'churned' )



plt.subplot(224)

sns.scatterplot(x='account_length', y= 'total_day_calls', data = data, hue = 'churned' )





plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,

                    wspace=0.35);

sns.set(style='white')



# Compute the correlation matrix

corr = data.loc[:,[i for i in list(data.columns) if i not in ['churned']]].corr()



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})



plt.show()
lb = LabelBinarizer()

for col in ['intl_plan', 'voice_mail_plan', 'churned']:

    data[col] = lb.fit_transform(data[col])
msc = MinMaxScaler()

data = pd.DataFrame(msc.fit_transform(data),  

                    columns=data.columns)
# Get a list of all the columns that don't contain the label

x_cols = [x for x in data.columns if x != 'churned']



# Split the data into two dataframes

X_data = data[x_cols]

y_data = data['churned']
# Train the KNN model and fit on training set

knn = KNeighborsClassifier(n_neighbors=3)

knn = knn.fit(X_data, y_data)

y_pred = knn.predict(X_data)
# Function to calculate the % of values that were correctly predicted

def accuracy(real, predict):

    return sum(y_data == y_pred) / float(real.shape[0])
print(accuracy(y_data, y_pred))
print(roc_auc_score(y_data, knn.predict_proba(X_data)[::,1]))
# Create trainset and testset 

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.30, random_state=42)
# Run the model on 5 StratifiedKFold and check ROC AUC score

kf = StratifiedKFold(n_splits=5,shuffle=False,random_state=121)

pred_test_full =0

cv_score =[]

i=1

for train_index,test_index in kf.split(X_data,y_data):

    print('{} of KFold {}'.format(i,kf.n_splits))

    xtr,xvl = X_data.loc[train_index],X_data.loc[test_index]

    ytr,yvl = y_data.loc[train_index],y_data.loc[test_index]

    

    #model

    knn = KNeighborsClassifier()

    knn.fit(xtr,ytr)

    score = roc_auc_score(yvl,knn.predict_proba(xvl)[::,1])

    print('ROC AUC score:',score)

    cv_score.append(score)    

    
print("Average CV ROC AUC score: {}".format(np.mean(cv_score)))
# Check model performance with different number of n_neighbours

for p in [1,2]:

    if p == 1:

        accuracies_1 = [] 

        number = []

        for i in range(1,21):

            knn = KNeighborsClassifier(n_neighbors=i, weights='uniform', p = p)

            knn = knn.fit(X_train, y_train)

            y_pred = knn.predict(X_test)

            acc = roc_auc_score(y_test, y_pred)

            accuracies_1.append(acc)

            number.append(i)

    else:

        accuracies_2 = [] 

        for i in range(1,21):

            knn = KNeighborsClassifier(n_neighbors=i, weights='uniform', p = p)

            knn = knn.fit(X_data, y_data)

            y_pred = knn.predict(X_data)

            acc = accuracy(y_data, y_pred)

            accuracies_2.append(acc)

result1 = pd.DataFrame({'NumberOfk':number,'Accuracy_1':accuracies_1})

result2 = pd.DataFrame({'NumberOfk':number,'Accuracy_2':accuracies_2})



final = pd.merge(result1, result2)



plt.figure(figsize=(12,6))

final.Accuracy_1.plot(label='1')

final.Accuracy_2.plot(label='2')

plt.legend()

plt.xlabel('n_neighbours')

plt.ylabel('Accuracy');
def plot_learning_curve(model, x, y ):

    # Learning curve 

    plt.figure(figsize=(12,6))

    train_sizes, train_scores, valid_scores = learning_curve(model, x, y, scoring = 'accuracy', n_jobs = -1, train_sizes=np.linspace(0.01, 1.0, 20))

    # Create means and standard deviations of training set scores

    train_mean = np.mean(train_scores, axis=1)

    train_std = np.std(train_scores, axis=1)

    # Create means and standard deviations of test set scores

    test_mean = np.mean(valid_scores, axis=1)

    test_std = np.std(valid_scores, axis=1)

    # Draw lines

    plt.plot(train_sizes, train_mean, '--', color="green",  label="Training score")

    plt.plot(train_sizes, test_mean, color="brown", label="Cross-validation score")

    # Draw bands

    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")

    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")

    # Create plot

    plt.title("Learning Curve", fontsize = 16)

    plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")

    plt.tight_layout()

    plt.show()
knn = KNeighborsClassifier()

plot_learning_curve(knn, X_train,y_train)
# Grid search for KNNclassifier tuning 

classifier = KNeighborsClassifier()

parameters = [{'metric':['minkowski','euclidean','manhattan'],

               'weights': ['uniform','distance'], 

               'n_neighbors':  np.arange(1,10)}]



grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy', cv = 5, n_jobs = -1)

grid_search = grid_search.fit(X_train, y_train)

best_accuracy = grid_search.best_score_

best_parameters = grid_search.best_params_
# Check hyperparameters

print(best_parameters )
# Run the model on 5 StratifiedKFold and check ROC AUC score

kf = StratifiedKFold(n_splits=5,shuffle=False,random_state=121)

pred_test_full =0

cv_score =[]

i=1

for train_index,test_index in kf.split(X_data,y_data):

    print('{} of KFold {}'.format(i,kf.n_splits))

    xtr,xvl = X_data.loc[train_index],X_data.loc[test_index]

    ytr,yvl = y_data.loc[train_index],y_data.loc[test_index]

    

    #model

    knn = KNeighborsClassifier(n_neighbors=8, weights='distance', metric = 'manhattan')

    knn.fit(xtr,ytr)

    score = roc_auc_score(yvl,knn.predict_proba(xvl)[::,1])

    print('ROC AUC score:',score)

    cv_score.append(score)    

print("Average CV ROC AUC score: {}".format(np.mean(cv_score)))
knn = KNeighborsClassifier(n_neighbors=8, weights='distance', metric = 'manhattan')

plot_learning_curve(knn, X_train,y_train)