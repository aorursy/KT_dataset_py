# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# create testing and training sets for hold-out verification using scikit learn method

# test classification dataset

from sklearn.datasets import make_classification

# evaluate a logistic regression model using k-fold cross-validation

from numpy import mean

from numpy import std

from sklearn.datasets import make_classification

from sklearn.model_selection import KFold

from sklearn.model_selection import RepeatedKFold

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LinearRegression

from sklearn.neural_network import MLPRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error



#import seaborn and plotly

import matplotlib

from matplotlib import pyplot as plt

import seaborn as sns



train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()
women = train_data.loc[train_data.Sex == 'female']["Survived"]

rate_women = sum(women)/len(women)



print("% of women who survived:", rate_women)



men = train_data.loc[train_data.Sex == 'male']["Survived"]

rate_men = sum(men)/len(men)



print("% of men who survived:", rate_men)
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
sex_pivot = train_data.pivot_table(index="Sex",values="Survived")

sex_pivot
survived = train_data[train_data["Survived"] == 1]

died = train_data[train_data["Survived"] == 0]

survived["Age"].plot.hist(alpha=0.5,color='red',bins=50)

died["Age"].plot.hist(alpha=0.5,color='blue',bins=50)

plt.legend(['Survived','Died'])

plt.show()



# example from https://www.kaggle.com/alexisbcook/titanic-tutorial

from sklearn.ensemble import RandomForestClassifier



y = train_data["Survived"]



features = ["Pclass", "Sex", "SibSp", "Parch"]

X = pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features])



model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X, y)

predictions = model.predict(X_test)



#output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

#output.to_csv('my_submission.csv', index=False)

#print("Your submission was successfully saved!")





predictions
train=train_data.copy()

test=test_data.copy()
import re
# Copy original dataset in case we need it later when digging into interesting features

# WARNING: Beware of actually copying the dataframe instead of just referencing it

# "original_train = train" will create a reference to the train variable (changes in 'train' will apply to 'original_train')

original_train = train.copy() # Using 'copy()' allows to clone the dataset, creating a different object with the same values



# Feature engineering steps taken from Sina and Anisotropic, with minor changes to avoid warnings

full_data = [train, test]



# Feature that tells whether a passenger had a cabin on the Titanic

train['Has_Cabin'] = train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

test['Has_Cabin'] = test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)



# Create new feature FamilySize as a combination of SibSp and Parch

for dataset in full_data:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

# Create new feature IsAlone from FamilySize

for dataset in full_data:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

# Remove all NULLS in the Embarked column

for dataset in full_data:

    dataset['Embarked'] = dataset['Embarked'].fillna('S')

# Remove all NULLS in the Fare column

for dataset in full_data:

    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())



# Remove all NULLS in the Age column

for dataset in full_data:

    age_avg = dataset['Age'].mean()

    age_std = dataset['Age'].std()

    age_null_count = dataset['Age'].isnull().sum()

    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)

    # Next line has been improved to avoid warning

    dataset.loc[np.isnan(dataset['Age']), 'Age'] = age_null_random_list

    dataset['Age'] = dataset['Age'].astype(int)



# Define function to extract titles from passenger names

def get_title(name):

    title_search = re.search(' ([A-Za-z]+)\.', name)

    # If the title exists, extract and return it.

    if title_search:

        return title_search.group(1)

    return ""



for dataset in full_data:

    dataset['Title'] = dataset['Name'].apply(get_title)

# Group all non-common titles into one single grouping "Rare"

for dataset in full_data:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')



for dataset in full_data:

    # Mapping Sex

    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

    

    # Mapping titles

    title_mapping = {"Mr": 1, "Master": 2, "Mrs": 3, "Miss": 4, "Rare": 5}

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)



    # Mapping Embarked

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

    

    # Mapping Fare

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] 						        = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] 							        = 3

    dataset['Fare'] = dataset['Fare'].astype(int)

    

    # Mapping Age

    dataset.loc[ dataset['Age'] <= 16, 'Age'] 					       = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age'] ;




# Feature selection: remove variables no longer containing relevant information

drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']

train = train.drop(drop_elements, axis = 1)

test  = test.drop(drop_elements, axis = 1)



colormap = plt.cm.viridis

plt.figure(figsize=(12,12))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
train[['Title', 'Survived']].groupby(['Title'], as_index=False).agg(['mean', 'count', 'sum'])

# Since "Survived" is a binary class (0 or 1), these metrics grouped by the Title feature represent:

    # MEAN: survival rate

    # COUNT: total observations

    # SUM: people survived



# title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5} 




train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).agg(['mean', 'count', 'sum'])

# Since Survived is a binary feature, this metrics grouped by the Sex feature represent:

    # MEAN: survival rate

    # COUNT: total observations

    # SUM: people survived

    

# sex_mapping = {{'female': 0, 'male': 1}} 







# Let's use our 'original_train' dataframe to check the sex distribution for each title.

# We use copy() again to prevent modifications in out original_train dataset

title_and_sex = original_train.copy()[['Name', 'Sex']]



# Create 'Title' feature

title_and_sex['Title'] = title_and_sex['Name'].apply(get_title)



# Map 'Sex' as binary feature

title_and_sex['Sex'] = title_and_sex['Sex'].map( {'female': 0, 'male': 1} ).astype(int)



# Table with 'Sex' distribution grouped by 'Title'

title_and_sex[['Title', 'Sex']].groupby(['Title'], as_index=False).agg(['mean', 'count', 'sum'])



# Since Sex is a binary feature, this metrics grouped by the Title feature represent:

    # MEAN: percentage of men

    # COUNT: total observations

    # SUM: number of men



import numpy as np

import pandas as pd

import re

import xgboost as xgb

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls



from sklearn import tree

from sklearn.metrics import accuracy_score

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from IPython.display import Image as PImage

from subprocess import check_call

from PIL import Image, ImageDraw, ImageFont
train = pd.read_csv("/kaggle/input/titanic/train.csv")

test = pd.read_csv("/kaggle/input/titanic/test.csv")
data=train.copy()

data.describe()
from sklearn.model_selection import train_test_split





# setting up new data types

dtypes_col       = data.columns

dtypes_type_old  = data.dtypes

dtypes_type      = ['int16', 'bool','category','object','category','float32','int8','int8','object','float32','object','category']

optimized_dtypes = dict(zip(dtypes_col, dtypes_type))



#read data once again with optimized columns

data_optimized = pd.read_csv("/kaggle/input/titanic/train.csv",dtype=optimized_dtypes)

test_optimized = pd.read_csv("/kaggle/input/titanic/test.csv",dtype=optimized_dtypes)



#splitting data to train and validation

train, valid = train_test_split(data_optimized, test_size=0.2)





combined = {"train":train,

            "valid":valid,

            "test":test_optimized}



print(data_optimized.info())




data_optimized.isnull().sum()



combined_cleaned = {}

for i,data in combined.items():

    combined_cleaned[i] = data.drop('Cabin', 1).copy()
#numerical features



train_numeric = combined_cleaned["train"].select_dtypes(include=['float32','int16','int8','bool'])



colormap = plt.cm.cubehelix_r

plt.figure(figsize=(16,12))



plt.title('Pearson correlation of numeric features', y=1.05, size=15)

sns.heatmap(train_numeric.corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
# category features# category features



#we do not count NaN categories

def survived_percent(categories,column):

    survived_list = []

    for c in categories.dropna():

        count = combined_cleaned["train"][combined_cleaned["train"][column] == c][column].count()

        survived = combined_cleaned["train"][combined_cleaned["train"][column] == c]["Survived"].sum()/count

        survived_list.append(survived)

    return survived_list    

   

category_features_list = ["Sex", "Embarked","Pclass"]

category_features = {}



for x in category_features_list:

    unique_values = combined_cleaned["train"][x].unique().dropna()

    survived = survived_percent(unique_values,x)

    category_features[x] = [unique_values, survived]





fig, axs = plt.subplots(1, 3, figsize=(18, 4), sharey=True)

cb_dark_blue = (0/255,107/255,164/255)

cb_orange = (255/255, 128/255, 14/255)

cb_grey = (89/255, 89/255, 89/255)

color=[cb_dark_blue,cb_orange,cb_grey]



font_dict = {'fontsize':20, 

             'fontweight':'bold',

             'color':"white"}



for i,cat in enumerate(category_features.keys()):

    number_categories = len(category_features[cat][0])

    axs[i].bar(range(number_categories), category_features[cat][1], color=color[:number_categories])

    axs[i].set_title("Survival rate " + cat ,fontsize=20, fontweight='bold' )

    for j,indx in enumerate(category_features[cat][1]):

        label_text = category_features[cat][0][j]

        x = j

        y = indx

        axs[i].annotate(label_text, xy = (x-0.15 ,y/2), **font_dict )



for i in range(3):

    axs[i].tick_params(

        axis='x',          # changes apply to the x-axis

        which='both',      # both major and minor ticks are affected

        bottom='off',      # ticks along the bottom edge are off

        top='off',         # ticks along the top edge are off

        labelbottom='off') # labels along the bottom edge are off

    axs[i].patch.set_visible(False)




# filling NaN in "Embarked" and "Fare"



for i,data in combined_cleaned.items():

    data["Embarked"].fillna(value="S",inplace=True) # S is most popular value 

    mean_Fare = data["Fare"].mean()

    data["Fare"].fillna(value=mean_Fare,inplace=True)



%matplotlib inline



import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt





# filling NaN in "Age" 

fig, ax = plt.subplots( figsize=(6,4))

x = combined_cleaned["train"]["Age"].dropna()

hist, bins = np.histogram( x,bins=15)



#plot of histogram

ax.hist(x, color='grey')

ax.set_title('Age histogram')

plt.show()







from random import choices



bin_centers = 0.5*(bins[:len(bins)-1]+bins[1:])

probabilities = hist/hist.sum()



#dictionary with random numbers from existing age distribution

for i,data in combined_cleaned.items():

    data["Age_rand"] = data["Age"].apply(lambda v: np.random.choice(bin_centers, p=probabilities))

    Age_null_list   = data[data["Age"].isnull()].index

    

    data.loc[Age_null_list,"Age"] = data.loc[Age_null_list,"Age_rand"]

    







from sklearn import preprocessing,tree

from sklearn.model_selection import GridSearchCV



tree_data = {}

tree_data_category = {}



for i,data in combined_cleaned.items():

    tree_data[i] = data.select_dtypes(include=['float32','int16','int8']).copy()

    tree_data_category[i] = data.select_dtypes(include=['category'])



    #categorical variables handling

    for column in tree_data_category[i].columns:

        le = preprocessing.LabelEncoder()

        le.fit(data[column])

        tree_data[i][column+"_encoded"] = le.transform(data[column])



#finding best fit with gridsearch

param_grid = {'min_samples_leaf':np.arange(20,50,5),

              'min_samples_split':np.arange(20,50,5),

              'max_depth':np.arange(3,6),

              'min_weight_fraction_leaf':np.arange(0,0.4,0.1),

              'criterion':['gini','entropy']}

clf = tree.DecisionTreeClassifier()

tree_search = GridSearchCV(clf, param_grid, scoring='average_precision')



X =  tree_data["train"].drop("PassengerId",axis=1)

Y = combined_cleaned["train"]["Survived"]

tree_search.fit(X,Y)



print("Tree best parameters :",tree_search.best_params_)

print("Tree best estimator :",tree_search.best_estimator_ )

print("Tree best score :",tree_search.best_score_ )
tree_best_parameters = tree_search.best_params_

tree_optimized = tree.DecisionTreeClassifier(**tree_best_parameters)

tree_optimized.fit(X,Y)



train_columns = list(tree_data["train"].columns)

train_columns.remove("PassengerId")

fig, ax = plt.subplots( figsize=(6,4))

ax.bar(range(len(X.columns)),tree_optimized.feature_importances_ )

plt.xticks(range(len(X.columns)),X.columns,rotation=90)

ax.set_title("Feature importance")

plt.show()
import graphviz 



dot_data = tree.export_graphviz(tree_optimized, 

                                out_file=None,

                                filled=True, 

                                rounded=True,  

                                special_characters=True,

                               feature_names = train_columns) 

graph = graphviz.Source(dot_data)

graph
test_without_PId = tree_data["test"].drop("PassengerId",axis=1)

prediction_values = tree_optimized.predict(test_without_PId).astype(int)

prediction = pd.DataFrame({"PassengerId":tree_data["test"]["PassengerId"],

                           "Survived":prediction_values})



prediction.head()

prediction.to_csv("Titanic_tree_prediction.csv",index=False)




from sklearn.metrics import confusion_matrix



evaluation = {}

cm = {}





valid_without_PId = tree_data["valid"].drop("PassengerId",axis=1)

evaluation["tree"] = tree_optimized.predict(valid_without_PId).astype(int)

survival_from_data = combined_cleaned["valid"]["Survived"].astype(int)



print(survival_from_data.value_counts())



cm["tree"] = confusion_matrix(survival_from_data, evaluation["tree"])

cm["tree"] = cm["tree"].astype('float') / cm["tree"].sum(axis=1)[:, np.newaxis]



cm["tree"]



import itertools



def plot_confusion_matrix(cm, classes,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    print(cm)



    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    fmt = '.2f'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')




from sklearn.ensemble import RandomForestClassifier



clf_forest = RandomForestClassifier(n_estimators=10,min_samples_leaf=20, max_depth=4,min_weight_fraction_leaf=0.1)

clf_forest.fit(X,Y)
prediction_values_forest = clf_forest.predict(test_without_PId).astype(int)

prediction_forest = pd.DataFrame({"PassengerId":tree_data["test"]["PassengerId"],

                           "Survived":prediction_values_forest})




param_grid = {'n_estimators':np.arange(3,11,2),

              'max_depth':np.arange(3,6),

              'min_weight_fraction_leaf':np.arange(0,0.4,0.1),

              'criterion':['gini','entropy']}

clf = RandomForestClassifier()

forest_search = GridSearchCV(clf, param_grid, scoring='precision')



forest_search.fit(X,Y)



print("Forest best parameters :",forest_search.best_params_)

print("Forest best estimator :",forest_search.best_estimator_ )

print("Forest best score :",forest_search.best_score_ )



clf_forest = RandomForestClassifier(**forest_search.best_params_)

clf_forest.fit(X,Y)
prediction_values_forest = clf_forest.predict(test_without_PId).astype(int)

prediction_forest = pd.DataFrame({"PassengerId":tree_data["test"]["PassengerId"],

                           "Survived":prediction_values_forest})



prediction_forest.to_csv("Titanic_tree_prediction_forest.csv",index=False)




evaluation["forest"] = clf_forest.predict(valid_without_PId).astype(int)



cm["forest"] = confusion_matrix(survival_from_data, evaluation["forest"])

cm["forest"] = cm["forest"].astype('float') / cm["forest"].sum(axis=1)[:, np.newaxis]



cm["forest"]



plot_confusion_matrix(cm["forest"], classes=["No","Yes"], 

                      title='Normalized confusion matrix')



from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

gnb.fit(X, Y)
prediction_values_NaiveBayes = gnb.predict(test_without_PId).astype(int)

prediction_NaiveBayes = pd.DataFrame({"PassengerId":tree_data["test"]["PassengerId"],

                           "Survived":prediction_values_NaiveBayes})
print(prediction_values_NaiveBayes)
y_train = train[['Survived']].astype(int)

print(y_train)




evaluation["NB"] = gnb.predict(valid_without_PId).astype(int)



cm["NB"] = confusion_matrix(survival_from_data, evaluation["NB"])

cm["NB"] = cm["NB"].astype('float') / cm["NB"].sum(axis=1)[:, np.newaxis]



cm["NB"]



plot_confusion_matrix(cm["NB"], classes=["No","Yes"], 

                      title='Normalized confusion matrix')



from sklearn import svm

clf_svm = svm.SVC()

clf_svm.fit(X,Y)




prediction_values_svm = clf_svm.predict(test_without_PId).astype(int)

prediction_svm = pd.DataFrame({"PassengerId":tree_data["test"]["PassengerId"],

                           "Survived":prediction_values_svm})
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib.pyplot import rcParams

%matplotlib inline

rcParams['figure.figsize'] = 10,8

sns.set(style='whitegrid', palette='muted',

        rc={'figure.figsize': (15,10)})

import os

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from keras.wrappers.scikit_learn import KerasClassifier

from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout



from numpy.random import seed

 
data = pd.read_csv("/kaggle/input/titanic/train.csv")

df=data.copy()

data.describe(include='all')




sns.countplot(x='Pclass', data=df, palette='hls', hue='Survived')

plt.xticks(rotation=45)

plt.show()







sns.countplot(x='Sex', data=df, palette='hls', hue='Survived')

plt.xticks(rotation=45)

plt.show()



# convert to cateogry dtype

df['Sex'] = df['Sex'].astype('category')

# convert to category codes

df['Sex'] = df['Sex'].cat.codes
df.head()




# drop the variables we won't be using

df.drop(['Cabin', 'Name', 'Ticket', 'PassengerId', 'Embarked'], axis=1, inplace=True)



df.head()
continuous = ['Age', 'Fare', 'Parch', 'Pclass', 'SibSp']



scaler = StandardScaler()



for var in continuous:

    df[var] = df[var].astype('float64')

    df[var] = scaler.fit_transform(df[var].values.reshape(-1, 1))




X_train = df[pd.notnull(df['Survived'])].drop(['Survived'], axis=1)

y_train = df[pd.notnull(df['Survived'])]['Survived']

X_test = df[pd.isnull(df['Survived'])].drop(['Survived'], axis=1)



def create_model(lyrs=[8], act='linear', opt='Adam', dr=0.0):

    

    # set random seed for reproducibility

    seed(42)

    

    model = Sequential()

    

    # create first hidden layer

    model.add(Dense(lyrs[0], input_dim=X_train.shape[1], activation=act))

    

    # create additional hidden layers

    for i in range(1,len(lyrs)):

        model.add(Dense(lyrs[i], activation=act))

    

    # add dropout, default is none

    model.add(Dropout(dr))

    

    # create output layer

    model.add(Dense(1, activation='sigmoid'))  # output layer

    

    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    

    return model
model = create_model()

print(model.summary())
# train model on full train set, with 80/20 CV split

#training = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)

training = model.fit(X_train, y_train, validation_split=0.33, epochs=150, batch_size=10, verbose=0)
print(training.history.keys())






val_acc = np.mean(training.history['val_accuracy'])

print("\n%s: %.2f%%" % ('val_acc', val_acc*100))



# list all data in history

print(training.history.keys())

# summarize history for accuracy

plt.plot(training.history['accuracy'])

plt.plot(training.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(training.history['loss'])

plt.plot(training.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()