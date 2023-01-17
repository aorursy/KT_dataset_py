# Import modules
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.metrics import accuracy_score

# Figures inline and set visualization style
%matplotlib inline
sns.set()
# Import test and train datasets
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')

# View first lines of training data
df_train.head(3)
# View first lines of test data
df_test.head(3)
df_train.info()
df_train.describe()
sns.countplot(x='Survived', data=df_train)
df_test['Survived'] = 0
df_test[['PassengerId', 'Survived']].to_csv('results/no_survivors.csv', index=False)
sns.countplot(x='Sex', data=df_train);
# kind is the facets
sns.factorplot(x='Survived', col='Sex', kind='count', data=df_train)
df_train.head(1)
# Chain a group by Sex, sum Survived
df_train.groupby(['Sex']).Survived.sum()
# Chain calculations
print(df_train[df_train.Sex == 'female'].Survived.sum() /
      df_train[df_train.Sex == 'female'].Survived.count())

print(df_train[df_train.Sex == 'male'].Survived.sum() /
      df_train[df_train.Sex == 'male'].Survived.count())
df_test['Survived'] = df_test.Sex == 'female'
df_test['Survived'] = df_test.Survived.apply(lambda x: int(x))
df_test.head(3)
df_test[['PassengerId', 'Survived']].to_csv('results/women_survived.csv', index=False)
# kind is the facets
sns.factorplot(x='Survived', col='Pclass', kind='count', data=df_train)
# kind is the facets
sns.factorplot(x='Survived', col='Embarked', kind='count', data=df_train)
sns.distplot(df_train.Fare, kde=False)
# Group by Survived, trace histograms of Fare with alpha color 0.6
df_train.groupby('Survived').Fare.hist(alpha=0.6)
# Remove NaN
df_train_drop = df_train.dropna()

sns.distplot(df_train_drop.Age, kde=False)
# Alternative to bars or scatter
sns.stripplot(x='Survived', 
              y='Fare', 
              data=df_train, 
              alpha=0.3, jitter=True)
# Alternative to bars or scatter
sns.swarmplot(x='Survived', 
              y='Fare', 
              data=df_train)
# Group by Survived, describe Fare (descriptive statistics)
df_train.groupby('Survived').Fare.describe()
sns.lmplot(x='Age', 
           y='Fare', 
           hue='Survived', 
           data=df_train, 
           fit_reg=False, scatter_kws={'alpha':0.5})
sns.lmplot(x='Age', 
           y='Fare', 
           hue='Survived', 
           data=df_train, 
           fit_reg=True, scatter_kws={'alpha':0.5})
sns.pairplot(df_train_drop, hue='Survived')
# Import modules
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Figures inline and set visualization style
%matplotlib inline
sns.set()
# Import data
df_train = pd.read_csv('data/train.csv')
df_test = pd.read_csv('data/test.csv')
df_train.info()
df_test.info()
# Store target variable of training data in a safe place
survived_train = df_train.Survived

# Concatenate (along the index or axis=1) training and test sets
# to preprocess the data a little bit
# and make sure that any operations that
# we perform on the training set are also
# being done on the test data set
data = pd.concat([df_train.drop(['Survived'], axis=1), df_test])
# The combined datasets (891+418 entries)
data.info()
# Impute missing numerical variables where NaN
data['Age'] = data.Age.fillna(data.Age.median())
data['Fare'] = data.Fare.fillna(data.Fare.median())

# Check out info of data
data.info()
data = pd.get_dummies(data, columns=['Sex'], drop_first=True)
data.head(3)
# Select columns and view head
data = data[['Sex_male', 'Fare', 'Age','Pclass', 'SibSp']]
data.head(3)
data.info()
data_train = data.iloc[:891]
data_test = data.iloc[891:]
X = data_train.values
test = data_test.values

# and from above: survived_train = df_train.Survived
y = survived_train.values
X
# Instantiate model and fit to data
# The max depth is set at 3
clf = tree.DecisionTreeClassifier(max_depth=3)

# X is the indenpendent variables, y is the dependent variable
clf.fit(X, y)
# Make predictions and store in 'Survived' column of df_test
Y_pred = clf.predict(test)
df_test['Survived'] = Y_pred

# Save it
df_test[['PassengerId', 'Survived']].to_csv('results/1st_dec_tree.csv',
                                            index=False)
# Compute accuracy on the training set
train_accuracy = clf.score(X, y)
train_accuracy
import graphviz

tree_data = tree.export_graphviz(clf, out_file=None) 
graph = graphviz.Source(tree_data)
# Save the pdf
graph.render("img/tree_data")
feature_names = list(data_train)
feature_names
#data_train
#data_test
tree_data = tree.export_graphviz(clf, out_file=None, 
                                feature_names=feature_names,
                                class_names=None,
                                filled=True, rounded=True,
                                special_characters=True)  
graph = graphviz.Source(tree_data)  
graph 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)
# Setup arrays to store train and test accuracies
dep = np.arange(1, 9)
train_accuracy = np.empty(len(dep))
test_accuracy = np.empty(len(dep))

# Loop over different values of k
for i, k in enumerate(dep):
    # Setup a k-NN Classifier with k neighbors: knn
    clf = tree.DecisionTreeClassifier(max_depth=k)

    # Fit the classifier to the training data
    clf.fit(X_train, y_train)

    # Compute accuracy on the training set
    train_accuracy[i] = clf.score(X_train, y_train)

    # Compute accuracy on the testing set
    test_accuracy[i] = clf.score(X_test, y_test)

# Generate plot
plt.title('clf: Varying depth of tree')
plt.plot(dep, test_accuracy, label = 'Testing Accuracy')
plt.plot(dep, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Depth of tree')
plt.ylabel('Accuracy')
plt.show()
# Imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
from sklearn import tree
from sklearn.model_selection import GridSearchCV

# Figures inline and set visualization style
%matplotlib inline
sns.set()
# Import data
df_train = pd.read_csv('data/train.csv')
df_test = pd.read_csv('data/test.csv')

# Store target variable of training data in a safe place
survived_train = df_train.Survived

# Concatenate training and test sets
data = pd.concat([df_train.drop(['Survived'], axis=1), df_test])

# View head
data.info()
# View head of 'Name' column
data.Name.tail()
# Extract Title from Name, store in column and plot barplot
# One upper character, one lower character, one dot
data['Title'] = data.Name.apply(lambda x: re.search(' ([A-Z][a-z]+)\.', x).group(1))
# New column Title is a new feature of the dataset 
data.Title.head(3)
sns.countplot(x='Title', data=data);
plt.xticks(rotation=45);
# Substitute some title with their English form
data['Title'] = data['Title'].replace({'Mlle':'Miss', 'Mme':'Mrs', 'Ms':'Miss'})
# Gather exceptions
data['Title'] = data['Title'].replace(['Don', 'Dona', 'Rev', 'Dr', 'Major', 'Lady', 'Sir', 'Col', 'Capt', 'Countess', 'Jonkheer'],'Special')
data.Title.head(3)
sns.countplot(x='Title', data=data);
plt.xticks(rotation=45);
# View tail of data (for change)
data.tail(3)
# View head of data
data[['Name', 'PassengerId', 'Ticket', 'Cabin']].head()
# Did they have a Cabin?
# Return True is the passenger has a cabin
data['Has_Cabin'] = ~data.Cabin.isnull()

# # View head of data
data[['Name', 'PassengerId', 'Ticket', 'Cabin', 'Has_Cabin']].head()
# Drop columns and view head
data.drop(['Cabin', 'Name', 'PassengerId', 'Ticket'], axis=1, inplace=True)
data.head()
data.info()
# Impute missing values for Age, Fare, Embarked
data['Age'] = data.Age.fillna(data.Age.median())
data['Fare'] = data.Fare.fillna(data.Fare.median())
data['Embarked'] = data['Embarked'].fillna('S')
data.info()
data.head(3)
# Binning numerical columns
# q=4 means 4 quantiles 0, 1, 2, 3
# labels=False are numbers, not characters
data['CatAge'] = pd.qcut(data.Age, q=4, labels=False )
data['CatFare']= pd.qcut(data.Fare, q=4, labels=False)
data.head(3)
# Drop the 'Age' and 'Fare' columns
data = data.drop(['Age', 'Fare'], axis=1)
data.head(3)
# Create column of number of Family members onboard
data['Fam_Size'] = data.Parch + data.SibSp

# Drop columns
data = data.drop(['SibSp','Parch'], axis=1)
data.head(3)
# Transform into binary variables
# Has_Cabin is a boolean
# Sex becomes Sex_male=1 or 0
# Embarked becomes Embarked_Q=1 or 0, Embarked_...
# Title becomes Title_Miss=1 or 0, ...
# The former variables are dropped, only the later variables remain
data_dum = pd.get_dummies(data, drop_first=True)
data_dum.head(3)
# Split into test.train
data_train = data_dum.iloc[:891]
data_test = data_dum.iloc[891:]

# Transform into arrays for scikit-learn
X = data_train.values
test = data_test.values
y = survived_train.values
# Setup the hyperparameter grid
dep = np.arange(1,9)
param_grid = {'max_depth' : dep}

# Instantiate a decision tree classifier: clf
clf = tree.DecisionTreeClassifier()

# Instantiate the GridSearchCV object: clf_cv
clf_cv = GridSearchCV(clf, param_grid=param_grid, cv=5)

# Fit it to the data
clf_cv.fit(X, y)

# Print the tuned parameter and score
print("Tuned Decision Tree Parameters: {}".format(clf_cv.best_params_))
print("Best score is {}".format(clf_cv.best_score_))
Y_pred = clf_cv.predict(test)
df_test['Survived'] = Y_pred
df_test[['PassengerId', 'Survived']].to_csv('results/dec_tree_feat_eng.csv', index=False)