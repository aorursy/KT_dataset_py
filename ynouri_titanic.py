import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import model_selection
import matplotlib.pyplot as plt
%matplotlib inline
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
np.set_printoptions(precision=4, linewidth=300)
# ---- LOAD DATA ----
df_test = pd.read_csv('../input/test.csv', index_col='PassengerId')
df = pd.read_csv('../input/train.csv', index_col='PassengerId')
# ---- DATA EXPLORATION ---- 

agg = {'PassengerId': 'count', 'Survived': 'mean'}

if 0:
    # Display 3 random examples
    df.sample(3).style

if 1:
    # Statiscally describes the data set
    df.describe()
    
if 1:
    # Is there any missing data that would require filling?
    df.isnull().sum()

if 0:
    # Kids have a higher survival rate
    bins = np.linspace(0, 60, 7)
    group = pd.cut(df.Age, bins).cat.add_categories('nan').fillna('nan')
    df.groupby(group).agg(agg)

if 0:
    # Class and sex are also a big factor in survival
    groups = ['Pclass', 'Sex']
    df.groupby(groups).agg(agg)

if 0:
    # Cherbourg passengers have a slightly higher survival rate
    df.groupby('Embarked').agg(agg)

if 0:
    # Survival by fare: same impact as class
    agg = {'PassengerId': 'count', 'Survived': 'mean', 'Pclass': 'mean'}
    bins = np.sort(np.append(np.linspace(0, 100, 11), [200, 1000]))
    df['Fare'].hist(bins=bins)
    df.groupby(pd.cut(df.Fare, bins)).agg(agg)
    
if 0:
    # Being alone (Parch,SibSp)=(0,0) is bad
    # Being with your spouse, sibling, parent or children is good
    df.groupby(['Parch', 'SibSp']).agg(agg)

if 1:
    # What about ticket code? What about the cabin? The name?
    df.sample(5)
# ---- FEATURE ENGINEERING ----

# Prepare the feature and label vectors
def prepare_features(df):
    # Handle the test set case
    if 'Survived' not in df.columns: df_test['Survived'] = np.nan
    # Create dummies
    df_d = df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare','Embarked']]
    df_d = pd.get_dummies(df_d, columns=['Pclass', 'Sex', 'Embarked'])
    # Fill missing age and fare values with means
    df_d.Age = df_d.Age.fillna(df_d.Age.mean())
    df_d.Fare = df_d.Fare.fillna(df_d.Fare.mean())
    # Split the labels from the features
    x = df_d.drop('Survived', axis=1)
    y = df_d.Survived
    return [x, y]

# Training, continuous validation test sets
training_ratio = 0.7
df_train, df_cv = np.split(df.sample(frac=1), [int(training_ratio*len(df))])
print("Number of samples\nTraining + CV = {}\nTraining = {}\nContinuous validation = {}\nTest = {}".format(
    len(df), len(df_train), len(df_cv), len(df_test)))

# Prepare features
x_train, y_train = prepare_features(df_train)
x_cv, y_cv = prepare_features(df_cv)
x_test, _ = prepare_features(df_test)

# Display training set
if 0:
    x_train.sample(5)
    x_train.describe()
# ---- LOGISTIC REGRESSION ----

# Fit model
logistic = linear_model.LogisticRegression()
logistic.fit(x_train, y_train)

# Model accuracy
print("Score training set = {:.4f}\nScore test set = {:.4f}".format(
    logistic.score(x_train, y_train),
    logistic.score(x_cv, y_cv)
))
# Prediction probability according to age
# Age	Pclass_1	Pclass_2	Pclass_3	Sex_female	Sex_male
classes = {'First class': 'Pclass_1', 'Second class': 'Pclass_2', 'Third class': 'Pclass_3'}
plt.xlim(0,80)
plt.ylim(0,1)
plt.plot([0, 100], [0.5, 0.5], 'k--')
for k, v in classes.items():
    n = 50
    x_class_median = x_train[x_train[v] == 1].median()
    X = np.tile(x_class_median, (n,1))
    X[:, 0] = np.linspace(0., 80., n)
    survival = logistic.predict_proba(X)[:, 1]
    plt.plot(X[:, 0], survival)
# ---- PRODUCE CSV PREDICTION FILE ----
y_test = logistic.predict(x_test)
df_prediction = pd.DataFrame(data=y_test, columns=['Survived'], index=x_test.index)
df_prediction.to_csv('logistic_submission.csv')