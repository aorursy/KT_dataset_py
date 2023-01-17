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
df = pd.read_csv('/kaggle/input/churn-modelling/Churn_Modelling.csv')
df.head()
df.info()

print("Shape of Data: ", df.shape)
df.describe()
import seaborn as sns

import matplotlib.pyplot as plt 

import plotly

import plotly.express as px

from plotly.offline import init_notebook_mode, iplot

from IPython.display import display, HTML

plotly.offline.init_notebook_mode(connected = True)
plt.figure(figsize=(15,15))

sns.heatmap(df.corr(), annot=True, cmap='viridis')
# Create Input Features, and Output Feature



X = df.iloc[:, 3:-1].values

y = df.iloc[:, -1].values



print("Input Features: ", X)

print("Target Variable: ", y)
# We can observe that the highest number of customers are from France, while Germany and Spain have almost equal number of customers, but France has almost double than them

country = df.groupby(['Geography']).count()

country.RowNumber
sns.countplot(data=df, x='Geography')
# The number of males is greater than the number of females

sns.countplot(data=df, x='Gender')



# Nearly 1000 more males are present in the subscription than females

genders = df.groupby(['Gender']).count()

print(genders)
# So, we can say that the users are almost equitably distributed except for new users and very loyal users (10 years)

sns.countplot(data=df, x='Tenure')
# The median of the data is clearly 37 years of age and the upper-quartile limit is 62 years, rest are outliers

fig = px.box(df, y="Age")

fig.show()
# So, there are only 359 users greater than age of 62 using the platform

total_aged = (df["Age"]>62)

total_aged.value_counts()
# The user base below 32 years old is substantial

total_young = (df["Age"]<32)

total_young.value_counts()
# The estimated salaries are within the range of nearly 1000000 for majority of cases

plt.figure(figsize=(20,20))

sns.catplot(x="Geography", y="EstimatedSalary", hue="Gender", kind="box", data=df)

plt.title("Geography VS Estimated Salary")

plt.xlabel("Geography")

plt.ylabel("Estimated Salary")
# Most users in all the countries are within the age of 65 at max

fig = px.box(df, x="Age", y="Geography", notched=True)

fig.show()
fig = px.parallel_categories(df, dimensions=['HasCrCard', 'IsActiveMember'],

                 color_continuous_scale=px.colors.sequential.Inferno,

                labels={'Gender':'Sex', 'HasCrCard':'Credit Card Holder', 'IsActiveMember':'Activity Status'})

fig.show()
fig = px.parallel_categories(df, dimensions=['HasCrCard', 'Gender','IsActiveMember'],

                 color_continuous_scale=px.colors.sequential.Inferno,

                labels={'Gender':'Sex', 'HasCrCard':'Credit Card Holder', 'IsActiveMember':'Activity Status'})

fig.show()
fig = px.scatter_matrix(df,

    dimensions=["Age"],

    color="Exited")

fig.show()
# Label Encoding the "Gender" column

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

X[:, 2] = label_encoder.fit_transform(X[:, 2])

print(X)

# One Hot Encoding the "Geography" column

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder

coltrans = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')

X = np.array(coltrans.fit_transform(X))

print(X.shape)
# Standardize all the values in order to make them comparable

from sklearn.preprocessing import StandardScaler

stand_sc = StandardScaler()

X = stand_sc.fit_transform(X)

print(X)
# To balance the number of males, and females in the data along with countries to prevent overfitting

from imblearn.over_sampling import SMOTE

k = 1

seed=100

sm = SMOTE(sampling_strategy='auto', k_neighbors=k, random_state=seed)

X_res, y_res = sm.fit_resample(X, y)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size = 0.3, random_state = 0)
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report, roc_curve



def evaluation(y_test, clf, X_test):

    """

        Method to compute the following:

            1. Classification Report

            2. F1-score

            3. AUC-ROC score

            4. Accuracy

        Parameters:

            y_test: The target variable test set

            grid_clf: Grid classifier selected

            X_test: Input Feature Test Set

    """

    y_pred = clf.predict(X_test)

    print('CLASSIFICATION REPORT')

    print(classification_report(y_test, y_pred))

    

    print('AUC-ROC')

    print(roc_auc_score(y_test, y_pred))

      

    print('F1-Score')

    print(f1_score(y_test, y_pred))

    

    print('Accuracy')

    print(accuracy_score(y_test, y_pred))

# looking at the importance of each feature

def feature_importance(model):

    importances=model.feature_importances_



    # visualize to see the feature importance

    indices=np.argsort(importances)[::-1]

    plt.figure(figsize=(20,10))

    plt.bar(range(X.shape[1]), importances[indices])

    plt.show()

    

def plot_loss(model):

    prob=model.predict_proba(X_test)[:,1]



    fpr, tpr, thresholds=roc_curve(y_test, prob)

    plt.plot(fpr, tpr)



    auc=roc_auc_score(y_test, prob)

    print(auc)
from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier()

dt_model.fit(X_train, y_train)

evaluation(y_test, dt_model, X_test)

feature_importance(dt_model)

plot_loss(dt_model)
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=100)

rf_model.fit(X_train, y_train)

evaluation(y_test, rf_model, X_test)

feature_importance(rf_model)

plot_loss(rf_model)
from sklearn.svm import SVC

svc_model=SVC(probability=True)

svc_model.fit(X_train, y_train)

evaluation(y_test, svc_model, X_test)

plot_loss(svc_model)
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression()

lr_model.fit(X_train, y_train)

evaluation(y_test, lr_model, X_test)

plot_loss(lr_model)
from xgboost import XGBClassifier

xg_model = XGBClassifier()

xg_model.fit(X_train, y_train)

evaluation(y_test, xg_model, X_test)

feature_importance(xg_model)

plot_loss(xg_model)
from sklearn.ensemble import GradientBoostingClassifier

gb_model = GradientBoostingClassifier()

gb_model.fit(X_train, y_train)

evaluation(y_test, gb_model, X_test)

feature_importance(gb_model)

plot_loss(gb_model)
# Let us try a small neural network to perform the required classification

import tensorflow as tf

# Initializing the ANN

nn = tf.keras.models.Sequential()



# Adding the input layer and the first hidden layer

nn.add(tf.keras.layers.Dense(units=10, activation='relu'))



# Adding the second hidden layer

nn.add(tf.keras.layers.Dense(units=6, activation='relu'))



# Adding the output layer

nn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))



# Part 3 - Training the ANN

# Compiling the ANN

nn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
nn.fit(X_train, y_train, batch_size = 128, steps_per_epoch = 64,  epochs = 50)
y_pred = nn.predict(X_test)

for i in range(0, y_pred.size):

    if y_pred[i] > 0.5:

        y_pred[i] = 1

    else:

        y_pred[i] = 0

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)

accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)

print(accuracy)
import seaborn as sns

import matplotlib.pyplot as plt

plt.figure(figsize = (15, 15))

sns.heatmap(cm, annot = True, fmt = '.0f', linewidths = .1, square = True, cmap='viridis')

plt.xlabel('Prediction')

plt.title('Accuracy: {0}'.format(round(accuracy, 2)))

plt.ylabel('Actual')

plt.show()