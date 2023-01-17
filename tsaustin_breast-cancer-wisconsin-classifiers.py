import pandas as pd

# Load the data set

df = pd.read_csv('../input/data.csv')
df.head()
# drop ID - we don't need that

df.drop('id',axis=1,inplace=True)

# there's an additional column 'unnamed' with no data. delete as well

df.drop('Unnamed: 32',axis=1,inplace=True)
df.describe().T
# check for missing data

all_data_na = (df.isnull().sum() / len(df)) * 100

all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]

missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})

missing_data.head(20)
from scipy.stats import skew 



# find skewed features

numeric_feats = df.dtypes[df.dtypes != "object"].index



# Check the skew of all numerical features

skewed_feats = df[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

skewness = pd.DataFrame({'Skew' :skewed_feats})

skewness.head(10)
# "unskew" the features

skewness = skewness[abs(skewness) > 0.75]



from scipy.special import boxcox1p

skewed_features = skewness.index

lam = 0.15

for feat in skewed_features:

    df[feat] = boxcox1p(df[feat], lam)
# Replace the target values (M = malignant, B = benign) with 1 for malignant and 0 for begnin tumors

df['diagnosis']= df['diagnosis'].map({'M':1,'B':0})
# split into target (y) and features (X)

y = df['diagnosis']

X = df.drop('diagnosis',axis=1)
# Scale the feature values

from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(X)
# and split into train and test data sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
import numpy as np



# scoring function for the model

# we score with the mean AUC from 5 folds of the training data

from sklearn.model_selection import KFold, cross_val_score

def score_model(model):

    kf = KFold(5, shuffle=True, random_state=42).get_n_splits(X_train)

    model_score = np.mean(cross_val_score(model, X_train, y_train, scoring="recall", cv = kf))

    return((type(model).__name__,model_score))
from sklearn.neighbors import KNeighborsClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from xgboost import XGBClassifier



model_scores = pd.DataFrame(columns=['model','Recall score'])

clfs = [KNeighborsClassifier(),  MLPClassifier(), SVC(), GaussianProcessClassifier(), DecisionTreeClassifier()

        , RandomForestClassifier(), AdaBoostClassifier(), GaussianNB(), QuadraticDiscriminantAnalysis(), XGBClassifier()]

for clf in clfs:

    sc = score_model(clf)

    model_scores = model_scores.append({'model':sc[0],'Recall score':sc[1]},ignore_index=True)    
model_scores.sort_values('Recall score',ascending=False)
# SVC

from sklearn.model_selection import GridSearchCV



parameter_space = {

    'C': np.logspace(-2, 10, 20,base=2),

    'gamma': np.logspace(-9, 3, 13),

}

grid_search = GridSearchCV(SVC(kernel='rbf'), parameter_space, n_jobs=-1, cv=5,scoring="recall")

grid_search.fit(X_train, y_train)

grid_search.best_params_
# MLP

parameter_space = {

    'hidden_layer_sizes': [(20,20), (100,), (50,), (30,)],

    'activation': ['tanh', 'relu'],

    'solver': ['sgd', 'adam'],

    'alpha': [0.0001, 0.05],

    'learning_rate': ['constant','adaptive'],

}

grid_search = GridSearchCV(MLPClassifier(), parameter_space, n_jobs=-1, cv=5,scoring="recall")

grid_search.fit(X_train, y_train)

grid_search.best_params_
model_scores = pd.DataFrame(columns=['model','Recall score'])

clfs = [SVC(kernel="rbf", gamma=0.001, C=47),MLPClassifier(max_iter=300,activation='relu',

                                                         hidden_layer_sizes=(30,),alpha=0.0001, learning_rate="adaptive",

                                                        solver="adam")]

for clf in clfs:

    sc = score_model(clf)

    model_scores = model_scores.append({'model':sc[0],'Recall score':sc[1]},ignore_index=True) 
model_scores
# train and then validate with the test data set

from sklearn.metrics import recall_score



clf = SVC(kernel="rbf", gamma=0.001, C=47)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Test Recall: {}".format(recall_score(y_test, y_pred)))

# calculate the confusion matrix

from sklearn.metrics import confusion_matrix

cnf_matrix = confusion_matrix(y_test,y_pred)
# Confusion matrix with Seaborn

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



class_names = [0,1]

fig,ax = plt.subplots()

tick_marks = np.arange(len(class_names))

plt.xticks(tick_marks,class_names)

plt.yticks(tick_marks,class_names)



#create a heat map

sns.heatmap(pd.DataFrame(cnf_matrix), annot = True, cmap = 'YlGnBu',

           fmt = 'g')

ax.xaxis.set_label_position('top')

plt.tight_layout()

plt.title('Confusion matrix for SVC classifier', y = 1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')

plt.show()
from sklearn.metrics import precision_score

from sklearn.metrics import accuracy_score

print("Precision: {}".format(precision_score(y_test,y_pred)))

print("Accuracy: {}".format(accuracy_score(y_test,y_pred)))