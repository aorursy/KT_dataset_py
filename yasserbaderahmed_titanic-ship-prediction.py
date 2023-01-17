import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import scipy

from matplotlib.pyplot import rcParams

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LogisticRegression

from sklearn.neural_network import MLPClassifier

from sklearn import svm



train_df = pd.read_csv('../input/titanic/train.csv')

test_df = pd.read_csv('../input/titanic/test.csv')



train_df.head()
# check datatypes

train_df.info()
#Categorical variables:

categorical = train_df.select_dtypes(include = ["object"]).keys()

print(categorical)
#Quantitative variables:

quantitative = train_df.select_dtypes(include = ["int64","float64"]).keys()

print(quantitative)
rcParams['figure.figsize'] = 9, 9

train_df[quantitative].hist()
plt.rcParams['figure.figsize'] = [10, 5]

train_df['Survived'].value_counts().plot(kind = 'bar',color='g')

plt.title("Survival rate")
women = train_df.loc[train_df.Sex == 'female']["Survived"]

rate_women = sum(women)/len(women)



print("% of women who survived:", rate_women)
men = train_df.loc[train_df.Sex == 'male']["Survived"]

rate_men = sum(men)/len(men)



print("% of men who survived:", rate_men)
plt.pie([rate_women,rate_men],labels=('Womens survival rate','mens survival rate'),

       autopct ='%1.1f%%',shadow = True)
# Random Forests

y = train_df["Survived"]



features = ["Pclass", "Sex", "SibSp", "Parch"]

X = pd.get_dummies(train_df[features])

X_test = pd.get_dummies(test_df[features])



model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=5)

model.fit(X, y)

predictions = model.predict(X_test)



round(model.score(X,y), 4)


LR = LogisticRegression(random_state=0, solver='lbfgs', multi_class='auto')

LR.fit(X, y) # 

LR.predict(X_test) # prediction value

round(LR.score(X,y),4) # accuracy to first 4 number


NN = MLPClassifier(solver='lbfgs', alpha=1e-4, hidden_layer_sizes=(1, 5), random_state=0)

NN.fit(X, y)

NN.predict(X_test) # prediction value

round(NN.score(X,y), 4)


SVM = svm.LinearSVC()

SVM.fit(X, y)

SVM.predict(X_test)

 # prediction value

round(SVM.score(X,y), 4)
plt.pie([round(model.score(X,y), 4),round(LR.score(X,y),4),

        round(NN.score(X,y), 4),round(SVM.score(X,y), 4)],

        explode = [0.2,0.0,0.0,0.0]

        ,labels=('Random Forests',

                 'Logistic Regression','Neural Networks',

                 'Support Vector Machines'),

       autopct ='%1.1f%%',shadow = True)

plt.title('predictions')
output=pd.DataFrame({'PassengerId': test_df.PassengerId, 'Survived': predictions})

print('Survived(1) and unSurvived(0)\n',output)