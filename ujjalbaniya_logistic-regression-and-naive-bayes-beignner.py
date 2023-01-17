#Loading csv file 

import pandas as pd

dataset = pd.read_csv("../input/mushroom-classification/mushrooms.csv")
dataset.head(5)
#Y as Target variable and remaining as Input variables

Y = dataset["class"]

X = dataset.iloc[:,1:]
#Converting character of target to integer

from sklearn.preprocessing import LabelEncoder

encoder_y = LabelEncoder()

Y = encoder_y.fit_transform(Y)

Y
#Checking for same value through out the column and deleting such column 

for col in X.columns:

    if X[col].nunique()==1:

        print(col)

        X.drop(col,axis=1,inplace=True)
#Converting Character of input to Integer

encoder_x = LabelEncoder()

for col in X.columns:

    X[col]=encoder_x.fit_transform(X[col])

X
#spliting data to training and test

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

print(f'X_TRAIN,X_TEST = {len(X_train)},{len(X_test)}')

print(f'Y_TRAIN,Y_TEST = {len(Y_train)},{len(Y_test)}')
#Feature scaling for logistic regression

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.transform(X_test)
#Importing GaussianNB, fitting dataset and Predicitng

from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit(X_train,Y_train)

Y_pred = classifier.predict(X_test)
#Using confusion matrix for Naive Bayes

from sklearn.metrics import confusion_matrix

conM = confusion_matrix(Y_test,Y_pred)

conM
#Importing Logistic Regression Fitting and Predicting

from sklearn.linear_model import LogisticRegression

reg = LogisticRegression()

reg.fit(X_train,Y_train)

y_pred = reg.predict(X_test)

y_pred
#Using confusion matrix for Logistic Regression

from sklearn.metrics import confusion_matrix

con = confusion_matrix(Y_test,y_pred)

con
print(f"Confusion Matrix Naive BAYES:  {conM}")

print(f"Confusion Matrix Logistic Reg:  {con}")