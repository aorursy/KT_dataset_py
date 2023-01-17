# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")
train.head()
train.describe()
sex = pd.read_csv("../input/gender_submission.csv")
sex.head()
sex.describe()
# add null columns to the sex dataset
sex["Pclass"] = np.nan
sex["Name"] = np.nan
sex["Sex"] = np.nan
sex["Age"] = np.nan
sex["SibSp"] = np.nan
sex["Parch"] = np.nan
sex["Ticket"] = np.nan
sex["Fare"] = np.nan
sex["Cabin"] = np.nan
sex["Embarked"] = np.nan
sex.head()
# append train and sex datasets
train1 = train.append(sex)
train1.head()
train1.describe()
# count missing values
train1.isnull().sum()
train2 = train1.drop(['PassengerId','Cabin','Ticket','Name'], axis=1)
train2.head()
train2.describe()
# impute missing values with one-hot encoding for categorical variables
train_predictors = train2[['Age', 'Sex', 'Fare','Parch','Pclass','SibSp','Embarked']]
one_hot_encoded_training_predictors = pd.get_dummies(train_predictors)
one_hot_encoded_training_predictors.isnull().sum()
#one_hot_encoded_training_predictors
# drop Sex_female and Embarked_C
predictors = one_hot_encoded_training_predictors.drop(['Sex_female','Embarked_C'],axis=1)
predictors.isnull().sum()
# impute missing values for continuous variables
from sklearn.preprocessing import Imputer
my_imputer = Imputer()
predictors1 = my_imputer.fit_transform(predictors)
predictors2 = pd.DataFrame(predictors1, columns=['Age', 'Fare', 'Parch', 'Pclass', 'SibSp', 'Sex_male','Embarked_Q','Embarked_S'])
predictors2.isnull().sum()
# outcome
y = train2.Survived
# predictors 
X = predictors2
# split into training and testing datasets
from sklearn.model_selection import train_test_split
train_X,  test_X, train_y, test_y = train_test_split(X,y,random_state=0)
train_X.head()
train_y.head()
# Data visualisation
import seaborn as sns
sns.pairplot(train_X)
# plot the outcome
sns.countplot(train_y)
df1 = train_X
df2 = pd.DataFrame(train_y)
df2.head()
df3 = df1.merge(df2, left_index=True, right_index=True)
df3.head()
# Multivariate plots
# Age and Fare by Survived - scatterplot
sns.lmplot(x='Fare', y='Age', hue='Survived', data=df3)
# Age and sex by Survived - boxplot
sns.boxplot(x='Sex_male', y='Age', hue='Survived', data=df3)
# Age and Parch by Survived - boxplot
sns.boxplot(x='Parch', y='Age', hue='Survived', data=df3)
# Age and SibSP by Survived - boxplot
sns.boxplot(x='SibSp', y='Age', hue='Survived', data=df3)
# Age and Embarked_Q by Survived - boxplot
sns.boxplot(x='Embarked_Q', y='Age', hue='Survived', data=df3)
# Age and Embarked_S by Survived - boxplot
sns.boxplot(x='Embarked_S', y='Age', hue='Survived', data=df3)
# Age and Pclass by Survived - boxplot
sns.boxplot(x='Pclass', y='Age', hue='Survived', data=df3)
# Heatmap
sns.heatmap(df3.loc[:, ['Age', 'Fare', 'Pclass', 'Parch', 'SibSp','Sex_male','Embarked_Q','Embarked_S','Survived']].corr(), annot = True)

# Fare and Pclass by Survived - boxplot
sns.boxplot(x='Pclass', y='Fare', hue='Survived', data=df3)
# Parch and SibSp by Survived - scatterplot
sns.lmplot(x='SibSp', y='Parch', hue='Survived', data=df3)
# create interactions
train_X["Pclass_Fare"] = train_X.Pclass * train_X.Fare
train_X["SibSp_Parch"] = train_X.SibSp * train_X.Parch
train_X["Age_Fare"] = train_X.Age * train_X.Fare
test_X["Pclass_Fare"] = test_X.Pclass * test_X.Fare
test_X["SibSp_Parch"] = test_X.SibSp * test_X.Parch
test_X["Age_Fare"] = test_X.Age * test_X.Fare
train_X.head()
# Partial dependence plots - GB classifier
#from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence
#from sklearn.ensemble import GradientBoostingClassifier
#classifier = GradientBoostingClassifier()
#train_X_colns = ['Age','Fare','Parch','Pclass','SbSp','Sex_male','Embarked_Q','Embarked_S','Pclass_Fare','SibSp_Parch','Age_Fare']
#classifier.fit(train_X, train_y)
#titanic_plots = plot_partial_dependence(classifier, features=[0,1,2,3,4,5,6,7,8,9,10], X=train_X, 
#                                        feature_names=train_X_colns, grid_resolution=8)
# feature importance
#names = train_X
#print("Features sorted by their score:")
#print(sorted(zip(map(lambda x: round(x, 4), classifier.feature_importances_), names), 
#             reverse=True))
# keep the most important predictors
#important_predictors = ['Age','Fare','Pclass','Sex_male','SibSp']
#important_predictors = ['Age','Fare','Pclass','Sex_male','Pclass_Fare','Age_Fare']
#train_X_new = train_X[important_predictors]
#train_X_new.head()
# same for the test dataset
#test_X_new = test_X[important_predictors]
#test_X_new.head()
# # Feature Scaling
# standardize the data
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#train_X_new = sc.fit_transform(train_X_new)
#test_X_new = sc.transform(test_X_new)
# Normalize the data
from sklearn.preprocessing import Normalizer
sc = Normalizer()
train_X = sc.fit_transform(train_X)
test_X = sc.transform(test_X)
# fit a model - GBoost
from sklearn.ensemble import GradientBoostingClassifier
classifier_gb = GradientBoostingClassifier(n_estimators=300,max_depth=5)
classifier_gb.fit(train_X,train_y)
# fit a model - XGBoost
from xgboost import XGBClassifier
classifier_xgb = XGBClassifier(n_estimators=300,max_depth=5)
classifier_xgb.fit(train_X,train_y)
# fit a Random Forest
from sklearn.ensemble import RandomForestClassifier
classifier_rf = RandomForestClassifier(n_estimators=300,max_depth=5)
classifier_rf.fit(train_X,train_y)
# fit a model - Logistic regression
from sklearn.linear_model import LogisticRegression
classifier_lr = LogisticRegression()
classifier_lr.fit(train_X,train_y)
# fit a Neural Network and SVM
# # Feature Scaling
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#train_X_new = sc.fit_transform(train_X_new)
#test_X_new = sc.transform(test_X_new)

# Importing the Keras libraries and packages
#import keras
#from keras.models import Model
#from keras.layers import Input, Dense, Dropout
#build the model

#inp = Input(shape=(6,))
#hidden_1 = Dense(12, activation='relu')(inp)
#dropout_1 = Dropout(0.2)(hidden_1)
#hidden_2 = Dense(12, activation='relu')(dropout_1)
#dropout_2 = Dropout(0.2)(hidden_2)
#hidden_3 = Dense(12, activation='relu')(dropout_2)
#dropout_3 = Dropout(0.2)(hidden_3)
#out = Dense(1, activation='sigmoid')(dropout_3)

#classifier_dnn = Model(inputs=inp, outputs=out)

# Compile the model
#classifier_dnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fit the model to the training set
#classifier_dnn.fit(train_X, train_y, epochs = 100, batch_size = 32, verbose=1, validation_split=0.1)
# Evaluate the model
#model.evaluate(test_X_new, test_y, verbose=1)
# Import keras
#import keras
#from keras.models import Sequential
#from keras.layers import Dense

# fit the model
# Initialising the ANN
#classifier = Sequential()

# Adding the input layer and the first hidden layer
#classifier.add(Dense(output_dim = 9, init = 'uniform', activation = 'relu', input_dim = 6))

# Adding the second hidden layer
#classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# Adding the third hidden layer
#classifier.add(Dense(output_dim = 2, init = 'uniform', activation = 'relu'))

# Adding the output layer
#classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# model output shape
#classifier.output_shape
# model summary
#classifier.summary()
# model configuration
#classifier.get_config()
# model weights
#classifier.get_weights()
# Compiling the ANN
#classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# Fitting the ANN to the Training set, holding out 10% of data for validation
#classifier.fit(train_X_new, train_y, batch_size = 32, epochs = 31, verbose = 1, validation_split = 0.1)
# Evaluate the trained model on the test dataset
#classifier.evaluate(test_X_new, test_y, verbose=1)
# Model tuning
#from keras.optimizers import adam
#opt = adam(lr=0.5, decay=1e-6)
#classifier.compile(loss = 'binary_crossentropy', optimizer = opt ,metrics = ['accuracy'])
# Fitting the ANN to the Training set, holding out 10% of data for validation
#classifier.fit(train_X_new, train_y, batch_size = 32, epochs = 100, verbose = 1, validation_split = 0.1)
# Evaluate the trained model on the test dataset
#classifier.evaluate(test_X_new, test_y, verbose=1)
# Fit an SVM model
from sklearn.svm import SVC
classifier_svm = SVC(kernel = 'rbf', random_state = 0)
classifier_svm.fit(train_X, train_y)

# make predictions - GBoost
preds = classifier_gb.predict(test_X)
from sklearn.metrics import accuracy_score
print("Accuracy : " + str(accuracy_score(preds, test_y)))
# make predictions - XGBoost
preds = classifier_xgb.predict(test_X)
from sklearn.metrics import accuracy_score
print("Accuracy : " + str(accuracy_score(preds, test_y)))
# make predictions - Random Forest
preds = classifier_rf.predict(test_X)
from sklearn.metrics import accuracy_score
print("Accuracy : " + str(accuracy_score(preds, test_y)))
# make predictions - Logistic regression
preds = classifier_lr.predict(test_X)
from sklearn.metrics import accuracy_score
print("Accuracy : " + str(accuracy_score(preds, test_y)))
# Predicting the Test set results -  Neural network
#y_pred = classifier.predict(test_X_new, batch_size=32)
#y_pred = (y_pred > 0.5)
#y_pred = classifier.predict_classes(test_X_new, batch_size=32)
# Making the Confusion Matrix
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(test_y, y_pred)
#cm
# make predictions - SVM
preds = classifier_svm.predict(test_X)
from sklearn.metrics import accuracy_score
print("Accuracy : " + str(accuracy_score(preds, test_y)))
# Cross-validation - GBoost
#classifier_gb = GradientBoostingClassifier()
#X_new = predictors2[important_predictors]
#X_new = train_X_new
from sklearn.model_selection import cross_val_score
scores_gb = cross_val_score(classifier_gb, train_X, train_y, cv=10)
#scores 
# Cross-validation - XGBoost
#classifier = XGBClassifier()
#X_new = predictors2[important_predictors]
#from sklearn.model_selection import cross_val_score
scores_xgb = cross_val_score(classifier_xgb, train_X, train_y, cv=10)
#scores 
# Cross-validation - Random Forest
#classifier = RandomForestClassifier()
#X_new = predictors2[important_predictors]
#from sklearn.model_selection import cross_val_score
scores_rf = cross_val_score(classifier_rf, train_X, train_y, cv=10)
#scores 
# Cross-validation - Logistic regression
#classifier = LogisticRegression()
#X_new = predictors2[important_predictors]
#from sklearn.model_selection import cross_val_score
scores_lr = cross_val_score(classifier_lr, train_X, train_y, cv=10)
#scores 
# Cross-validation - SVM
#classifier = classifier = SVC(kernel = 'rbf', random_state = 0)
#X_new = predictors2[important_predictors]
#from sklearn.model_selection import cross_val_score
scores_svm = cross_val_score(classifier_svm, train_X, train_y, cv=10)
#scores 
# mean accuracy and 95% CIs - GBoost
print("Accuracy: %0.2f (+/- %0.2f)" % (scores_gb.mean(), scores_gb.std() * 2))
# mean accuracy and 95% CIs - XGBoost
print("Accuracy: %0.2f (+/- %0.2f)" % (scores_xgb.mean(), scores_xgb.std() * 2))
# mean accuracy and 95% CIs - Random Forest
print("Accuracy: %0.2f (+/- %0.2f)" % (scores_rf.mean(), scores_rf.std() * 2))
# mean accuracy and 95% CIs - Logistic regression
print("Accuracy: %0.2f (+/- %0.2f)" % (scores_lr.mean(), scores_lr.std() * 2))
# mean accuracy and 95% CIs - SVM
print("Accuracy: %0.2f (+/- %0.2f)" % (scores_svm.mean(), scores_svm.std() * 2))
# Model tuning - GBoost
# fit a model
#classifier = GradientBoostingClassifier(n_estimators=1000)
#classifier.fit(train_X_new,train_y)
# make predictions
#preds = classifier.predict(test_X_new)
#from sklearn.metrics import accuracy_score
#print("Accuracy : " + str(accuracy_score(preds, test_y)))
# Cross-validation - GBoost
#classifier = GradientBoostingClassifier(n_estimators=1000)
#X_new = predictors2[important_predictors]
#from sklearn.model_selection import cross_val_score
#scores = cross_val_score(classifier, train_X_new, train_y, cv=10)
#scores 
# mean accuracy and 95% CIs - GBoost
#print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
# Model tuning - XGBoost
# fit a model
#classifier = XGBClassifier(n_estimators=1000, learning_rate=0.5)
#classifier.fit(train_X_new,train_y)
# make predictions
#preds = classifier.predict(test_X_new)
#from sklearn.metrics import accuracy_score
#print("Accuracy : " + str(accuracy_score(preds, test_y)))
# Cross-validation - XGBoost
#classifier = XGBClassifier(n_estimators=1000, learning_rate=0.5)
#X_new = predictors2[important_predictors]
#from sklearn.model_selection import cross_val_score
#scores = cross_val_score(classifier, train_X_new, train_y, cv=10)
#scores 
# mean accuracy and 95% CIs - XGBoost
#print("ccuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
# Model tuning - Random Forest
# fit a model
#classifier = RandomForestClassifier(n_estimators=100)
#classifier.fit(train_X_new,train_y)
# make predictions
#preds = classifier.predict(test_X_new)
#from sklearn.metrics import accuracy_score
#print("Accuracy : " + str(accuracy_score(preds, test_y)))
# Cross-validation - Random Forest
#classifier = RandomForestClassifier(n_estimators=100)
#X_new = predictors2[important_predictors]
#from sklearn.model_selection import cross_val_score
#scores = cross_val_score(classifier, train_X_new, train_y, cv=10)
#scores 
# mean accuracy and 95% CIs - Ranfom Forest
#print("ccuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
# Model tuning - Neural network
# fit a model
#classifier = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
#classifier.fit(train_X_new,train_y)
# make predictions
#preds = classifier.predict(test_X_new)
#from sklearn.metrics import accuracy_score
#print("Accuracy : " + str(accuracy_score(preds, test_y)))
# Cross-validation - Neural network
#classifier = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
#X_new = predictors2[important_predictors]
#from sklearn.model_selection import cross_val_score
#scores = cross_val_score(classifier, train_X_new, train_y, cv=10)
#scores 
# mean accuracy and 95% CIs - Neural network
#print("ccuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
# Import the test dataset
test = pd.read_csv("../input/test.csv")
test.head()
# drop unwanted variables
test2 = test.drop(['PassengerId','Cabin','Ticket','Name'], axis=1)
test2.head()
# impute missing values with one-hot encoding for categorical variables
test_predictors = test2[['Age', 'Sex', 'Fare','Parch','Pclass','SibSp','Embarked']]
one_hot_encoded_test_predictors = pd.get_dummies(test_predictors)
one_hot_encoded_test_predictors.isnull().sum()
#one_hot_encoded_training_predictors
# drop Sex_female and Embarked_C
predictors_test = one_hot_encoded_test_predictors.drop(['Sex_female','Embarked_C'],axis=1)
predictors_test.isnull().sum()
# impute missing values for Age and Fare
from sklearn.preprocessing import Imputer
my_imputer = Imputer()
predictors1 = my_imputer.fit_transform(predictors_test)
predictors2 = pd.DataFrame(predictors1, columns=['Age', 'Fare', 'Parch', 'Pclass', 'SibSp', 'Sex_male','Embarked_Q','Embarked_S'])
predictors2.isnull().sum()
# create interactions
predictors2["Pclass_Fare"] = predictors2.Pclass * predictors2.Fare
predictors2["SibSp_Parch"] = predictors2.SibSp * predictors2.Parch
predictors2["Age_Fare"] = predictors2.Age * predictors2.Fare

predictors2.head()
# keep the most important predictors
#important_predictors = ['Age','Fare','Pclass','Sex_male','SibSp']
#important_predictors = ['Age','Fare','Pclass','Sex_male','Pclass_Fare','Age_Fare']
#test_X_new = predictors2[important_predictors]
#test_X_new.head()
# Normalize the data
from sklearn.preprocessing import Normalizer
sc = Normalizer()
test3 = sc.fit_transform(predictors2)
# fit a model - GBoost
#classifier = GradientBoostingClassifier(n_estimators=100,learning_rate=0.5)
#classifier.fit(train_X_new,train_y)
# make predictions to the test dataset
#predictions = classifier.predict(test_X_new)
#predictions
# fit a model - XGBoost
#classifier = XGBClassifier(n_estimators=100)
#classifier.fit(train_X_new,train_y)
# make predictions to the test dataset
#predictions = classifier_xgb.predict(test3)
#predictions
# fit a model - Random Forest
#classifier = RandomForestClassifier(n_estimators=100)
#classifier.fit(train_X_new,train_y)
# make predictions to the test dataset
predictions = classifier_rf.predict(test3)
#predictions
# fit a model - Logistic regression
#classifier = LogisticRegression()
#classifier.fit(train_X_new,train_y)
# make predictions to the test dataset
#predictions = classifier.predict(test_X_new)
#predictions
# fit a model - Neural network
#test_X_new = sc.fit_transform(test_X_new)
#test_X_new
# Predicting the Test set results
#y_pred = model.predict(test_X_new,batch_size=32)
#y_pred = np.array(y_pred > 0.5)
#y_pred
#predictions = y_pred * 1
#predictions = y_pred.flatten() * 1
#predictions
# fit a model - SVM
#test_X_new = sc.fit_transform(test_X_new)
#classifier = classifier = SVC(kernel = 'rbf', random_state = 0)
#classifier.fit(train_X_new,train_y)
# make predictions to the test dataset
#predictions = classifier.predict(test_X_new)
#predictions
# Prepare submission file - GBoost
#xgb3 = pd.DataFrame({'PassengerId':test.PassengerId,'Survived':predictions})
# prepare the csv file
#xgb3.to_csv('xgb3.csv',index=False)
# Prepare submission file - XGBoost
#xgb6 = pd.DataFrame({'PassengerId':test.PassengerId,'Survived':predictions})
# prepare the csv file
#xgb6.to_csv('xgb6.csv',index=False)
# Prepare submission file - Random Forest
rf2 = pd.DataFrame({'PassengerId':test.PassengerId,'Survived':predictions})
# prepare the csv file
rf2.to_csv('rf2.csv',index=False)
# Prepare submission file - Logistic Regression
#lr1 = pd.DataFrame({'PassengerId':test.PassengerId,'Survived':predictions})
# prepare the csv file
#lr1.to_csv('lr1.csv',index=False)
# Prepare submission file - SVM
#svm2 = pd.DataFrame({'PassengerId':test.PassengerId,'Survived':predictions})
# prepare the csv file
#svm2.to_csv('svm2.csv',index=False)
# Prepare submission file - Neural network
#nn4 = pd.DataFrame({'PassengerId':test.PassengerId,'Survived':predictions})
# prepare the csv file
#nn4.to_csv('nn4.csv',index=False)