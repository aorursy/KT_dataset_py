####Loading useful packages

#For data manipulation
import numpy as np
import pandas as pd

#For plotting
import matplotlib.pyplot as pp
import seaborn as sns

#This just ensures that our plots appear
%matplotlib inline                   


#For surpressing warnings
import warnings
warnings.filterwarnings('ignore')
#loading in the data
train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")

#Making's lists on variables which belong to each group
categorical_variables = [train_data.columns[i] for i in range(len(train_data.columns)) if 'cat' in train_data.columns[i]]

binary_variables = [train_data.columns[i] for i in range(len(train_data.columns)) if 'bin' in train_data.columns[i]]

interval_variables = [train_data.columns[i] for i in range(len(train_data.columns)) if (train_data.loc[:,train_data.columns[i]].dtype==float and 'cat' not in train_data.columns[i] and 'bin' not in train_data.columns[i])]

ordinal_variables = [train_data.columns[i] for i in range(len(train_data.columns)) if (train_data.loc[:,train_data.columns[i]].dtype == 'int64' and 'cat' not in train_data.columns[i] and 'bin' not in train_data.columns[i])][2:]


#Let's encode the variables
from sklearn.preprocessing import LabelEncoder

#Create a label encoder object
le = LabelEncoder()
le_count = 0

#Iterate through the columns
for col in train_data:
    if col in categorical_variables:
        #If 2 or fewer unique categories
        if len(list(train_data[col].unique())) <=2:
            #Train on the training data
            le.fit(train_data[col])
            #Transform both training and testing data
            train_data[col] = le.transform(train_data[col])
            test_data[col] = le.transform(test_data[col])
            
            #Keep track of how many columns were label encoded
            le_count +=1

print('%d columns were label encoded.' % le_count)

#One-hot encode variables
train_data = pd.get_dummies(train_data)
test_data = pd.get_dummies(test_data)

print('Training Features shape: ', train_data.shape)
print('Testing Features shape: ', test_data.shape)
from sklearn.preprocessing import MinMaxScaler

#Let's drop 'target' from the training data
train = train_data.drop(['target','id'],axis=1)

#Make a list of feature names
features = list(train.columns)

#Copy of testing data
test = test_data.copy()
test = test.drop('id',axis =1)
#Scale each feature from 0 to 1
scaler = MinMaxScaler(feature_range=(0,1))

#Fit and transform data
scaler.fit(train)
train = scaler.transform(train)
test = scaler.transform(test)
#Let's load in the required package for a logistic regression
from sklearn.linear_model import LogisticRegression

#Make a logistic regression with the specified regularization parameter
log_reg = LogisticRegression(C = 0.5)

#Train on the data
log_reg.fit(train,train_data.loc[:,'target'])
##Making predictions
#Make sure to selct the second column only. This is because the ".predict_proba()" method outputs two columns: the first being the probability of not
#claiming and the second being the probability of making a claim. We're only interested in the probability of making a claim.

log_regs_pred = log_reg.predict_proba(test)[:,1]
##Making  a csv file to submit our predictions

#Submission data frame
submit = test_data[['id']]

#Add in prediction column as 'target'
submit['target'] = log_regs_pred

submit.to_csv('log_reg_pred.csv', index = False)
#Let's load in the required package for a Stochastic Gradient Descent Classifier
from sklearn.linear_model import SGDClassifier

#Make a Stochastic Gradient Descent Classifier
sgdc = SGDClassifier(loss='log')

#Train on the data
sgdc.fit(train, train_data.loc[:,'target'])
##Making predictions
sgdc_preds = sgdc.predict_proba(test)[:,1]
##Making  a csv file to submit our predictions

#Submission data frame
submit2 = test_data[['id']]

#Add in prediction column as 'target'
submit2['target'] = sgdc_preds

submit2.to_csv('sgdc_pred.csv', index = False)
#Let's load in the required package for a Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

#Make a random forest classifier with the specified parameters
random_forest = RandomForestClassifier(n_estimators =100, random_state =78, verbose=1, n_jobs=-1)

#Train on the training data
random_forest.fit(train, train_data.loc[:,'target'])
#Random Forest's also allow us to see the importance of each feature. Let's use that ability
rf_feature_importance_values = random_forest.feature_importances_
rf_feature_importances = pd.DataFrame({'feature':features,'importance':rf_feature_importance_values})

#Make predictions on the test data
rf_preds = random_forest.predict_proba(test)[:,1]
##Making  a csv file to submit our predictions

#Submission data frame
submit3 = test_data[['id']]

#Add in prediction column as 'target'
submit3['target'] = rf_preds

submit3.to_csv('rf_pred.csv', index = False)
#Let's load in the required package for a Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier

#Make a gradient boosting classifier
gbm = GradientBoostingClassifier(n_estimators = 100, random_state=50)

#Train on the training data
gbm.fit(train, train_data.loc[:,'target'])
#Extract features importance
gbm_feature_importance_values = gbm.feature_importances_
gbm_feature_importances = pd.DataFrame({'feature':features,'importance':gbm_feature_importance_values})
#Make predictions on the test data
gbm_preds = gbm.predict_proba(test)[:,1]
##Making  a csv file to submit our predictions
submit4 = test_data[['id']]

submit4['target'] = gbm_preds

submit4.to_csv('gbm_pred.csv', index = False)
rf_feature_importances.set_index('feature').sort_values(by='importance',ascending=False).iloc[:15,:].plot.bar()
pp.title("Feature importance according to Random Forest")
pp.ylabel("Normalized Importance")
pp.xlabel("Feature")
pp.show()

gbm_feature_importances.set_index('feature').sort_values(by='importance',ascending=False).iloc[:15,:].plot.bar()
pp.title("Feature importance according to Gradient Boosting")
pp.ylabel("Normalized Importance")
pp.xlabel("Feature")
pp.show()