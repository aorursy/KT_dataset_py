import pandas as pd

from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier

from sklearn import preprocessing



from sklearn.metrics import f1_score

from sklearn.metrics import accuracy_score
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

test_data = pd.read_csv('/kaggle/input/titanic/test.csv')

all_data = [train_data, test_data]
# Fill in missing values



for datadf in all_data:    

    #complete missing age with median

    datadf['Age'].fillna(datadf['Age'].median(), inplace = True)



    #complete embarked with mode

    datadf['Embarked'].fillna(datadf['Embarked'].mode()[0], inplace = True)



    #complete missing fare with median

    datadf['Fare'].fillna(datadf['Fare'].median(), inplace = True)

    

# Getting rid of irrelevant columns: Passanger ID, Cabin number, and Ticker number

drop_column = ['PassengerId','Cabin', 'Ticket']

for datadf in all_data:    

    datadf.drop(drop_column, axis=1, inplace = True)
# Replace Categorical variables with numbers

train_data['Sex'].replace(to_replace=['male','female'], value=[0,1],inplace=True)

train_data['Embarked'].replace(to_replace=['C','Q', 'S'], value=[0,1,2],inplace=True)



test_data['Sex'].replace(to_replace=['male','female'], value=[0,1],inplace=True)

test_data['Embarked'].replace(to_replace=['C','Q', 'S'], value=[0,1,2],inplace=True)
X = train_data

y = train_data['Survived'].values

X.drop(['Name'],axis = 1, inplace = True)

X.drop(['Survived'],axis = 1, inplace = True)



X_submit = test_data

X_submit.drop(['Name'],axis = 1, inplace = True)



X = preprocessing.StandardScaler().fit(X).transform(X)

X_submit = preprocessing.StandardScaler().fit(X_submit).transform(X_submit)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
model = XGBClassifier(n_estimators=250, learning_rate=0.02).fit(X_train,y_train)



# Make predictions

predictions = model.predict(X_test)

    

# Evaluate the model

print("XGB accuracy: %.2f" % accuracy_score(y_test, predictions))

print("XGB F1-score: %.2f" % f1_score(y_test, predictions, average='weighted') )    
prediction_submit = model.predict(X_submit)



# Save the predictions as a csv file

testdata = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")

output = pd.DataFrame({'PassengerId': testdata.PassengerId, 'Survived': prediction_submit})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")