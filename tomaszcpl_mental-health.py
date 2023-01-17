import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
data = pd.read_csv("../input/survey.csv")
data.head()
data.shape
list(data.columns.values)
print(data.isnull().sum())
del data['comments']
del data['state']
del data['Timestamp']
del data['work_interfere']
data.reset_index()
def age_process(age):
    if age>=0 and age<=120:
        return age
    else:
        return np.nan
data['Age'] = data['Age'].apply(age_process)
label_encoder = LabelEncoder()
X = data.drop(['treatment'],axis=1)
y = data['treatment']
X = pd.get_dummies(data)
y = label_encoder.fit_transform(y)
X = pd.DataFrame(X).fillna(value=0)
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.20, random_state=1)
def random_forest_classifier(features, target):
    clf = RandomForestClassifier()
    clf.fit(features, target)
    return clf
trained_model = random_forest_classifier(train_x, train_y)
predictions = trained_model.predict(test_x)

print ("Train Accuracy :: ", accuracy_score(train_y, trained_model.predict(train_x)))
print ("Test Accuracy  :: ", accuracy_score(test_y, predictions))
print ("Confusion matrix ", confusion_matrix(test_y, predictions))