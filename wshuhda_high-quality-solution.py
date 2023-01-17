import numpy as np
import pandas as pd
from sklearn import tree
gender_map = {'female': 0, 'male': 1}
fields = ['Pclass', 'Sex', 'Age', 'Fare']
train['Sex'] = train['Sex'].map(gender_map).astype(int)
train['Age'] = train['Age'].fillna(train['Age'].median())

test['Sex'] = test['Sex'].map(gender_map).astype(int)
test['Age'] = test['Age'].fillna(test['Age'].median())
test['Fare'] = test['Fare'].fillna(test['Fare'].median())
target = train['Survived'].values
features = train[fields].values
test_features = test[fields].values
dec_tree = tree.DecisionTreeClassifier(min_samples_split = 4)
dec_tree = dec_tree.fit(features, target)
prediction = dec_tree.predict(test_features)

pindex = test['PassengerId'].values
solution = pd.DataFrame(prediction, pindex, columns = ['Survived'])

solution.to_csv('solution.csv', index_label = 'PassengerId') 