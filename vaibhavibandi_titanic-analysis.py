import pandas as pd

import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
pwd
variable=pd.read_csv("../input/titanic/train.csv")

variable.head()
y=variable.pop("Survived")
y.head()
numeric_var=list(variable.dtypes[variable.dtypes !="object"].index)

variable[numeric_var].head()
variable["Age"].fillna(variable.Age.mean(),inplace=True)

variable.describe()
variable.tail()
variable[numeric_var].head()
model=RandomForestClassifier(n_estimators=100)
model.fit(variable[numeric_var],y)
test=pd.read_csv("../input/titanic/test.csv")

test[numeric_var].head()
test["Age"].fillna(test.Age.mean(),inplace=True)
test=test[numeric_var].fillna(test.mean()).copy()

y_pred= model.predict(test[numeric_var])

y_pred
submission=pd.DataFrame({

    "PassengerId":test ["PassengerId"],

    "Survived":y_pred

})

submission.to_csv("titanic analysis.csv",index=False)
submission.head()