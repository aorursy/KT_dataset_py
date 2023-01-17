from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import roc_auc_score

import pandas as pd
# import the data from csv files

df = pd.read_csv("../input/train.csv")

y = df.pop("Survived")
# Preview train dataframe

df.describe()
df["Age"].fillna(df.Age.mean(), inplace=True)
df.describe()
numeric_variables = list(df.dtypes[df.dtypes != 'object'].index)

df[numeric_variables]
model = RandomForestRegressor(n_estimators=100, oob_score=True, random_state=42)

model.fit(df[numeric_variables], y)
model.oob_score_
y_oob = model.oob_prediction_

roc_auc_score(y, y_oob)
def describe_categorical(df):

    from IPython.display import display, HTML

    display(HTML(df[df.columns[df.dtypes == 'object']].describe().to_html()))
describe_categorical(df)
df.drop(['Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)
def clean_cabin_column(column):

    try:

        return column[0]

    except TypeError:

        return "NA"
df["Cabin"] = df.Cabin.apply(clean_cabin_column)
categorical_variables = ['Sex', 'Cabin', 'Embarked']



for variable in categorical_variables:

    df[variable].fillna("Missing", inplace=True)

    temp = pd.get_dummies(df[variable], prefix=variable)

    df = pd.concat([df, temp], axis=1)

    df.drop([variable], axis=1, inplace=True)
def print_all(df, max_rows=10):

    from IPython.display import display, HTML

    display(HTML(df.to_html(max_rows=max_rows)))
print_all(df)
model = RandomForestRegressor(100, oob_score=True,n_jobs=-1, random_state=42)

model.fit(df, y)
roc_auc_score(y, model.oob_prediction_)