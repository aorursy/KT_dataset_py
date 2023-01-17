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
# Dateipfad in einer Variable speichern

video_games_path = '../input/Video_Games_Sales_as_at_22_Dec_2016.csv'

# Daten auslesen

vgd = pd.read_csv(video_games_path)

# Zusammenfassung der Daten ausgeben

vgd.describe()
vgd.columns


def missing_values_table(df):

        # Anzahl der fehlenden Daten

        mis_val = df.isnull().sum()

        

        # Anzahl der fehlenden Daten in %

        mis_val_percent = 100 * df.isnull().sum() / len(df)

        

        # Erstellen einer Tabelle mit den Ergebnissen

        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

        

        # Benennung der Spalten

        mis_val_table_ren_columns = mis_val_table.rename(

        columns = {0 : "Missing Values", 1 : "% of Total Values"})

        

        # Absteigendes Sortieren der Zeilen

        mis_val_table_ren_columns = mis_val_table_ren_columns[

            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(

        "% of Total Values", ascending=False).round(1)

        

        

        

        return mis_val_table_ren_columns
#Aufrufen der Tabelle

missing_values_table(vgd)
vgd_clean = vgd.dropna(axis=0)
missing_values_table(vgd_clean)
vgd_clean.describe()
#Prediction Target

y = vgd_clean.Global_Sales



# neue Features werden für das Modell genommen

vgd_features = ['User_Score', 'User_Count', 'Critic_Score', 'Critic_Count']



X = vgd_clean[vgd_features]
y.describe()
X.describe(include="all")
y.head()
X.head()
from sklearn.tree import DecisionTreeRegressor



vgd_model = DecisionTreeRegressor(random_state=1)



vgd_model.fit(X, y)
vgd_model = DecisionTreeRegressor(random_state=1)



vgd_model.fit(X, y)
X.head()
y.head()
print("Die ersten 5 Spiele:")

print(vgd_model.predict(X.head()))

print("Die letzen 5 Spiele:")

print(vgd_model.predict(X.tail()))
print("Die ersten 5 Spiele:")

print(y.head())

print("Die letzen 5 Spiele:")

print(y.tail())
from sklearn.metrics import mean_absolute_error



prediction = vgd_model.predict(X)

mean_absolute_error(y, prediction)
from sklearn.model_selection import train_test_split



train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)



vgd_model = DecisionTreeRegressor()



vgd_model.fit(train_X, train_y)



val_prediction = vgd_model.predict(val_X)
train_X
val_X
print(mean_absolute_error(val_y, val_prediction))
mean_absolute_error(y, prediction)
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):

    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)

    model.fit(train_X, train_y)

    preds_val = model.predict(val_X)

    mae = mean_absolute_error(val_y, preds_val)

    return(mae)
for max_leaf_nodes in [5, 10, 50, 100, 500]:

    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)

    print("Max leaf Nodes: ", max_leaf_nodes, "\t\t", "MAE: ", my_mae)
vgd_model = DecisionTreeRegressor(random_state=1, max_leaf_nodes=50)



vgd_model.fit(train_X, train_y)

print(vgd_model.predict(val_X.head()))
print(val_y.head())
from sklearn.ensemble import RandomForestRegressor



forest_model = RandomForestRegressor(random_state=1)

forest_model.fit(train_X, train_y)

vgd_preds = forest_model.predict(val_X)

print(mean_absolute_error(val_y, vgd_preds))
from sklearn.linear_model import LinearRegression

clf = LinearRegression()
# Modell erstellen

clf.fit(train_X, train_y)
# Vorhersage der ersten und letzen 5 Daten

print(clf.predict(val_X.head()))

print(clf.predict(val_X.tail()))
# Vergleich mit den ursprünglichen Daten

print(val_y.head())

print(val_y.tail())
clf.score(val_X, val_y)
# Get list of categorical variables

s = (train_X.dtypes == 'object')

object_cols = list(s[s].index)



print("Categorical variables:")

print(object_cols)
# Function for comparing different approaches

def score_dataset(train_X, val_X, train_y, val_y):

    model = RandomForestRegressor(n_estimators=50, random_state=0)

    model.fit(train_X, train_y)

    preds = model.predict(val_X)

    return mean_absolute_error(val_y, preds)
drop_train_X = train_X.select_dtypes(exclude=['object'])

drop_val_X = val_X.select_dtypes(exclude=['object'])



print("MAE from Approach 1 (Drop categorical variables):")

print(score_dataset(drop_train_X, drop_val_X, train_y, val_y))
from sklearn.preprocessing import LabelEncoder



# Make copy to avoid changing original data 

label_X_train = train_X.copy()

label_X_valid = val_X.copy()



# Apply label encoder to each column with categorical data

label_encoder = LabelEncoder()

for col in object_cols:

    label_X_train[col] = label_encoder.fit_transform(train_X[col])

    label_X_valid[col] = label_encoder.transform(val_X[col])



print("MAE from Approach 2 (Label Encoding):") 

print(score_dataset(label_X_train, label_X_valid, train_y, val_y))
import matplotlib.pyplot as plt

%matplotlib inline
import seaborn as sns

plt.style.use("fivethirtyeight")



# Änderung der Schrift

plt.rcParams["font.size"] = 24

plt.rcParams["figure.facecolor"] = "white"

plt.rcParams["axes.facecolor"] = "white"



# Internal ipython tool for setting figure size

from IPython.core.pylabtools import figsize

figsize(15, 12)
plt.scatter(vgd_clean['User_Count'], vgd_clean['Global_Sales'])
plt.scatter(vgd_clean['User_Score'], vgd_clean['Global_Sales'])
plt.scatter(vgd_clean['Critic_Score'], vgd_clean['Global_Sales'])
plt.scatter(vgd_clean['Critic_Count'], vgd_clean['Global_Sales'])
def rm_outliers(df, list_of_keys):

    df_out = df

    for key in list_of_keys:

        # Calculate first and third quartile

        first_quartile = df_out[key].describe()["25%"]

        third_quartile = df_out[key].describe()["75%"]



        # Interquartile range

        iqr = third_quartile - first_quartile



        # Remove outliers

        removed = df_out[(df_out[key] <= (first_quartile - 3 * iqr)) |

                    (df_out[key] >= (third_quartile + 3 * iqr))] 

        df_out = df_out[(df_out[key] > (first_quartile - 3 * iqr)) &

                    (df_out[key] < (third_quartile + 3 * iqr))]

    return df_out, removed
vgd_clean, rmvd_global = rm_outliers(vgd_clean, ["Global_Sales"])

vgd_clean.describe()
vgd_clean, rmvd_global = rm_outliers(vgd_clean, ["User_Count"])

vgd_clean.describe()
vgd_clean, rmvd_global = rm_outliers(vgd_clean, ["Critic_Score"])

vgd_clean.describe()
vgd_clean, rmvd_global = rm_outliers(vgd_clean, ["Critic_Count"])

vgd_clean.describe()
y = vgd_clean.Global_Sales



vgd_features = ['User_Score', 'Critic_Score', 'User_Count','User_Score']



X = vgd_clean[vgd_features]
from sklearn.model_selection import train_test_split



train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)



vgd_model = DecisionTreeRegressor()



vgd_model.fit(train_X, train_y)



val_prediction = vgd_model.predict(val_X)

print(mean_absolute_error(val_y, val_prediction))
from sklearn.linear_model import LinearRegression

clf = LinearRegression()
clf.fit(train_X, train_y)
print(clf.predict(val_X.head()))

print(clf.predict(val_X.tail()))
print(val_y.head())

print(val_y.tail())
clf.score(val_X, val_y)