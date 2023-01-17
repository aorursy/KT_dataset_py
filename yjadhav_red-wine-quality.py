import os
import pandas as pd

INPUT_FILE = "winequality-red.csv"

def load_wine_data(file=INPUT_FILE, header=True):
    csv_path = os.path.join("", file)
    if header:
        return pd.read_csv(csv_path)
    else:
        return pd.read_csv(csv_path, header=None)


data = load_wine_data(INPUT_FILE)
data.head()
data.info()
data.describe()
%matplotlib inline
import matplotlib.pyplot as plt
data.hist(bins=50, figsize=(20,15))
plt.show()
import numpy as np
data["quality_label"] = np.floor(data["quality"] / 7.0)

%matplotlib inline
import matplotlib.pyplot as plt
data.hist(bins=50, figsize=(20,15))
plt.show()

from sklearn .model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(data, data["quality_label"]):
    strat_train_set = data.loc[train_index]
    strat_test_set = data.loc[test_index]

data["quality_label"].value_counts() / len(data)
strat_train_set["quality_label"].value_counts() / len(strat_train_set)
strat_test_set["quality_label"].value_counts() / len(strat_test_set)
data_copy = data.copy()
for set in (strat_train_set, strat_test_set):
    set.drop(["quality_label"], axis=1, inplace=True)

corr_data = data.corr()
corr_data
#from pandas.plotting import scatter_matrix

#attributes = ["fixed acidity", "citric acid", "pH", "density", "free sulfur dioxide", "total sulfur dioxide"]
#scatter_matrix(data[attributes], figsize=(20,20))
#data.plot(kind="scatter", x="free sulfur dioxide", y="total sulfur dioxide", alpha=0.1)
#data.plot(kind="scatter", y="fixed acidity", x="citric acid", alpha=0.1)
#data.plot(kind="scatter", x="density", y="citric acid", alpha=0.1)
#data.plot(kind="scatter", y="density", x="fixed acidity", alpha=0.1)
data["sulfur_dioxide"] = data["free sulfur dioxide"] / data["total sulfur dioxide"]
from pandas.plotting import scatter_matrix

attributes = ["free sulfur dioxide", "total sulfur dioxide", "sulfur_dioxide"]
scatter_matrix(data[attributes], figsize=(20,20))
corr_data = data.corr()
corr_data
data = data.drop("total sulfur dioxide", axis=1)
data = data.drop("free sulfur dioxide", axis=1)
from sklearn .model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(data, data["quality_label"]):
    strat_train_set = data.loc[train_index]
    strat_test_set = data.loc[test_index]

label = strat_train_set["quality"].copy()
data = strat_train_set.drop("quality_label", axis=1)
test_label = strat_test_set["quality"].copy()
test_data = strat_test_set.drop("quality_label", axis=1)
from sklearn.preprocessing import Imputer
imputer = Imputer(strategy="median")
imputer.fit(data)
pre_tr_data = data.copy()
data_x = imputer.transform(data)
data = pd.DataFrame(data_x, columns=data.columns)

from sklearn.preprocessing import StandardScaler

scalar = StandardScaler()
scalar.fit(data)
data = pd.DataFrame(scalar.transform(data), columns=data.columns)

pre_tr_test_data = test_data.copy()
test_data = pd.DataFrame(imputer.transform(test_data), columns=test_data.columns)
test_data = pd.DataFrame(scalar.transform(test_data), columns=test_data.columns)
data.to_csv("data.csv", encoding='utf-8', index=False)
test_data.to_csv("test_data.csv", encoding='utf-8', index=False)
label.to_csv("label.csv", encoding='utf-8', index=False)
test_label.to_csv("test_label.csv", encoding='utf-8', index=False)
data = load_wine_data("data.csv")
data = data[list(data)].values

test_data = load_wine_data("test_data.csv")
test_data = test_data[list(test_data)].values

label = load_wine_data("label.csv", header=False)
test_label = load_wine_data("test_label.csv", header=False)

label = label.values
test_label = test_label.values
#Linear Regression

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(data, label)

predicted_test_label = lin_reg.predict(test_data)
y = np.floor(test_label / 7.0)
y_dash = np.floor(predicted_test_label / 7.0)
print("Accuracy on test data", sum(y == y_dash) * 100.0 / len(test_label), "%")



from sklearn.metrics import mean_squared_error

predicted_label = lin_reg.predict(data)
lin_mse = mean_squared_error(label, predicted_label)
lin_rmse = np.sqrt(lin_mse)
print("RMSE on training data", lin_rmse)


#Saving the model
from sklearn.externals import joblib

LINEAR_REG_MODEL_NAME = "linear_reg_model.pk1"
joblib.dump(lin_reg, LINEAR_REG_MODEL_NAME)


# Cross Validation
lin_reg = joblib.load(LINEAR_REG_MODEL_NAME)

from sklearn.model_selection import cross_val_score

scores = cross_val_score(lin_reg, data, label, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)
print("Mean", scores.mean())
print("SD", scores.std())

# Decision tree Regression

from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(data, label)

predicted_test_label = tree_reg.predict(test_data)
y = np.floor(test_label / 7.0).flatten()
y_dash = np.floor(predicted_test_label / 7.0).flatten()
print("Accuracy on test data", sum(y == y_dash) * 100.0 / len(test_label), "%")


from sklearn.metrics import mean_squared_error

predicted_label = tree_reg.predict(data)
tree_mse = mean_squared_error(label, predicted_label)
tree_rmse = np.sqrt(tree_mse)
print("RMSE on training data", tree_rmse)


#Saving the model
from sklearn.externals import joblib

TREE_MODEL_NAME = "tree_reg_model.pk1"
joblib.dump(tree_reg, TREE_MODEL_NAME)


# Cross Validation
tree_reg = joblib.load(TREE_MODEL_NAME)

from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree_reg, data, label, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)
print("Mean", scores.mean())
print("SD", scores.std())

