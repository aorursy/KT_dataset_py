import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns
data_train = pd.read_csv("../input/train.csv")

data_test = pd.read_csv("../input/test.csv")
data_train.info()
data_train.describe()
plt.scatter(data_train.GrLivArea, data_train.SalePrice)

plt.show()
data_train = data_train.drop(data_train[(data_train["GrLivArea"] > 4000) & (data_train["SalePrice"]<300000)].index)

plt.scatter(data_train.GrLivArea, data_train.SalePrice)

plt.show()
sns.scatterplot("LotArea", "SalePrice", data=data_train)

plt.show()
data_train = data_train.drop(data_train[(data_train["LotArea"] > 100000) & (data_train["SalePrice"]<400000)].index)

plt.scatter(data_train.GrLivArea, data_train.SalePrice)

plt.show()
ntrain = data_train.shape[0]

ntest = data_test.shape[0]

train_label = data_train["SalePrice"]

data_train = data_train.drop(["SalePrice"], axis=1)

all_data = pd.concat((data_train, data_test)).reset_index(drop=True)
total = all_data.isnull().sum().sort_values(ascending=False)

percent = (all_data.isnull().sum()/all_data.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(35)
for col in ("PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu"):

    all_data["has_" + col] = all_data[col].apply(lambda x: 0 if pd.isnull(x) else 1)

    all_data[col] =  all_data[col].fillna("None")
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(

    lambda x: x.fillna(x.mean()))
all_data["has_garage"] = all_data["GarageCond"].apply(lambda x: 0 if pd.isnull(x) else 1)

for col in ("GarageType", "GarageFinish", "GarageQual","GarageCond"):

    all_data[col] = all_data[col].fillna('None')

for col in ("GarageYrBlt", "GarageCars","GarageArea"):

    all_data[col] = all_data[col].fillna(0)
for col in ("BsmtCond", "BsmtExposure", "BsmtQual", "BsmtFinType2", "BsmtFinType1"):

    all_data[col] = all_data[col].fillna('None')

for col in ("BsmtHalfBath", "BsmtFullBath", "BsmtFinSF2", "BsmtUnfSF", "BsmtFinSF1","TotalBsmtSF"):

    all_data[col] = all_data[col].fillna(0)
all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")

all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)
for col in ("MSZoning","Functional", "Electrical","KitchenQual", "Exterior1st","Exterior2nd","SaleType"):

    all_data[col] = all_data[col].fillna(all_data[col].mode()[0])
all_data.Utilities.value_counts()
all_data = all_data.drop(["Utilities"], axis=1)
all_data["Age_of_house"] = all_data.YrSold - all_data.YearBuilt 
all_data.columns[all_data.dtypes == "object"]
from sklearn.preprocessing import LabelEncoder

for col in all_data.columns[all_data.dtypes == "object"]:

    all_data[col] = all_data[col].factorize()[0]

all_data = all_data.drop(["Id"], axis=1)
data_train = all_data.iloc[:ntrain]

data_test = all_data.iloc[ntrain:]
from sklearn.model_selection import train_test_split

train_x, val_x, train_y, val_y = train_test_split(data_train, train_label, test_size=0.3, random_state=42)
from xgboost import XGBRegressor

from sklearn.model_selection import cross_val_score

def model_tuning(n_jobs=4,colsample_bylevel = 1, colsample_bytree=1, gamma=0.0, 

                         learning_rate=0.1, max_depth= 3, min_child_weight=1, n_estimators=300, reg_alpha=0, reg_lambda=1,subsample=1):

    model = XGBRegressor(n_jobs=n_jobs, objective='reg:linear', seed=42,silent=True,colsample_bylevel = colsample_bylevel, colsample_bytree=colsample_bytree,

                         gamma=gamma,learning_rate=learning_rate, max_depth= max_depth,

                         min_child_weight=min_child_weight, n_estimators=n_estimators, reg_alpha=reg_alpha, reg_lambda=reg_lambda,subsample=subsample)

    model_cv = cross_val_score(model, train_x, train_y, cv=5, scoring="neg_mean_squared_log_error")

    return np.sqrt(-1*model_cv.mean())
accuracies = []

params = np.arange(1,8)

for i in params:

    accuracies.append(model_tuning(max_depth=i)) #0.2, 0.3 is looks like good

plt.plot(params,accuracies)

plt.show()
accuracies = []

params = np.arange(1,11)/10

for i in params:

    accuracies.append(model_tuning(max_depth=3, subsample=i))

plt.plot(params,accuracies)

plt.show()
accuracies = []

params = np.arange(1,11)/10

for i in params:

    accuracies.append(model_tuning(max_depth=3, subsample=0.8, colsample_bytree=i))

plt.plot(params,accuracies)

plt.show()
accuracies = []

params = np.arange(1,11)/10

for i in params:

    accuracies.append(model_tuning(max_depth=3, subsample=0.8, colsample_bytree=0.2, colsample_bylevel=i))

plt.plot(params,accuracies)

plt.show()
accuracies = []

params = np.arange(650,1000,200)

for i in params:

    accuracies.append(model_tuning(max_depth=3, subsample=0.8, colsample_bytree=0.2, colsample_bylevel=1.0, n_estimators=i))

plt.plot(params,accuracies)

plt.show()
accuracies = []

params = [0.01, 0.03, 0.05, 0.07, 0.1]

for i in params:

    accuracies.append(model_tuning(max_depth=3, subsample=0.8, colsample_bytree=0.2, colsample_bylevel=1.0, n_estimators=650,learning_rate=i))

plt.plot(params,accuracies)

plt.show()
model = XGBRegressor(max_depth=3, subsample=0.8, colsample_bytree=0.2, colsample_bylevel=1.0, n_jobs=4, n_estimators=600, learning_rate=0.03, seed=42)

model.fit(train_x, train_y)
from sklearn.metrics import mean_squared_log_error

print("val acc", np.sqrt(mean_squared_log_error(model.predict(val_x), val_y)))

print("train acc", np.sqrt(mean_squared_log_error(model.predict(train_x), train_y)))
def model_tuner(max_depth=4, subsample=0.68, 

                colsample_bytree=0.2, colsample_bylevel=1.0,

                n_estimators=600, learning_rate=0.03,

                min_child_weight=4.5, reg_lambda=5.9):

    

    model = XGBRegressor(max_depth=max_depth, subsample=subsample, 

                         colsample_bytree=colsample_bytree, colsample_bylevel=colsample_bylevel,

                         n_jobs=4, n_estimators=n_estimators, learning_rate=learning_rate, seed=42,

                         min_child_weight=min_child_weight, reg_lambda=reg_lambda)

    model.fit(train_x, train_y)

    train_accuracy = np.sqrt(mean_squared_log_error(model.predict(train_x), train_y))

    test_accuracy = np.sqrt(mean_squared_log_error(model.predict(val_x), val_y))

    return train_accuracy, test_accuracy
# train_accuracies = []

# test_accuracies = []

# params = np.arange(15,75)/10

# for i in params:

#     train_acc, test_acc = model_tuner(min_child_weight=i)

#     train_accuracies.append(train_acc)

#     test_accuracies.append(test_acc)

# sns.lineplot(params,train_accuracies,color="r" )

# sns.lineplot(params,test_accuracies,color="b" )

# plt.show()
# df = pd.DataFrame({"param": params, "test_acc":test_accuracies, "train_acc":train_accuracies})

# df.sort_values(by="test_acc").head(10)
# train_accuracies = []

# test_accuracies = []

# params = np.arange(1,100)/10

# for i in params:

#     train_acc, test_acc = model_tuner(reg_lambda=i)

#     train_accuracies.append(train_acc)

#     test_accuracies.append(test_acc)

# sns.lineplot(params,train_accuracies,color="r" )

# sns.lineplot(params,test_accuracies,color="b" )

# plt.show()
# df = pd.DataFrame({"param": params, "test_acc":test_accuracies, "train_acc":train_accuracies})

# df.sort_values(by="test_acc").head(10)
# train_accuracies = []

# test_accuracies = []

# params = np.arange(1,15)

# for i in params:

#     train_acc, test_acc = model_tuner(max_depth=i)

#     train_accuracies.append(train_acc)

#     test_accuracies.append(test_acc)

# sns.lineplot(params,train_accuracies,color="r" )

# sns.lineplot(params,test_accuracies,color="b" )

# plt.show()
# df = pd.DataFrame({"param": params, "test_acc":test_accuracies, "train_acc":train_accuracies})

# df.sort_values(by="test_acc").head(10)
# train_accuracies = []

# test_accuracies = []

# params = np.arange(1,100)/100

# for i in params:

#     train_acc, test_acc = model_tuner(subsample=i)

#     train_accuracies.append(train_acc)

#     test_accuracies.append(test_acc)

# sns.lineplot(params,train_accuracies,color="r" )

# sns.lineplot(params,test_accuracies,color="b" )

# plt.show()
# df = pd.DataFrame({"param": params, "test_acc":test_accuracies, "train_acc":train_accuracies})

# df.sort_values(by="test_acc").head(10)
# train_accuracies = []

# test_accuracies = []

# params = np.arange(1,100)/100

# for i in params:

#     train_acc, test_acc = model_tuner(colsample_bytree=i)

#     train_accuracies.append(train_acc)

#     test_accuracies.append(test_acc)

# sns.lineplot(params,train_accuracies,color="r" )

# sns.lineplot(params,test_accuracies,color="b" )

# plt.show()
# df = pd.DataFrame({"param": params, "test_acc":test_accuracies, "train_acc":train_accuracies})

# df.sort_values(by="test_acc").head(10)
# train_accuracies = []

# test_accuracies = []

# params = np.arange(1,101)/100

# for i in params:

#     train_acc, test_acc = model_tuner(colsample_bylevel=i)

#     train_accuracies.append(train_acc)

#     test_accuracies.append(test_acc)

# sns.lineplot(params,train_accuracies,color="r" )

# sns.lineplot(params,test_accuracies,color="b" )

# plt.show()
# df = pd.DataFrame({"param": params, "test_acc":test_accuracies, "train_acc":train_accuracies})

# df.sort_values(by="test_acc").head(10)
model = XGBRegressor(max_depth=4, subsample=0.68, colsample_bytree=0.2,

                     colsample_bylevel=1.0, n_estimators=5000, 

                     learning_rate=0.01, min_child_weight=4.5, 

                     reg_lambda=5.9, n_jobs=4, seed=42)

history = model.fit(train_x, train_y,

             eval_set=[(val_x, val_y)], verbose=False)
plt.plot(np.arange(1000,5000),history.evals_result_["validation_0"]["rmse"][1000:])

plt.show()
model = XGBRegressor(max_depth=4, subsample=0.68, colsample_bytree=0.2,

                     colsample_bylevel=1.0, n_estimators=3000, 

                     learning_rate=0.01, min_child_weight=4.5, 

                     reg_lambda=5.9, n_jobs=4, seed=42)

accuracy = cross_val_score(model, data_train, train_label, cv=5, scoring="neg_mean_squared_log_error")

print("Accuracy", np.sqrt(-accuracy.mean()))
model.fit(data_train, train_label)
y_pred = model.predict(data_test)

submission = pd.read_csv("../input/sample_submission.csv")

submission["SalePrice"] = y_pred

submission.to_csv('submission.csv', index=False)