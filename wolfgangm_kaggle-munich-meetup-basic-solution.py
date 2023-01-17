import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from scipy.stats import skew

from sklearn import linear_model

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import cross_val_score

from IPython.display import display
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
print("Number of features in training set: {}".format(train.shape[1]))

print("Number of training data entries: {}".format(train.shape[0]))

print("Number of test data entries: {}".format(test.shape[0]))
print("First column in both sets is: {}".format(train.columns[0]))

print("Last column in training set is: {}".format(train.columns[-1]))
train.head(10)
fig, axs = plt.subplots(1,2,figsize=(12,7))



train['LotArea'].plot.density(ax=axs[0])

train['LotArea'].plot.density(ax=axs[1])



print("Max. lot area: {} ft²".format(train['LotArea'].max()))

print("Mean lot area: {:.2f} ft²".format(train['LotArea'].mean()))
axs[0].set_xlim([0,train['LotArea'].max()])

axs[0].set_xlabel("ft²")

axs[0].set_ylabel("density")



axs[1].set_xlim([30000,train['LotArea'].max()]); axs[1].set_ylim([0.,0.00000075])

axs[1].set_xlabel("ft²"); axs[1].set_ylabel("")
plt.show()
train.set_index('Id', inplace=True)

test.set_index('Id', inplace=True)
train_price = train["SalePrice"]

train.drop("SalePrice", axis=1, inplace=True)
plt.rcParams['figure.figsize'] = (12, 6)

train.boxplot(showfliers=False, rot=90)

plt.show()
# extract locations of numerical features

num_feat = (train.dtypes != "object").as_matrix()

print("Number of numerical features: {}".format(np.sum(num_feat)))

    

train_num = train.iloc[:, num_feat]

test_num = test.iloc[:, num_feat]
print("All numerical values in training set positive? {}"

      .format(not (train_num < 0).any().any()))

# fill missing values in training set with column means

train_num = train_num.fillna(train_num.mean())

train.iloc[:, num_feat] = train_num



print("All numerical values in test set positive? {}"

      .format(not (test_num < 0).any().any()))

# fill missing values in test set with TRAINING column means

test_num = test_num.fillna(train_num.mean())

test.iloc[:, num_feat] = test_num
ax = train_price.plot.density()

ax.set_xlim([0,train_price.max()])

ax.set_xlabel("price")

ax.set_ylabel("density")

plt.show()
print("Numerical feature columns:")

display(train.columns[num_feat])

print("Skeweness of numerical training features:")

display(skew(train_num))
skewed = (np.absolute(skew(train_num)) > 1)

train_num.iloc[:, skewed] = np.log1p(train_num.iloc[:, skewed])

test_num.iloc[:, skewed] = np.log1p(test_num.iloc[:, skewed])

train_price = np.log1p(train_price)



print("Skeweness of numerical training features after transformation:")

display(skew(train_num))
scaler = StandardScaler().fit(train_num)

train_num = scaler.transform(train_num)

test_num = scaler.transform(test_num)
train.iloc[:, num_feat] = train_num

test.iloc[:, num_feat] = test_num
train_test = pd.concat([train, test])

train_test = pd.get_dummies(train_test)

train_test.iloc[:,40:].head(4)
train = train_test.iloc[:train.shape[0], :]

test = train_test.iloc[train.shape[0]:, :]
lin_reg = linear_model.LinearRegression()

scores = cross_val_score(lin_reg, train, train_price,

                         cv=5, scoring='neg_mean_squared_error')

print("Mean of 5 CV sqrt MSE: {:.4f}".format(np.sqrt(-scores.mean())))
ridge = linear_model.Ridge(alpha=10.)

scores = cross_val_score(ridge, train, train_price,

                         cv=5, scoring='neg_mean_squared_error')

print("Mean of 5 CV sqrt MSE: {:.4f}".format(np.sqrt(-scores.mean())))
ridge.fit(train, train_price)
preds = ridge.predict(test)
preds_price = np.expm1(preds)
test_results = pd.DataFrame({'SalePrice': preds_price,

                             'Id': test.index})

test_results.set_index('Id', inplace=True)



test_results.to_csv("test_results.csv")