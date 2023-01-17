import pandas as pd

import numpy as np



from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import Normalizer

from sklearn.cross_validation import cross_val_score

from sklearn.preprocessing import Imputer



from scipy.stats import skew



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



plt.style.use('ggplot')
train = '../input/train.csv'

test = '../input/test.csv'



df_train = pd.read_csv(train)

df_test = pd.read_csv(test)



print("a")
def is_outlier(points, thresh = 3.5):

    if len(points.shape) == 1:

        points = points[:,None]

    median = np.median(points, axis=0)

    diff = np.sum((points - median)**2, axis=-1)

    diff = np.sqrt(diff)

    med_abs_deviation = np.median(diff)



    modified_z_score = 0.6745 * diff / med_abs_deviation



    return modified_z_score > thresh
target = df_train[df_train.columns.values[-1]]

target_log = np.log(target)



plt.figure(figsize=(10,5))

plt.subplot(1,2,1)

sns.distplot(target, bins=50)

plt.title('Original Data')

plt.xlabel('Sale Price')



plt.subplot(1,2,2)

sns.distplot(target_log, bins=50)

plt.title('Natural Log of Data')

plt.xlabel('Natural Log of Sale Price')

plt.tight_layout()



print("b")
df_train = df_train[df_train.columns.values[:-1]]

df = df_train.append(df_test, ignore_index = True)
cats = []

for col in df.columns.values:

    if df[col].dtype == 'object':

        cats.append(col)
df_cont = df.drop(cats, axis=1)

df_cat = df[cats]



print("c")
for col in df_cont.columns.values:

    if np.sum(df_cont[col].isnull()) > 50:

        df_cont = df_cont.drop(col, axis = 1)

    elif np.sum(df_cont[col].isnull()) > 0:

        median = df_cont[col].median()

        idx = np.where(df_cont[col].isnull())[0]

        df_cont[col].iloc[idx] = median



        outliers = np.where(is_outlier(df_cont[col]))

        df_cont[col].iloc[outliers] = median

        

        if skew(df_cont[col]) > 0.75:

            df_cont[col] = np.log(df_cont[col])

            df_cont[col] = df_cont[col].apply(lambda x: 0 if x == -np.inf else x)

        

        df_cont[col] = Normalizer().fit_transform(df_cont[col].reshape(1,-1))[0]

        

print("d")
for col in df_cat.columns.values:

    if np.sum(df_cat[col].isnull()) > 50:

        df_cat = df_cat.drop(col, axis = 1)

        continue

    elif np.sum(df_cat[col].isnull()) > 0:

        df_cat[col] = df_cat[col].fillna('MIA')

        

    df_cat[col] = LabelEncoder().fit_transform(df_cat[col])

    

    num_cols = df_cat[col].max()

    for i in range(num_cols):

        col_name = col + '_' + str(i)

        df_cat[col_name] = df_cat[col].apply(lambda x: 1 if x == i else 0)

        

    df_cat = df_cat.drop(col, axis = 1)

    

print("e")
df_new = df_cont.join(df_cat)



df_train = df_new.iloc[:len(df_train) - 1]

df_train = df_train.join(target_log)



df_test = df_new.iloc[len(df_train) + 1:]



X_train = df_train[df_train.columns.values[1:-1]]

y_train = df_train[df_train.columns.values[-1]]



X_test = df_test[df_test.columns.values[1:]]



print("f")
from sklearn.metrics import make_scorer, mean_squared_error

scorer = make_scorer(mean_squared_error, False)



clf = RandomForestRegressor(n_estimators=530, n_jobs=-1)

cv_score = np.sqrt(-cross_val_score(estimator=clf, X=X_train, y=y_train, cv=3, scoring = scorer))



print(cv_score)

print(" ")

print(cv_score.mean())



#plt.figure(figsize=(10,5))

#plt.bar(range(len(cv_score)), cv_score)

#plt.title('Cross Validation Score')

#plt.ylabel('RMSE')

#plt.xlabel('Iteration')



#plt.plot(range(len(cv_score) + 1), [cv_score.mean()] * (len(cv_score) + 1))

#plt.tight_layout()



print("g")
# Fit model with training data

#clf.fit(X_train, y_train)



# Output feature importance coefficients, map them to their feature name, and sort values

#coef = pd.Series(clf.feature_importances_, index = X_train.columns).sort_values(ascending=False)



#plt.figure(figsize=(10, 5))

#coef.head(25).plot(kind='bar')

#plt.title('Feature Significance')

#plt.tight_layout()
from sklearn.cross_validation import train_test_split



X_train1, X_test1, y_train1, y_test1 = train_test_split(X_train, y_train)

clf = RandomForestRegressor(n_estimators=530, n_jobs=-1)



clf.fit(X_train1, y_train1)

y_pred = clf.predict(X_test1)



test_preds = clf.predict(X_test)



#print(cv_score.mean())



plt.figure(figsize=(10, 5))

plt.scatter(y_test1, y_pred, s=20)

plt.title('Predicted vs. Actual')

plt.xlabel('Actual Sale Price')

plt.ylabel('Predicted Sale Price')



plt.plot([min(y_test1), max(y_test1)], [min(y_test1), max(y_test1)])

plt.tight_layout()



print("h")



a = np.array(len(y_pred))

w = np.array(len(y_pred))

#for i in range(len(y_pred)):

   # a[i] = y_test1[i]

 #   w[i] = abs(y_test1[i] - y_pred[i])

w = abs(y_test1 - y_pred)



plt.hist(y_test1, bins = 50, weights = w)  # plt.hist passes it's arguments to np.histogram

plt.title("error distribution")

plt.show()



submission = pd.DataFrame()

submission['Id'] = df_test['Id']

submission["SalePrice"] = test_preds

submission.to_csv("lasso_by_Sarthak.csv", index=False, header = True)



print("j")