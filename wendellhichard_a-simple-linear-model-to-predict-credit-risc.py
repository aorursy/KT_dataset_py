# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')
# import dependencies

# data cleaning and manipulation 

import pandas as pd

import numpy as np



# data visualization

import matplotlib.pyplot as plt

import seaborn as sns



# machine learning

from sklearn.preprocessing import StandardScaler



import sklearn.linear_model as skl_lm

from sklearn import preprocessing

from sklearn import neighbors

from sklearn.feature_selection import SelectKBest, chi2

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.metrics import confusion_matrix, classification_report, precision_score

from sklearn.model_selection import train_test_split





import statsmodels.api as sm

import statsmodels.formula.api as smf





# initialize some package settings

sns.set(style="whitegrid", color_codes=True, font_scale=1.3)



%matplotlib inline
df_train.shape
pd.set_option('display.max_columns', 100)

pd.set_option('display.max_rows', 100)
df_train.head(3)
ax = sns.distplot(df_train['Time'])
pd.DataFrame(df_train.isna().sum(),columns = ['nan_values']).T
df_dummies = df_train
y_labels = ['Class']

X_labels = list(set(list(df_dummies.columns)) - set(y_labels))
X = df_dummies[X_labels]  #independent columns

y = df_dummies[y_labels]    #target column i.e price range
top = 20
plt.figure(figsize=(20,20))

model = ExtraTreesClassifier();

model.fit(X,y);

# print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers

#plot graph of feature importances for better visualization

feat_importances = pd.Series(model.feature_importances_, index=X.columns);

feat_importances.nlargest(top).plot(kind='barh');

plt.show();
hdd_list_02 = feat_importances.nlargest(top).index.tolist()

hdd_list_02.extend(y_labels)
# Generate and visualize the correlation matrix

corr = df_dummies[hdd_list_02].corr().round(2)



# Mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set figure size

f, ax = plt.subplots(figsize=(20, 20))



# Define custom colormap

# cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap

sns.heatmap(corr, mask=mask, cmap="RdYlGn", vmin=-1, vmax=1, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)



plt.tight_layout()
best_features = hdd_list_02[0:13]
df_dummies[best_features].head(3)
# generate a scatter plot matrix with the "mean" columns

df_pairplot = pd.concat([df_dummies[best_features], y],axis=1);
X_train, X_test, y_train, y_test = train_test_split(

    df_pairplot, df_dummies[y_labels], test_size=0.33, random_state=42)
cols = best_features

formula = y_labels[0]+' ~ ' + ' + '.join(cols)

print(formula, '\n')
# Run the model and report the results

model = smf.glm(formula=formula, data=X_train, family=sm.families.Binomial())

logistic_fit = model.fit()



print(logistic_fit.summary())
# predict the test data and show the first 5 predictions

predictions = logistic_fit.predict(X_test)

predictions[1:6]
predictions_nominal = [ 0 if x < 0.5 else 1 for x in predictions]

predictions_nominal[1:6]
print(classification_report(y_test, predictions_nominal, digits=3))



cfm = confusion_matrix(y_test, predictions_nominal)



true_negative = cfm[0][0]

false_positive = cfm[0][1]

false_negative = cfm[1][0]

true_positive = cfm[1][1]



print('Confusion Matrix: \n', cfm, '\n')



print('True Negative:', true_negative)

print('False Positive:', false_positive)

print('False Negative:', false_negative)

print('True Positive:', true_positive)

print('Correct Predictions', 

      round((true_negative + true_positive) / len(predictions_nominal) * 100, 1), '%')