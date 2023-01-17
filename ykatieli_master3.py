# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

#df = pd.read_csv("../input/master/master.csv")

df = pd.read_csv("/kaggle/input/masterr/master2.csv")

print(f'Number of rows    = {len(df)}')

print(f'Number of columns = {len(df.columns)}')

df.head()
df['holistic'] = df.Founding_Year*0.1 + df.Article_Mentions*0.1 + df.Portfolio*0.1 + df.Investments*0.1 + df.Lead_Investments*0.1 + df.Exits*0.1 + df.TVPI*0.1 + df.IRR*0.1 + df.DPI*0.1 + df.RVPI*0.1 + df.Gain_Since_Inception*0.1 + df.Funding_Rounds*-0.1 + df.Total_Funding*0.1

df.sort_values(by='holistic', ascending=False, kind='quicksort', na_position='last', ignore_index=True)
for col in df.columns: 

    print(col)
df.sort_values(by='Name', ascending=True, kind='quicksort', na_position='last', ignore_index=True)
df.sort_values(by='Total_Funding', ascending=False, kind='quicksort', na_position='last', ignore_index=True)
df.groupby(['Founding_Year']).Investments.agg([min, max])
import seaborn as sns



#sns.heatmap(

#    df.loc[:, ['Investments', 'Funding_Rounds', 'Exits', 'IRR', 'Founding_Year', 'Total_Funding']].corr(),

#    annot=True

#)



(df[['Investments','Lead_Investments', 'Article_Mentions', 'Total_Funding']].corr())

ax = sns.heatmap(df.corr(),cmap="Blues",annot=False,annot_kws={"size": 5},linewidths=0.2)

ax.set_xticklabels(ax.get_xticklabels(), rotation=50, ha="right");

# plt.tight_layout()
#df[df['Investments'] < 100].plot.hexbin(x='Exits', y='Investments', gridsize=15)

df[df['Investments'] < 900].sample(900).plot.scatter(x='Investments', y='Lead_Investments')

df[df['Total_Funding'] < 100000000].sample(100).plot.scatter(x='Total_Funding', y='Investments')
missing_values = (df.isnull().sum())

print(missing_values[missing_values > 0])
#Total_Funding = list(Total_Funding[:10])

#print(Total_Funding)

#print(sorted(Total_Funding))
import pandas as pd



# Load data

melb_data = pd.read_csv('../input/master/master.csv')



from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split



melb_target = melb_data.Total_Funding

melb_predictors = melb_data.drop(['Total_Funding'], axis=1)



# Using only numeric predictors

melb_numeric_predictors = melb_predictors.select_dtypes(exclude=['object'])





X_train, X_test, y_train, y_test = train_test_split(melb_numeric_predictors, 

                                                    melb_target,

                                                    train_size=0.7, 

                                                    test_size=0.3, 

                                                    random_state=0)



def score_dataset(X_train, X_test, y_train, y_test):

    model = RandomForestRegressor()

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    return mean_absolute_error(y_test, preds)





cols_with_missing = [col for col in X_train.columns 

                                 if X_train[col].isnull().any()]

reduced_X_train = X_train.drop(cols_with_missing, axis=1)

reduced_X_test  = X_test.drop(cols_with_missing, axis=1)

print("Mean Absolute Error from dropping columns with Missing Values:")

print(score_dataset(reduced_X_train, reduced_X_test, y_train, y_test))





from sklearn.impute import SimpleImputer



my_imputer = SimpleImputer()

imputed_X_train = my_imputer.fit_transform(X_train)

imputed_X_test = my_imputer.transform(X_test)

print("Mean Absolute Error from Imputation:")

print(score_dataset(imputed_X_train, imputed_X_test, y_train, y_test))





imputed_X_train_plus = X_train.copy()

imputed_X_test_plus = X_test.copy()



cols_with_missing = (col for col in X_train.columns 

                                 if X_train[col].isnull().any())

for col in cols_with_missing:

    imputed_X_train_plus[col + '_was_missing'] = imputed_X_train_plus[col].isnull()

    imputed_X_test_plus[col + '_was_missing'] = imputed_X_test_plus[col].isnull()



# Imputation

my_imputer = SimpleImputer()

imputed_X_train_plus = my_imputer.fit_transform(imputed_X_train_plus)

imputed_X_test_plus = my_imputer.transform(imputed_X_test_plus)



print("Mean Absolute Error from Imputation while Track What Was Imputed:")

print(score_dataset(imputed_X_train_plus, imputed_X_test_plus, y_train, y_test))