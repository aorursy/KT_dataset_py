# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

%matplotlib inline

from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
pd.set_option('display.max_columns', None)
raw_train_data = pd.read_csv('/kaggle/input/titanic/train.csv', index_col='PassengerId')

raw_gender_data = pd.read_csv('/kaggle/input/titanic/gender_submission.csv', index_col='PassengerId')

raw_test_data = pd.read_csv('/kaggle/input/titanic/test.csv', index_col='PassengerId')
raw_train_data.head()
raw_train_data.isnull().sum()
raw_train_data.describe()
col_names = set(raw_train_data.columns)

object_nunique = list(map(lambda col: raw_train_data[col].nunique(), col_names))

d = dict(zip(col_names, object_nunique))

sorted(d.items(), key=lambda x: x[1])
def survived_count_by_key(data, key):

    survived_counted_by_key = []

    not_survived_counted_by_key = []   

    survived_rate_by_key = []

    range_value = range(int(data[key].min()), int(data[key].max()))

        

    for i in range_value:

        num_key = data[data[key] == i]['Survived'].count()

        survied_counted_by_key_each = data[data[key] == i]['Survived'].sum()

        not_survived_counted_by_key_each = num_key - survied_counted_by_key_each

        survived_rate = (survied_counted_by_key_each / num_key) if num_key != 0 else 0

        

        survived_counted_by_key.append(survied_counted_by_key_each)

        not_survived_counted_by_key.append(not_survived_counted_by_key_each)

        survived_rate_by_key.append(survived_rate)



    return (range_value, (survived_counted_by_key, not_survived_counted_by_key, survived_rate_by_key))
def plot_scatter_survived_counted_by_key(data, key):

    x, y_tupple = survived_count_by_key(data, key)

    

    plt.title('Survived - Each ' + key)

    plt.scatter(x, y_tupple[0])

    plt.show()



    plt.title('NOT Survived - Each ' + key)

    plt.scatter(x, y_tupple[1])

    plt.show()



    plt.title('Survived Rate- Each ' + key)

    plt.scatter(x, y_tupple[2])

    plt.show()
def plot_hist(data, key, bins):

    survived_data = [data[data.Survived == 1]]

    not_survived_data = [data[data.Survived == 0]]

    splited_data = survived_data + not_survived_data

    hist_data = [d[key].dropna() for d in splited_data]

    

    plt.title('Hist of ' + key  + ' ▼Survived - ▲Not Surved')

    plt.hist(hist_data, histtype="barstacked", bins=bins)

    plt.show()
plot_hist(raw_train_data, 'Sex', 2)

plot_hist(raw_train_data, 'Pclass', 3)

plot_hist(raw_train_data, 'Embarked', 3)

plot_hist(raw_train_data, 'SibSp', 8)

plot_hist(raw_train_data, 'Parch', 7)

plot_hist(raw_train_data, 'Age', 15)

plot_hist(raw_train_data, 'Fare', 20)
plot_scatter_survived_counted_by_key(raw_train_data, 'Age')
plot_scatter_survived_counted_by_key(raw_train_data, 'Pclass')
y = raw_train_data.Survived

X_full = raw_train_data.drop(['Survived'], axis=1)

X_test = raw_test_data.copy()



# Break off validation set from training data

X_train, X_valid, y_train, y_valid = train_test_split(X_full, y, train_size=0.8, test_size=0.2, random_state=0)
droped_features = ['Cabin', 'Name', 'Ticket']

reduced_X_train = X_train.drop(droped_features, axis=1)

reduced_X_valid = X_valid.drop(droped_features, axis=1)
def impute_missing_num_to_mean(data, key):

    imputed_value = data[key].mean()

    data[key] = [imputed_value if np.isnan(data) else data for data in data[key]]

    return data
def impute_missing_str_to_dummy(data, key):

    imputed_value = 'MISSING VALUE'

    data[key] = [imputed_value if pd.isnull(data) else data for data in data[key]]

    return data
imputed_X_train = reduced_X_train.copy()

imputed_X_valid = reduced_X_valid.copy()



imputed_X_train = impute_missing_num_to_mean(imputed_X_train, 'Age')

imputed_X_valid = impute_missing_num_to_mean(imputed_X_valid, 'Age')

imputed_X_train = impute_missing_str_to_dummy(imputed_X_train, 'Embarked')

imputed_X_valid = impute_missing_str_to_dummy(imputed_X_valid, 'Embarked')





imputed_X_train.isnull().sum()
def label_encode(data):    

    from sklearn.preprocessing import LabelEncoder

    label_encoder = LabelEncoder()

    label_cols = ['Sex', 'Embarked']

    for col in label_cols:

        data[col] = label_encoder.fit_transform(data[col])

    return data
label_X_train = imputed_X_train.copy()

label_X_valid = imputed_X_valid.copy()



label_X_train = label_encode(label_X_train)

label_X_valid = label_encode(label_X_valid)



label_X_train.describe()
def evaluate_model(y_valid, preds_valid):

    l_y_valid = list(y_valid)

    num_correct = 0

    for i in range(0, len(l_y_valid)):

        if preds_valid[i] == l_y_valid[i]:

            num_correct += 1

    return num_correct / (len(y_valid))

    
preds_valid_list = []

for i in range(10, 101):

    t_model = RandomForestClassifier(n_estimators=i, random_state=0)

    t_model.fit(label_X_train, y_train)

    t_preds_valid = t_model.predict(label_X_valid)

    t_evaluated = evaluate_model(y_valid, t_preds_valid)

    preds_valid_list.append((i, t_evaluated))

    print('n_estimator = ' + str(i) + ' => ' + str(t_evaluated * 100) + '%')

    

preds_valid_list    
model = RandomForestClassifier(n_estimators=100, random_state=0)

model.fit(label_X_train, y_train)

preds_valid = model.predict(label_X_valid)

evaluate_model(y_valid, preds_valid)
X_test.isnull().sum()
reduced_X_test = X_test.drop(droped_features, axis=1)

imputed_X_test = impute_missing_num_to_mean(reduced_X_test, 'Age')

imputed_X_test = impute_missing_num_to_mean(reduced_X_test, 'Fare')

label_X_test = label_encode(imputed_X_test)



label_X_test.head()
preds_test = model.predict(label_X_test)
# Save test predictions to file

output = pd.DataFrame({'PassengerId': label_X_test.index,

                       'Survived': preds_test})

output.to_csv('submission.csv', index=False)