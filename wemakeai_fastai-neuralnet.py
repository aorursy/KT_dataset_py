import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
%matplotlib inline

from fastai.tabular import *
# setting data path

DATA_PATH = Path('../input')
DATA_PATH.ls()
df_train = pd.read_csv(DATA_PATH/'train.csv')

df_train.drop('Name', 1, inplace=True)

df_train.drop('PassengerId', 1, inplace=True)

df_train.drop('Cabin', 1, inplace=True)

#df_train
df_train.columns
# I remove the Cabin column because there are many Nans in this column

# I remove the Name column because I think that the name of the person won't have

# any consequence on surving or not

df_test = pd.read_csv(DATA_PATH/'test.csv')

df_test.drop('Name', 1, inplace=True)

df_test.drop('Cabin', 1, inplace=True)
df_test.columns
# fill remaining NAs with mean

df_test.fillna(df_test.mean(), inplace=True)
# Did the person travel alone?

for df in [df_train, df_test]:

  df['Alone'] = df['SibSp'] + df['Parch'] == 0

  df['Relatives'] = df['SibSp'] + df['Parch']
# remove columns with 0 or less fare

len(df_train[df_train.Fare==0]), len(df_train)
df_train = (df_train[df_train.Fare>0])
# instead of age as a value create age bins and remove age column:

# infant, child, young adult, adult, elderly

for df in [df_train, df_test]:

  df['Infant'] = df['Age'] <= 1

  df['Child'] = (df['Age'] > 1) & (df['Age'] <= 14)

  df['YoungAdult'] = (df['Age'] > 14) & (df['Age'] <= 22)

  df['Adult'] = (df['Age'] > 22) & (df['Age'] <= 60)

  df['Elderly'] = (df['Age'] > 60)

  df['Relatives'] = df['SibSp'] + df['Parch']

  df.drop('Age', 1, inplace=True)
# defining variables

dep_var = 'Survived'

cat_names = ['Pclass', 'Sex', 'Ticket', 'Embarked', 'Alone', 'SibSp', 'Parch', 'Infant', 'Child', 'YoungAdult', 'Adult', 'Elderly']

cont_names = ['Fare', 'Relatives']

procs = [FillMissing, Categorify, Normalize]
# creating test and training data

test = TabularList.from_df(df_test, path=DATA_PATH, cat_names=cat_names, cont_names=cont_names)

data = (TabularList.from_df(df_train, path=DATA_PATH, cat_names=cat_names, cont_names=cont_names, procs=procs)

        .split_by_rand_pct(0.1)        

        .label_from_df(cols=dep_var)   

        .add_test(test, label=0)

        .databunch(bs=8))
# creating the learner

learn = tabular_learner(data, layers=[5, 10], metrics=accuracy, emb_drop=0.2, y_range=(0,1.05), model_dir='../')
learn.lr_find()

learn.recorder.plot()
lr = 1e-1/2
learn.fit_one_cycle(10, lr)
learn.recorder.plot_losses()
learn.validate()
learn.save('titanic-nn-stage1')
learn.export('../export.pkl')
learn = load_learner('../', test=test)
preds, _ = learn.get_preds(ds_type=DatasetType.Test)
labels = np.argmax(preds, 1)
df_sub = pd.DataFrame({'PassengerId': df_test['PassengerId'], 'Survived': labels})

df_sub.to_csv('titanic_submission.csv', index=False)