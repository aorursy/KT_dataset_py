!pip install -U pandas-profiling
import numpy as np

import pandas as pd

import pandas_profiling as pp

from pandas_profiling import ProfileReport
pp.__version__
traindf = pd.read_csv('../input/titanic/train.csv').set_index('PassengerId')

testdf = pd.read_csv('../input/titanic/test.csv').set_index('PassengerId')
traindf.head(3)
testdf.head(3)
# Thanks to: 

# https://www.kaggle.com/mauricef/titanic

# https://www.kaggle.com/vbmokin/titanic-top-3-one-line-of-the-prediction-code



df = pd.concat([traindf, testdf], axis=0, sort=False)

df['Title'] = df.Name.str.split(',').str[1].str.split('.').str[0].str.strip()

df['IsWomanOrBoy'] = ((df.Title == 'Master') | (df.Sex == 'female'))

df['LastName'] = df.Name.str.split(',').str[0]

family = df.groupby(df.LastName).Survived

df['WomanOrBoyCount'] = family.transform(lambda s: s[df.IsWomanOrBoy].fillna(0).count())

df['WomanOrBoyCount'] = df.mask(df.IsWomanOrBoy, df.WomanOrBoyCount - 1, axis=0)

df['FamilySurvivedCount'] = family.transform(lambda s: s[df.IsWomanOrBoy].fillna(0).sum())

df['FamilySurvivedCount'] = df.mask(df.IsWomanOrBoy, df.FamilySurvivedCount - \

                                    df.Survived.fillna(0), axis=0)

df['WomanOrBoySurvived'] = df.FamilySurvivedCount / df.WomanOrBoyCount.replace(0, np.nan)

df.WomanOrBoyCount = df.WomanOrBoyCount.replace(np.nan, 0)

df['Alone'] = (df.WomanOrBoyCount == 0)



#Thanks to: https://www.kaggle.com/kpacocha/top-6-titanic-machine-learning-from-disaster

#"Title" improvement

df['Title'] = df['Title'].replace('Ms','Miss')

df['Title'] = df['Title'].replace('Mlle','Miss')

df['Title'] = df['Title'].replace('Mme','Mrs')

# Embarked

df['Embarked'] = df['Embarked'].fillna('S')



# Thanks to https://www.kaggle.com/erinsweet/simpledetect

# Fare

med_fare = df.groupby(['Pclass', 'Parch', 'SibSp']).Fare.median()[3][0][0]

df['Fare'] = df['Fare'].fillna(med_fare)

#Age

df['Age'] = df.groupby(['Sex', 'Pclass', 'Title'])['Age'].apply(lambda x: x.fillna(x.median()))

# Family_Size

df['Family_Size'] = df['SibSp'] + df['Parch'] + 1



#Thanks to https://www.kaggle.com/kpacocha/top-6-titanic-machine-learning-from-disaster

# Cabin, Deck

#df['Deck'] = df['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')

#df.loc[(df['Deck'] == 'T'), 'Deck'] = 'A'



df.WomanOrBoySurvived = df.WomanOrBoySurvived.fillna(0)

df.WomanOrBoyCount = df.WomanOrBoyCount.fillna(0)

df.FamilySurvivedCount = df.FamilySurvivedCount.fillna(0)

df.Alone = df.Alone.fillna(0)
train_x, test_x = df.loc[traindf.index], df.loc[testdf.index]

test_x = test_x.drop('Survived', axis=1)
train_x.head(3)
test_x.head(3)
train_x.describe()
test_x.describe()
ProfileReport(train_x, title='Pandas Profiling Report for training dataset', html={'style':{'full_width':True}})
# %%time

# profile = train_x.profile_report(title='Pandas Profiling Report for training dataset')

# profile.to_file(output_file="train_profile.html")
%%time

profile = ProfileReport(train_x, title='Pandas Profiling Report for training dataset', minimal=True)

profile.to_file(output_file="train_short_profile.html")
ProfileReport(test_x, title='Pandas Profiling Report for test dataset')
# The one line of the code for prediction : LB = 0.80382 (Titanic Top 6%) 

test_x = pd.concat([test_x.WomanOrBoySurvived.fillna(0), test_x.Alone, \

                    test_x.Sex.replace({'male': 0, 'female': 1})], axis=1)

pd.DataFrame({'Survived': (((test_x.WomanOrBoySurvived <= 0.2381) & (test_x.Sex > 0.5) & (test_x.Alone > 0.5)) | \

                        ((test_x.WomanOrBoySurvived > 0.2381) & \

                       ~((test_x.WomanOrBoySurvived > 0.55) & (test_x.WomanOrBoySurvived <= 0.633)))).astype(int)}, index=testdf.index).reset_index().to_csv('submission.csv', index=False)