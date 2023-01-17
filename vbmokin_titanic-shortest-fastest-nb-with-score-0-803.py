# Import libraries

import pandas as pd; import numpy as np 



# Download data

traindf = pd.read_csv('../input/titanic/train.csv').set_index('PassengerId')

testdf = pd.read_csv('../input/titanic/test.csv').set_index('PassengerId')

df = pd.concat([traindf, testdf], axis=0, sort=False)



# FE

df['Title'] = df.Name.str.split(',').str[1].str.split('.').str[0].str.strip()

df['IsWomanOrBoy'] = ((df.Title == 'Master') | (df.Sex == 'female'))

df['LastName'] = df.Name.str.split(',').str[0]

family = df.groupby(df.LastName).Survived

df['WomanOrBoyCount'] = family.transform(lambda s: s[df.IsWomanOrBoy].fillna(0).count())

df['WomanOrBoyCount'] = df.mask(df.IsWomanOrBoy, df.WomanOrBoyCount - 1, axis=0)

df['FamilySurvivedCount'] = family.transform(lambda s: s[df.IsWomanOrBoy].fillna(0).sum())

df['FamilySurvivedCount'] = df.mask(df.IsWomanOrBoy, df.FamilySurvivedCount - df.Survived.fillna(0), axis=0)

df['WomanOrBoySurvived'] = df.FamilySurvivedCount / df.WomanOrBoyCount.replace(0, np.nan)

df['Alone'] = (df.WomanOrBoyCount == 0)

test_x = pd.concat([df.WomanOrBoySurvived.fillna(0), df.Alone, df.Sex.replace({'male': 0, 'female': 1})], axis=1).loc[testdf.index]



# Prediction

test_x['Survived'] = (((test_x.WomanOrBoySurvived <= 0.238) & (test_x.Sex > 0.5) & (test_x.Alone > 0.5)) | \

          ((test_x.WomanOrBoySurvived > 0.238) & \

           ~((test_x.WomanOrBoySurvived > 0.55) & (test_x.WomanOrBoySurvived <= 0.633))))



# Submission

pd.DataFrame({'Survived': test_x['Survived'].astype(int)}, \

             index=testdf.index).reset_index().to_csv('survived.csv', index=False)