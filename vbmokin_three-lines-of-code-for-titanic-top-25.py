import pandas as pd

import warnings

def ignore_warn(*args, **kwargs):

    pass

warnings.warn = ignore_warn
# Preparatory part of the code

test = pd.read_csv('../input/titanic/test.csv') # load test dataset

test['Boy'] = (test.Name.str.split().str[1] == 'Master.').astype('int')

test['Family'] = test['SibSp'] + test['Parch']

submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': pd.Series(dtype='int32')})



# Three lines of code for LB = 0.79425 (not less 85% teams - Titanic Top 15%) 

# Reasoning the statements see below (EDA)

# Statement 1

test['Survived'] = [1 if (x == 'female') else 0 for x in test['Sex']]

# Statement 2

test.loc[(test.Boy == 1), 'Survived'] = 1

# Statements 3,4

test.loc[((test.Pclass == 3) & (test.Embarked == 'S') & ~((test.Boy == 1) & (test.Family > 0) & (test.Family < 4))), 'Survived'] = 0



# Saving the result

submission.Survived = test.Survived

submission.to_csv('submission_S_Boy_Sex_Family.csv', index=False)
# Reasoning for Statement 1 

# Thanks for the idea to: https://www.kaggle.com/mylesoneill/tutorial-1-gender-based-model-0-76555 

# Thanks for the idea of plot to: https://www.kaggle.com/pavlofesenko/simplest-top-10-titanic-0-80861

import matplotlib.pyplot as plt



def highlight(value):

    if value >= 0.5:

        style = 'background-color: palegreen'

    else:

        style = 'background-color: pink'

    return style



train = pd.read_csv('../input/titanic/train.csv') # load train dataset

pd.pivot_table(train, values='Survived', index=['Sex']).style.applymap(highlight)
# Reasoning for Statement 2

# Thanks for the plot to: https://www.kaggle.com/pavlofesenko/simplest-top-10-titanic-0-80861

train['Boy'] = (train.Name.str.split().str[1] == 'Master.').astype('int')

pd.pivot_table(train, values='Survived', index='Pclass', columns='Boy').style.applymap(highlight)
# Reasoning for Statement 3

# Thanks for the plot to: https://www.kaggle.com/pavlofesenko/simplest-top-10-titanic-0-80861

pd.pivot_table(train, values='Survived', index=['Pclass', 'Embarked'], columns='Sex').style.applymap(highlight)
# Reasoning for Statement 4

train['Family'] = train['SibSp'] + train['Parch']

pd.pivot_table(train, values='Survived', index=['Pclass','Embarked','Boy','Family']).style.applymap(highlight)
train
# Statements 1,2,3,4 in 4 lines of code:

test['Survived'] = [1 if (x == 'female') else 0 for x in test['Sex']]

test.loc[(test.Boy == 1), 'Survived'] = 1

test.loc[((test.Pclass == 3) & (test.Embarked == 'S')), 'Survived'] = 0

test.loc[((test.Pclass == 3) & (test.Embarked == 'S') & (test.Boy == 1) & (test.Family > 0) & (test.Family < 4)), 'Survived'] = 1
# Statements 1,2,3,4 in 3 lines of code (see above):

test['Survived'] = [1 if (x == 'female') else 0 for x in test['Sex']]

test.loc[(test.Boy == 1), 'Survived'] = 1

test.loc[((test.Pclass == 3) & (test.Embarked == 'S') & ~((test.Boy == 1) & (test.Family > 0) & (test.Family < 4))), 'Survived'] = 0