# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import seaborn as sb

import matplotlib.pyplot as plt



%matplotlib inline



train_set = pd.read_csv('../input/train.csv')

test_set = pd.read_csv('../input/test.csv')



grp_dt1 = train_set.groupby(pd.cut(train_set['Age'], np.arange(0, 90, 10))).agg(['mean'])['Survived']

grp_dt2 = train_set.groupby('Sex').agg(['mean'])['Survived']

grp_dt3 = train_set.groupby(pd.cut(train_set['Fare'], np.arange(0, 200, 50))).agg(['mean'])['Survived']



fig = plt.figure()

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharey=True, figsize=(12,8))



sb.set_style("whitegrid")

sb.barplot(grp_dt1.index, grp_dt1['mean'], ax=ax1)

sb.barplot(grp_dt2.index, grp_dt2['mean'], ax=ax2)

sb.barplot(grp_dt3.index, grp_dt3['mean'], ax=ax3)
from sklearn import preprocessing

le = preprocessing.LabelEncoder()



def clean_data(data):

    data = data.set_index('PassengerId').copy()

    data['LastName'] = [name[:name.index(',')] for name in data['Name']]

    data['Sex'] = le.fit_transform(data['Sex'])

    data['LastName'] = le.fit_transform(data['LastName'])

    data = data.drop(['Ticket', 'Cabin', 'Name', 'Embarked','Parch'], axis=1)

    data['Age'][np.isnan(data['Age'])] = data['Age'].median()

    

    medians = data.groupby('Pclass').median()



    for i, row in data[np.isnan(data['Fare'])].iterrows(): 

        if np.isnan(row['Fare']): data.loc[i, 'Fare'] = medians.loc[row['Pclass']]['Fare']



    return data



clean_data(train_set).head()
from sklearn.ensemble import RandomForestClassifier

from sklearn import cross_validation



class predict(object):

    def train(self, dt):

        self.x = dt.drop('Survived', axis=1)

        self.y = dt['Survived']



        self.rf =  RandomForestClassifier(max_depth=6)

        self.rf.fit(self.x, self.y)

      

    def train_score(self):

        score = cross_validation.cross_val_score(self.rf, self.x, self.y, cv=25)

        return('{0:.3f} +/- {1:.3f}'.format(score.mean(), score.std()))

    

    def test(self, data):

        dt = pd.DataFrame()

        dt['PassengerId'] = data.index

        dt['Survived'] = self.rf.predict(data)

        return dt



pdct = predict()

pdct.train(clean_data(train_set))

pdct.train_score()
pdct.test(clean_data(test_set)).set_index('PassengerId')