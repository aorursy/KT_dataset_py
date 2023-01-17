#WORK LEFT:

#Figure out how to judge importance of features

#Apply basic models

#Change both test and training sets at the same time!
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

pd.set_option('display.float_format',lambda x: "%.3f" % x)

pd.set_option('display.max_rows',1000)

pd.set_option('display.max_columns',1000)

plt.rcParams['figure.figsize'] = (11,5)
train_df = pd.read_csv("/kaggle/input/titanic/train.csv",index_col=False)

train_df.head(20)
train_df.describe()
train_df.isna().sum()
train_df.Age.fillna(train_df.groupby(['Pclass','Sex']).Age.transform('median'),inplace=True)

train_df.Embarked.fillna(train_df.Embarked.mode()[0],inplace=True)
train_df.isna().sum()
# z = pd.crosstab(train_df.Cabin.apply(lambda x: x[:1] if type(x).__name__=='str' else np.nan),train_df.Survived)

# cabin_mapping = z.div(z.sum(axis=1),axis=0).sort_values(by=1)[1].to_dict()

# cabin_mapping[np.nan] = 0

#train_df['CabinCode'] = train_df.Cabin.map(cabin_mapping)



#y = pd.crosstab(train_df.Embarked,train_df.Survived)

#z = y.div(y.sum(axis=1),axis=0)

#(z/z.max(axis=0))*3



title_mapping = {

    'Capt':1,

    'Col':5,

    'Don':1,

    'Dr':5,

    'Jonkheer':1,

    'Lady':5,

    'Major':5,

    'Master':3,

    'Miss':3,

    'Mlle':5,

    'Mme':5,

    'Mr':2,

    'Mrs':4,

    'Ms':5,

    'Rev':1,

    'Sir':5,

    'the Countess':5

}

train_df['TitleCode'] = train_df.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip()).map(title_mapping)

train_df['GenderCode'] = train_df.Sex.map({'male':0,'female':1})



train_df['FamilySize'] = train_df.SibSp + train_df.Parch + 1

train_df['isAlone'] = train_df.FamilySize==1

train_df['hasCabin'] = ~train_df.Cabin.isna()

train_df['Embarked'] = train_df.Embarked.map({'C':3.0,'Q':2.1,'S':1.8}).astype(np.float64)

train_df['AgeBin'] = pd.cut(train_df.Age,bins=8).cat.codes

train_df['FareBin'] = pd.cut(train_df.Fare,bins=15).cat.codes

train_df['isKid'] = train_df.Age<=13.0
def plot_feature(train_df,feature):

    survived_df = train_df[train_df.Survived==True][feature].value_counts()

    dead_df = train_df[train_df.Survived==False][feature].value_counts()

    counts_df = pd.DataFrame([survived_df,dead_df],index=['Survived','Dead'])

    counts_df.plot(kind='barh',stacked=True,legend=True,title='Frequency by '+feature)

    (counts_df.div(counts_df.sum(axis=1),axis=0)*100).plot(kind='barh',stacked=True,legend=True,title='Proportion by '+feature)

    

def plot_kde(train_df,split_feature,kde_feature):

    legend = sorted(train_df.dropna(subset=[split_feature])[split_feature].unique())

    for key in legend:

        sns.kdeplot(train_df[train_df[split_feature]==key][kde_feature],shade=True)#.plot(kind='kde',legend=True)

    plt.legend(tuple(legend))
plt.rcParams['figure.figsize'] = (13,10)

sns.heatmap(train_df[['Survived','Pclass','Fare','Age','GenderCode','hasCabin','TitleCode','isAlone','isKid','Embarked','AgeBin','FareBin']].corr(),annot=True);
plt.rcParams['figure.figsize'] = (9,6)

plot_feature(train_df,'Sex')
plot_feature(train_df,'Pclass')
plot_feature(train_df,'Embarked')
plot_feature(train_df,'FamilySize')
plt.clf()

plot_kde(train_df,'Pclass','Fare')

plt.show()

plt.clf()

plot_kde(train_df,'Pclass','Age')

plt.show()
plt.rcParams['figure.figsize'] = (11,5)

plt.clf()

plot_kde(train_df,'Embarked','Fare')

plt.show()

# plt.clf()

# plot_kde(train_df,'Embarked','Age')

# plt.show()

plt.rcParams['figure.figsize'] = (11,5)

plt.clf()

plot_kde(train_df,'Sex','Fare')

plt.show()

plt.clf()

plot_kde(train_df,'Sex','Age')

plt.show()
plt.rcParams['figure.figsize'] = (12,8)

survived_df = train_df[train_df.Survived == True]

dead_df = train_df[train_df.Survived == False]

sns.scatterplot(train_df.Age,train_df.Fare,hue=train_df.Survived,size=train_df.Fare,palette={0:'red',1:'green'});
sns.violinplot(train_df.Sex,train_df.Age,hue=train_df.Survived,split=True,palette={0:'r',1:'g'});
train_df['GenderAndClass'] = list(zip(train_df.Sex,train_df.Pclass))

train_df.sort_values(by='GenderAndClass',inplace=True);

sns.violinplot(train_df.GenderAndClass,train_df.Age,hue=train_df.Survived,split=True,palette={0:'r',1:'g'});
sns.violinplot(train_df.Embarked,train_df.Fare, hue=train_df.Survived, split=True, palette={0: "r", 1: "g"});
plt.rcParams['figure.figsize'] = (9,6)

train_df.groupby('Pclass').mean()[['Fare','Age']].plot(kind='bar',title='Mean Age and Fare by Class');
plt.hist([train_df[(train_df.Survived==False)]['Fare'],train_df[(train_df.Survived==True)]['Fare']],color=['r','g'],label=['Dead','Survived'],stacked=True,bins=30,)

plt.title('Ticket Fare Histogram')

plt.xlabel('Fare')

plt.ylabel('Frequency')

plt.legend();
plt.hist([train_df[train_df.Survived==False].Age,train_df[train_df.Survived==True].Age],color=['r','g'],stacked=True,label=['Dead','Survived'],bins=20)

plt.title('Age Histogram')

plt.xlabel('Age')

plt.ylabel('Frequency')

plt.legend();
plt.hist([train_df[(train_df.Survived==False) & (train_df.Sex=='male')].Age,

          train_df[(train_df.Survived==True) & (train_df.Sex=='male')].Age,

          train_df[(train_df.Survived==False) & (train_df.Sex=='female')].Age,

          train_df[(train_df.Survived==True) & (train_df.Sex=='female')].Age],

         color=['red','green','indianred','lime'],

         stacked=True,

         label=['Dead Male','Rescued Male','Dead Female','Rescued Female'],

         bins=20)

plt.title('Age Histogram')

plt.xlabel('Age')

plt.ylabel('Frequency')

plt.legend();