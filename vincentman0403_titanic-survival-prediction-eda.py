import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# turn off warning: SettingWithCopyWarning
pd.set_option('chained_assignment', None)
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
print('train data set, shape: ', train.shape)
print('test data set, shape: ', test.shape)
dataset =  pd.concat(objs=[train, test], axis=0, sort=False).reset_index(drop=True)
print('total data set, shape: ', dataset.shape)
dataset.head(5)
dataset.tail(5)
train.info()
train.describe()
train.describe(include=['O'])
dataset.isnull().sum()
train.isnull().sum()
# The distribution of Survived
sns.countplot(dataset['Survived'])
# The distribution of Embarked
sns.countplot(dataset['Embarked'])
# The distribution of Survived on Pclass
sns.countplot(dataset['Pclass'], hue=dataset['Survived'])
# The distribution of Survived on Sex
sns.countplot(dataset['Sex'], hue=dataset['Survived'])
# Survival probability on Sex
g = sns.barplot(x="Sex",y="Survived",data=train)
g = g.set_ylabel("Survival Probability")
train[["Sex","Survived"]].groupby('Sex').mean()
train[["Sex","Survived"]].groupby('Sex').std()
# The distribution of Survived on Embarked
sns.countplot(dataset['Embarked'], hue=dataset['Survived'])
#Fill Embarked nan values of dataset set with 'S' most frequent value
dataset["Embarked"] = dataset["Embarked"].fillna("S")
# Survival probability on Embarked
g = sns.factorplot(x="Embarked", y="Survived",  data=train,
                   size=6, kind="bar", palette="muted")
g = g.set_ylabels("survival probability")
# The distribution of Age on Survived 0 or 1: distplot
g = sns.FacetGrid(dataset, col='Survived')
g.map(sns.distplot, 'Age', kde=False)
# The distribution of Age on Survived 0 or 1: kdeplot
g = sns.kdeplot(train["Age"][(train["Survived"] == 0) & (train["Age"].notnull())], color="Red", shade = True)
g = sns.kdeplot(train["Age"][(train["Survived"] == 1) & (train["Age"].notnull())], ax =g, color="Blue", shade= True)
g.set_xlabel("Age")
g.set_ylabel("Frequency")
g = g.legend(["Not Survived","Survived"])
# The distribution of Fare on Survived 0 or 1 
g = sns.FacetGrid(dataset, col='Survived')
g.map(sns.distplot, 'Fare', kde=False)
#Fill Fare missing values with the median value
dataset["Fare"] = dataset["Fare"].fillna(dataset["Fare"].median())
# Explore Fare distribution 
g = sns.distplot(dataset["Fare"], color="m", label="Skewness : %.2f"%(dataset["Fare"].skew()))
g = g.legend(loc="best")
# Apply log to Fare to reduce skewness distribution
dataset["Fare"] = dataset["Fare"].map(lambda i: np.log(i) if i > 0 else 0)
# Explore Fare distribution after log
g = sns.distplot(dataset["Fare"], color="b", label="Skewness : %.2f"%(dataset["Fare"].skew()))
g = g.legend(loc="best")
# The distribution of SibSp on Survived 0 or 1 
g = sns.FacetGrid(dataset, col='Survived')
g.map(sns.distplot, 'SibSp', kde=False)
# Survival probability on SibSp 
g = sns.factorplot(x="SibSp",y="Survived",data=train,kind="bar", size = 6 , palette = "muted")
g = g.set_ylabels("survival probability")
# The distribution of Parch on Survived 0 or 1 
g = sns.FacetGrid(dataset, col='Survived')
g.map(sns.distplot, 'Parch', kde=False)
# Survival probability on Parch
g  = sns.factorplot(x="Parch",y="Survived",data=train,kind="bar", size = 6 ,palette = "muted")
g = g.set_ylabels("survival probability")
dataset['Family_Size'] = dataset['Parch'] + dataset['SibSp']
# The distribution of Family_Size on Survived 0 or 1 
g = sns.FacetGrid(dataset, col='Survived')
g.map(sns.distplot, 'Family_Size', kde=False)
# Survival probability on Pclass by Sex
g = sns.factorplot(x="Pclass", y="Survived", hue="Sex", data=train,
                   size=6, kind="bar", palette="muted")
g = g.set_ylabels("survival probability")
train[["Pclass", "Sex","Survived"]].groupby(["Pclass", "Sex"]).mean()
# The distribution of Pclass on Embarked
g = sns.factorplot("Pclass", col="Embarked",  data=train,
                   size=6, kind="count", palette="muted")
g = g.set_ylabels("Count")
# Survival probability on Pclass by Sex, Embarked
g = sns.factorplot(x="Pclass", y="Survived", col="Embarked", hue="Sex", data=train,
                   size=6, kind="bar", palette="muted")
g = g.set_ylabels("survival probability")
g = sns.heatmap(train[["Survived","SibSp","Parch","Age","Fare"]].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")
# Explore Age vs Sex, Parch , Pclass and SibSP
g = sns.factorplot(y="Age",x="Sex",data=dataset,kind="box", size=6)
g = sns.factorplot(y="Age",x="Sex",hue="Pclass", data=dataset,kind="box", size=6)
g = sns.factorplot(y="Age",x="Parch", data=dataset,kind="box", size=6)
g = sns.factorplot(y="Age",x="SibSp", data=dataset,kind="box", size=6)
# convert Sex into categorical value 0 for male and 1 for female
dataset["Sex"] = dataset["Sex"].map({"male": 0, "female":1})
g = sns.heatmap(dataset[["Age","Sex","SibSp","Parch","Pclass"]].corr(),cmap="BrBG",annot=True)
# In order to fill Age null value, I pick out samples whose Age value is null. 
# Then I pick out samples(Samples_A) whose SibSp, Parch, Pclass values are the same as these values of samples whose Age value is null.
# Finally I use median of Age value of Samples_A to fill Age null value.
index_NaN_age = list(dataset["Age"][dataset["Age"].isnull()].index)
for i in index_NaN_age:
    age_med = dataset["Age"].median()
    age_pred = dataset["Age"][(
            (dataset['SibSp'] == dataset.iloc[i]["SibSp"]) & (
            dataset['Parch'] == dataset.iloc[i]["Parch"]) & (
                    dataset['Pclass'] == dataset.iloc[i]["Pclass"]))].median()
    if not np.isnan(age_pred):
        dataset['Age'].iloc[i] = age_pred
    else:
        dataset['Age'].iloc[i] = age_med
# Explore Age vs Survived after filling NA
g = sns.factorplot(x="Survived", y = "Age",data = dataset, kind="box")
g = sns.factorplot(x="Survived", y = "Age",data = dataset, kind="violin")
dataset[["Survived","Age"]].groupby('Survived').median()
dataset["Cabin"].unique()
# Replace the Cabin number by the type of cabin 'X' if not
dataset["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in dataset['Cabin'] ])
dataset["Cabin"].unique()
g = sns.countplot(dataset["Cabin"],order=['A','B','C','D','E','F','G','T','X'])

train[["Survived","Age"]].groupby('Survived').median()
# Explore Cabin vs Survived after filling NA
g = sns.factorplot(y="Survived",x="Cabin",data=dataset,kind="bar",order=['A','B','C','D','E','F','G','T','X'])
g = g.set_ylabels("Survival Probability")
#Fill Fare missing values with the median value
dataset["Fare"] = dataset["Fare"].fillna(dataset["Fare"].median())
# Explore Fare distribution before log transformation
g = sns.distplot(dataset["Fare"], color="m", label="Skewness : %.2f"%(dataset["Fare"].skew()))
g = g.legend(loc="best")
# Apply log to Fare to reduce skewness distribution
dataset["Fare"] = dataset["Fare"].map(lambda i: np.log(i) if i > 0 else 0)
# Explore Fare distribution after log transformation
g = sns.distplot(dataset["Fare"], color="b", label="Skewness : %.2f"%(dataset["Fare"].skew()))
g = g.legend(loc="best")
dataset["Name"].head()
# Get Title from Name
dataset_title = [i.split(",")[1].split(".")[0].strip() for i in dataset["Name"]]
dataset["Title"] = pd.Series(dataset_title)
dataset["Title"].unique()
# The distribution of Title
g = sns.countplot(x="Title",data=dataset)
g = plt.setp(g.get_xticklabels(), rotation=45)
# Convert to categorical values Title 
dataset["Title"] = dataset["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
dataset["Title"] = dataset["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})
dataset["Title"] = dataset["Title"].astype(int)
# The distribution of Title: less classes
g = sns.countplot(dataset["Title"])
g = g.set_xticklabels(["Master","Miss/Ms/Mme/Mlle/Mrs","Mr","Rare"])
# Explore Title vs Survived
g = sns.factorplot(x="Title",y="Survived",data=dataset,kind="bar")
g = g.set_xticklabels(["Master","Miss-Mrs","Mr","Rare"])
g = g.set_ylabels("survival probability")
dataset["Fsize"] = dataset["SibSp"] + dataset["Parch"] + 1
# Explore Fize vs Survived
g = sns.factorplot(x="Fsize",y="Survived",data = dataset, kind='bar')
g = g.set_ylabels("Survival Probability")
# Create new feature of family size
dataset['Single'] = dataset['Fsize'].map(lambda s: 1 if s == 1 else 0)
dataset['SmallF'] = dataset['Fsize'].map(lambda s: 1 if  s == 2  else 0)
dataset['MedF'] = dataset['Fsize'].map(lambda s: 1 if 3 <= s <= 4 else 0)
dataset['LargeF'] = dataset['Fsize'].map(lambda s: 1 if s >= 5 else 0)
# Explore new feature vs Survived
g = sns.factorplot(x="Single",y="Survived",data=dataset,kind="bar")
g = g.set_ylabels("Survival Probability")
g = sns.factorplot(x="SmallF",y="Survived",data=dataset,kind="bar")
g = g.set_ylabels("Survival Probability")
g = sns.factorplot(x="MedF",y="Survived",data=dataset,kind="bar")
g = g.set_ylabels("Survival Probability")
g = sns.factorplot(x="LargeF",y="Survived",data=dataset,kind="bar")
g = g.set_ylabels("Survival Probability")
dataset["Ticket"].head()
## Treat Ticket by extracting the ticket prefix. When there is no prefix it returns X. 

Ticket = []
for i in list(dataset.Ticket):
    if not i.isdigit() :
        Ticket.append(i.replace(".","").replace("/","").strip().split(' ')[0]) #Take prefix
    else:
        Ticket.append("X")
        
dataset["Ticket"] = Ticket
dataset["Ticket"].head()
dataset["Ticket"].describe()
dataset["Ticket"].value_counts()
