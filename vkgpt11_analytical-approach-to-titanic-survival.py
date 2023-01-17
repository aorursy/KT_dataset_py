import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
data_train = pd.read_csv("../input/train.csv")
data_train.head()
data_train.shape
data_train.dtypes
data_train.describe()
cat=data_train.dtypes[data_train.dtypes=="object"].index

print(cat)

data_train[cat].describe()
del data_train["PassengerId"]
del data_train["Name"] 



#We can also delete both together using below code 



#data_train.drop(["Cabin","PassengerId"],axis=1,inplace=True)
print(data_train["Ticket"][0:5])

print("------------------------")

print(data_train["Ticket"].describe())
del data_train["Ticket"]
data_train["Cabin"][0:10]
data_train["Cabin"].describe()
x=data_train["Cabin"].astype(str)

new_cabin = np.array([cabin[0] for cabin in x]) #get the first letter

new_cabin = pd.Categorical(new_cabin)

new_cabin.describe()
#Lets keep it 

data_train["Cabin"] = new_cabin
new_survived = pd.Categorical(data_train["Survived"])

new_survived = new_survived.rename_categories(["Died","Survived"])              

new_survived.describe()            
new_Pclass = pd.Categorical(data_train["Pclass"],ordered=True)

new_Pclass = new_Pclass.rename_categories(["High","Medium","Low"])     

new_Pclass.describe()
data_train["Pclass"] = new_Pclass
def find_missing(data):

    missing={}

    for i in range(data.shape[1]):

        totalNan = sum(pd.isnull(data[data.columns[i]]))

        if(totalNan>0):

            missing[data.columns[i]] =  totalNan

    return missing



print(find_missing(data_train))
data_train.groupby("Embarked").size()
data_train["Embarked"].mode()
data_train["Embarked"].fillna('S',inplace=True)

data_train.groupby("Embarked").size()
data_train.groupby("Sex").median()["Age"]
data_train.groupby("Sex").mean()["Age"]
data_train["Age"].fillna(data_train.groupby('Sex')["Age"].transform("median"),inplace=True)
data_train.groupby("Sex").mean()["Age"]
data_train["Fare"].plot(kind="box")
sns.boxplot(x="Pclass",y="Fare",data=data_train)
row_indexes = np.where(data_train["Fare"] == max(data_train["Fare"]))

data_train.loc[row_indexes]
data_train.groupby(["Embarked","Pclass"]).mean()["Fare"]
for i in row_indexes:

    data_train.set_value(i,"Fare",104)
data_train["Fare"].plot(kind="box")
row_indexes = np.where(data_train["Fare"] == max(data_train["Fare"]))

data_train.loc[row_indexes]
del data_train["Fare"]
data_train["Age"].plot(kind="box")
row_indexes = np.where(data_train["Age"] == min(data_train["Age"]))

print(data_train.loc[row_indexes])

print()

row_indexes = np.where(data_train["Age"] == max(data_train["Age"]))

print(data_train.loc[row_indexes])
data_train["Family"] = data_train["SibSp"] + data_train["Parch"]
large_family = np.where(data_train["Family"] == max(data_train["Family"]))

data_train.loc[large_family]
# Specify the parameter for our graphs 

fig = plt.figure(figsize=(21,8))





#Create the frames to plot different garphs 

plt.subplot2grid((1,3),(0,0))



data_train["Survived"].value_counts().plot(kind='bar',title="Distribution of Survival, (1 = Survived)",alpha=.7)

plt.grid(b=True,which='major', axis='both')

plt.title("Distribution of Survival, (1 = Survived)")



plt.subplot2grid((1,3),(0,1))

data_train["Pclass"].value_counts().plot(kind='bar',title="Distribution of Pclass",alpha=.3)

plt.grid(b=True,which='major', axis='both')



plt.subplot2grid((1,3),(0,2))

data_train["Sex"].value_counts().plot(kind='bar',title="Distribution of Sex",alpha=.7)

plt.grid(b=True,which='major', axis='both')
# Specify the parameter for our graphs 

fig = plt.figure(figsize=(21,8))



plt.subplot2grid((1,2),(0,0))

data_train["Embarked"].value_counts().plot(kind='bar',title="Distribution of Embarked",alpha=.7)

plt.grid(b=True,which='major', axis='both')





plt.subplot2grid((1,2),(0,1))

data_train["Age"].plot(kind='Hist',title="Age Distribution",alpha=.7)

plt.grid(b=True,which='major', axis='both')
#plots a kernal density estimate of the subset of the 1st class passangers's age

data_train.Age[data_train.Pclass=="High"].plot(kind='kde',color='black',label='1st class')

data_train.Age[data_train.Pclass=="Medium"].plot(kind='kde',color='blue',label='2nd class')

data_train.Age[data_train.Pclass=="Low"].plot(kind='kde',color='yellow',label='3rd class')

plt.legend(loc='best')

 # plots an axis lable

plt.xlabel("Age")    

plt.title("Age Distribution within classes")
survived_sex_xt = pd.crosstab(data_train['Sex'],data_train['Survived'])

ax = survived_sex_xt.plot(kind='bar',title='Survived vs Sex',alpha=0.5)

ax.set(xlabel="Sex",ylabel="Frequency")

plt.grid(b=True,which="major")
sns.barplot(x="Sex", y="Survived",hue="Pclass",palette="Greens_d", data=data_train)
sns.barplot(x="Pclass", y="Survived",hue="Sex",palette={"male": "g", "female": "m"},alpha=0.5, data=data_train)
fig = plt.figure(figsize=(18,4))

alpha_level = 0.50



ax1 = fig.add_subplot(141)

female_highclass = data_train.Survived[data_train.Sex == 'female'][data_train.Pclass!='Low'].value_counts().sort_index(

    ascending=False)

female_highclass.plot(kind='bar',label='female, highclass',color='gray',alpha=alpha_level)

ax1.set_xticklabels(['Survived','Died'],rotation=0)

ax1.set_xlim(-1,len(female_highclass))

ax1.set_ylim(0,250)

plt.title("Survival with respect to gender and class"); plt.legend(loc='best')

plt.grid(b=True,which='major', axis='both')







ax2 = fig.add_subplot(142)

female_lowclass = data_train.Survived[data_train.Sex=='female'][data_train.Pclass=='Low'].value_counts().sort_index(ascending=False)

female_lowclass.plot(kind='bar',label='female, lowclass',color='green', alpha=alpha_level)

ax2.set_xticklabels(['Survived','Died'],rotation=0)

ax2.set_xlim(-1, len(female_lowclass))

ax2.set_ylim(0,250)

plt.legend(loc='best')

plt.grid(b=True,which='major', axis='both')







ax3 = fig.add_subplot(143)

male_highclass = data_train.Survived[data_train.Sex == 'male'][data_train.Pclass!='Low'].value_counts().sort_index(ascending=False)

male_highclass.plot(kind='bar',label='male, highclass',color='blue',alpha=alpha_level)

ax3.set_xticklabels(['Survived','Died'],rotation=0)

ax3.set_xlim(-1,len(male_highclass))

plt.legend(loc='best')

ax3.set_ylim(0,250)

plt.grid(b=True,which='major', axis='both')







ax4 = fig.add_subplot(144)

male_lowclass = data_train.Survived[data_train.Sex=='male'][data_train.Pclass=="Low"].value_counts().sort_index(ascending=False)

male_lowclass.plot(kind='bar',label='male, lowclass',color='darkgray', alpha=alpha_level)

ax4.set_xticklabels(['Survived','Died'],rotation=0)

ax4.set_xlim(-1, len(female_lowclass))

ax4.set_ylim(0,250)

plt.legend(loc='best')

plt.grid(b=True,which='major', axis='both')

survived_Family_xt = pd.crosstab(data_train['Family'],data_train['Survived'])

ax = survived_Family_xt.plot(kind='bar',title='Survived vs Family',alpha=0.5)

ax.set(xlabel="Family",ylabel="Frequency")

plt.grid(b=True,which="major")
Family_Survived = data_train.groupby(["Family","Survived"]).size()

Family_Survived.groupby(level=0).apply(lambda x:100 * x / float(x.sum()))
from sklearn import linear_model

from sklearn import preprocessing

from sklearn.metrics import accuracy_score
def get_features(dataframe):

    data=dataframe.copy()

    #Initialize label encoder

    label_encoder = preprocessing.LabelEncoder()



    # Convert Sex variable to dummy variables

    encoded_sex = label_encoder.fit_transform(data["Sex"])

    encoded_class = label_encoder.fit_transform(data["Pclass"])

    encoded_cabin = label_encoder.fit_transform(data["Cabin"])

    encoded_Embarked = label_encoder.fit_transform(data["Embarked"])



    features = pd.DataFrame([encoded_class,

                                   encoded_sex,

                                   encoded_cabin,

                                   encoded_Embarked,

                                   data["Family"],

                                   data["Age"]]).T

    return features
logit_model = linear_model.LogisticRegression()

logit_model.fit(get_features(data_train) ,data_train["Survived"])

# Check trained model intercept

print(logit_model.intercept_)



# Check trained model coefficients

print(logit_model.coef_)
# Perfromance on training data

accuracy_score(data_train["Survived"],logit_model.predict(get_features(data_train)))
# lets clean and transform test data also

def clean_transform_test_data(data):

    data_test=data.copy()

    

    for key,value  in find_missing(data_test).items():

        if(key=="Age"):

            data_test[key].fillna(data_test.groupby('Sex')[key].transform("median"),inplace=True)

        elif(key=="Fare"):

            data_test[key].fillna(data_test.groupby('Pclass')[key].transform("median"),inplace=True)

        elif(key=="Embarked"):

            data_test[key].fillna(data_test[key].transform("mode"),inplace=True)

    

    x1=data_test["Cabin"].astype(str)

    new_cabin1 = np.array([cabin[0] for cabin in x1])

    new_cabin1 = pd.Categorical(new_cabin1)

    data_test["Cabin"] = new_cabin1

    

    new_pclass1 =pd.Categorical(data_test["Pclass"],ordered=True)

    new_Pclass11 = new_pclass1.rename_categories(["High","Medium","Low"])     

    data_test["Pclass"] = new_Pclass11

    

    data_test["Family"] = data_test["SibSp"] + data_test["Parch"]

    

    return data_test
data_test_1 = pd.read_csv("../input/test.csv")

test = clean_transform_test_data(data_test_1).copy()

test['Survived'] = 0



test_preds = logit_model.predict(X=get_features(test))





data_test_1["Survived"] =test_preds

submission=pd.DataFrame({"PassengerId":data_test_1["PassengerId"],

                           "Survived":data_test_1["Survived"]})



# Save submission to CSV

submission.to_csv("submission.csv", 

                  index=False)
sns.barplot(x="Pclass", y="Survived",hue="Sex",palette={"male": "g", "female": "m"},alpha=0.5, data=data_test_1)