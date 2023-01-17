# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid")



import seaborn as sns

from collections import Counter



import warnings

warnings.filterwarnings("ignore")



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df=pd.read_csv('/kaggle/input/titanic/train.csv')

test_df=pd.read_csv('/kaggle/input/titanic/test.csv')

test_pass_id=test_df["PassengerId"] # for storing the first version of the passenger id for future usage.

train_df.info
train_df.head() # for first 5 records
train_df.describe() # statistical info for numeric columns
train_df[0:30]
train_df.info()
def bar_plot(variable): # method

    

   # ***

   #     input=variable ex:"sex"

   #     output= bar plot

   # ***

    

    # get feature

    var=train_df[variable]

    

    #count number of categorical variable

    varValue=var.value_counts()

    

    #visualize

    plt.figure(figsize=(9,3))

    plt.bar(varValue.index, varValue)

    plt.xticks(varValue.index,varValue.index.values)

    plt.ylabel("Frequency")

    plt.title(variable)

    plt.show()

    print("{}: \n {}".format(variable,varValue))

    
category1={"Survived", "Sex", "Pclass", "Embarked", "SibSp","Parch"}



for c in category1:

    bar_plot(c)
category2={"Cabin", "Name", "Ticket"} # splitted because of complex structures of the variables



for c in category2:

    print ("{}\n", format(train_df[c].value_counts()))









def plot_hist(variable):

    plt.figure(figsize=(9,3))

    plt.hist(train_df[variable],bins=50) # bin changes the bar frequency

    plt.xlabel(variable)

    plt.ylabel("Frequency")

    plt.title(variable)

    plt.show()
numericVar={"Fare", "Age", "PassengerId"}

for n in numericVar:

    plot_hist(n)
train_df[["Pclass", "Survived"]]

# Pclass - Survived



train_df[["Pclass", "Survived"]].groupby(["Pclass"],as_index=False).mean()
train_df[["Pclass", "Survived"]].groupby(["Pclass"],as_index=False).mean().sort_values(by="Survived", ascending=False)
train_df[["Sex", "Survived"]].groupby(["Sex"],as_index=False).mean().sort_values(by="Survived", ascending=False)
train_df[["SibSp", "Survived"]].groupby(["SibSp"],as_index=False).mean().sort_values(by="Survived", ascending=False)
train_df[["Parch", "Survived"]].groupby(["Parch"],as_index=False).mean().sort_values(by="Survived", ascending=False)
def detect_outlier(df,features):

    

    outlier_indices=[]

    

    for c in features:

        # first quartile

        Q1= np.percentile(df[c],25)

        

        # third qurtile

        Q3= np.percentile(df[c],75)

        

        # IQO

        IQR=Q3-Q1

        

        # outlier step

        out_step=IQR * 1.5

                

        # outlier detection 

        outlier_list_c=df[(df[c] < Q1-out_step) | (df[c] > Q3+out_step)].index

                

        #store the indeces

        outlier_indices.extend(outlier_list_c)

        

    outlier_indices=Counter(outlier_indices)

    

    multiple_outliers=list(i for i, v in outlier_indices.items() if v > 2)

                

    return multiple_outliers

    
train_df.loc[detect_outlier(train_df, ["Age","SibSp","Parch","Fare"])]
# drop outlieers



train_df=train_df.drop(detect_outlier(train_df, ["Age","SibSp","Parch","Fare"]),axis=0).reset_index(drop=True) 
train_df_len=len(train_df)

train_df_merged=pd.concat([train_df,test_df],axis=0).reset_index(drop=True)
train_df_merged.info
train_df.info
train_df_merged.head()

train_df_merged.columns[train_df.isnull().any()]
train_df_merged.isnull().sum()

train_df_merged[train_df_merged["Embarked"].isnull()]
# embarked can be filled by taken into account the fare variable which passengers have same fare rates and their embarked vlue can be filled with their values.



train_df_merged.boxplot(column="Fare",by="Embarked")

plt.show()
train_df_merged["Embarked"]=train_df_merged["Embarked"].fillna("C")
train_df_merged[train_df_merged["Embarked"].isnull()]

train_df_merged[train_df_merged["Fare"].isnull()]
train_df_merged[train_df_merged["Pclass"]==3]
np.mean(train_df_merged[train_df_merged["Pclass"]==3]["Fare"])
train_df_merged["Fare"]=train_df_merged["Fare"].fillna(np.mean(train_df_merged[train_df_merged["Pclass"]==3]["Fare"]))
train_df_merged[train_df_merged["Fare"].isnull()]

train_df_merged.info
list1=["SibSp","Parch","Age","Fare","Survived"]

sns.heatmap(train_df_merged[list1].corr(),annot=True,fmt=".2f")

plt.show()
g=sns.factorplot(x="SibSp", y="Survived",data=train_df_merged, kind="bar",size=6)

g.set_ylabels("Survival Probability")

plt.show()
g=sns.factorplot(x="Parch", y="Survived",data=train_df_merged, kind="bar",size=6)

g.set_ylabels("Survival Probability")

plt.show()
g=sns.factorplot(x="Pclass", y="Survived",data=train_df_merged, kind="bar",size=5)

g.set_ylabels("Survival Probability")

plt.show()
g=sns.FacetGrid(train_df_merged,col="Survived")

g.map(sns.distplot,"Age",bins=25)

plt.show()
g=sns.FacetGrid(train_df_merged,col="Survived",row="Pclass")

g.map(plt.hist,"Age",bins=25)

g.add_legend()

plt.show()
g=sns.FacetGrid(train_df_merged,row="Embarked", size=3)

g.map(sns.pointplot,"Pclass","Survived", "Sex")

g.add_legend()

plt.show()
g=sns.FacetGrid(train_df_merged,row="Embarked", col="Survived",size=2.5)

g.map(sns.barplot,"Sex","Fare")

g.add_legend()

plt.show()
train_df_merged[train_df_merged["Age"].isnull()]
sns.factorplot(x="Sex",y="Age",data=train_df_merged,kind="box")

plt.show()
sns.factorplot(x="Sex",y="Age",hue="Pclass",data=train_df_merged,kind="box")

#hue: for different class analysis

plt.show()
sns.factorplot(x="Parch",y="Age",data=train_df_merged,kind="box")

sns.factorplot(x="SibSp",y="Age",data=train_df_merged,kind="box")

plt.show()
#sex feature sayısala çevrilmeli

train_df_merged["Sex"]=[1 if i=="male" else 0 for i in train_df_merged["Sex"]]



#tüm feature lar arasında bir korelasyon var mı?



sns.heatmap(train_df[["Age","Sex","SibSp","Parch", "Pclass"]].corr(),annot=True)

plt.show()
index_nan_age=list(train_df_merged["Age"][train_df_merged["Age"].isnull()].index)

# age boş olanları bul, indeksini yaz ve bunu listeye yaz



for i in index_nan_age:

    

    age_pred=train_df_merged["Age"][((train_df_merged["SibSp"]==train_df_merged.iloc[i]["SibSp"]) & (train_df_merged["Parch"]==train_df_merged.iloc[i]["Parch"]) & (train_df_merged["Pclass"]==train_df_merged.iloc[i]["Pclass"]))].median()

   #  dfnin sibsp sini al, i.indekse git bunun sibsp sini al. ve aynısını parch ve pclass içinde yap.hepsinin medianını bul. 

    

    age_med=train_df_merged["Age"].median()

    

    if not np.isnan (age_pred): # age_pred boş değilse

    

        train_df_merged["Age"].iloc[i]=age_pred

        

    else: # boşsa 

            

            train_df_merged["Age"].iloc[i]=age_med



    
age_pred

index_nan_age
age_med
train_df_merged[train_df_merged["Age"].isnull()]
train_df_merged.info
train_df_len
train_df.head()

        
train_df_merged["Fsize"]=train_df_merged["SibSp"]+train_df_merged["Parch"]+1 

# toplamı 0 olmasın diye +1 eklendi. 1 birey olarak aile tanımlanacak
train_df_merged.head()

#fsize ile survived arasındaki ilişki incelenecek



g=sns.factorplot(x="Fsize",y="Survived", data=train_df_merged,kind="bar")

g.set_ylabels("Survival")

plt.show()
# 2 categories can be done with 4.5



train_df_merged["Family_size"]=[1 if i<5 else 0 for i in train_df_merged["Fsize"]]
train_df_merged.head(5)
sns.countplot(x="Family_size",data=train_df_merged)

plt.show()
g=sns.factorplot(x="Family_size",y="Survived", data=train_df_merged,kind="bar")

g.set_ylabels("Survival")

plt.show()
train_df_merged.head(25)



train_df_merged=pd.get_dummies(train_df_merged,columns=["Family_size"])

train_df_merged.head()
train_df_merged["Embarked"].head()
sns.countplot(train_df_merged["Embarked"])

plt.show()
train_df_merged=pd.get_dummies(train_df_merged,columns=["Embarked"])

train_df_merged.head()
train_df_merged["Ticket"].head(12)
a="A/5 21171"

a.replace(".","")

a.replace("/","").strip().split(" ")[0] # ekstra başta ve sondaki boşlukları alıyor



tickets=[]

for i in list(train_df_merged.Ticket):



    if not i.isdigit():

            tickets.append(i.replace(".","").replace("/","").strip().split(" ")[0])

            

    else: #başında hibir şey yoksa

        

        tickets.append("X")

        

train_df_merged["Tickets"] =tickets
train_df_merged["Tickets"].head()
train_df_merged.head()
train_df_merged=pd.get_dummies(train_df_merged,columns=["Tickets"],prefix="T")

train_df_merged.head()

# ticket ismi yerine T ile diğer featurelara isim veriyor
sns.countplot(x="Pclass",data=train_df_merged)

plt.show()
train_df_merged["Pclass"]=train_df_merged["Pclass"].astype("category")

train_df_merged=pd.get_dummies(train_df_merged,columns=["Pclass"])

train_df_merged.head()
train_df_merged["Sex"].head()
train_df_merged["Sex"]=train_df_merged["Sex"].astype("category")

train_df_merged=pd.get_dummies(train_df_merged,columns=["Sex"])

train_df_merged.head()
train_df_merged.drop(labels=["PassengerId","Cabin"],axis=1,inplace=True)
train_df_merged.columns
train_df.info
# import libraries for training the dataset



from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
train_df_merged.drop(labels=["Ticket"],axis=1,inplace=True)
train_df_merged.info
train_df_len
test=train_df_merged[train_df_len:]

test.drop(labels="Survived", axis=1, inplace=True)
test

test.drop(labels="Name", axis=1, inplace=True)
train_df_merged.drop(labels="Name", axis=1, inplace=True)
test

train=train_df_merged[:train_df_len]

x_train=train.drop(labels="Survived",axis=1)

y_train=train["Survived"]

x_train,x_test,y_train,y_test=train_test_split(x_train,y_train,test_size=0.30, random_state=42)

print("x_train", len(x_train))

print("x_test", len(x_test))

print("y_train", len(y_train))

print("y_test", len(y_test))

print("test", len(test))







train_df_merged
logreg=LogisticRegression()

logreg.fit(x_train,y_train)

acc_log_train=round(logreg.score(x_train,y_train)*100,2)

acc_log_test=round(logreg.score(x_test,y_test)*100,2)

print("Training Accuracy: % {}",format(acc_log_train))

print("Testing Accuracy: % {}",format(acc_log_test))



random_state=42

classifier=[DecisionTreeClassifier(random_state=random_state),

           SVC(random_state=random_state),

           RandomForestClassifier(random_state=random_state),

           LogisticRegression(random_state=random_state),

           KNeighborsClassifier()]



DT_param_grid={"min_samples_split" : range(10,500,20),

              "max_depth": range(1,20,2)}



SVC_param_grid={"kernel" : ["rbf"],

               "gamma" : [0.001,0.01,0.1,1],

               "C" : [1,10,100,200,300,1000]}



RF_param_grid={"max_features" : [1,3,10],

              "min_samples_split": [2,3,10],

              "min_samples_leaf" : [1,3,10],

              "bootstrap" : [False],

              "n_estimators": [100,300],

              "criterion": ["gini"]}



LR_param_grid={"C" : np.logspace(-3,3,7),

              "penalty": ["l1","l2"]}



KNN_param_grid={"n_neighbors" : np.linspace(1,19,10, dtype=int).tolist(),

               "weights": ["uniform","distance"],

               "metric": ["minkowski","manhattan"]}



classifier_param=[DT_param_grid,

                  SVC_param_grid,

                  RF_param_grid,

                  LR_param_grid,

                  KNN_param_grid]



                 

Cross_Val_Res=[]

Best_estimator=[]



for i in range(len(classifier)):

    

    clf = GridSearchCV(classifier[i],param_grid=classifier_param[i],cv=StratifiedKFold(n_splits=10), scoring="accuracy", n_jobs=-1,verbose=1)

    # n_jobs parametrenin daha hızlı çalışması için -1 yapılıyor, verbose ise sonuçları bize sürekli gösterecek

    clf.fit(x_train,y_train)

    Cross_Val_Res.append(clf.best_score_)

    Best_estimator.append(clf.best_estimator_)

    print(Cross_Val_Res[i])

Cross_Val_Res_2=pd.DataFrame({"Cross Validation means":Cross_Val_Res,"ML Models" : ["DecisionTreeClassifier","SVC","RandomForestClassifier","LogisticRegression","KNeighborsClassifier"]})



g=sns.barplot("Cross Validation means","ML Models", data=Cross_Val_Res_2)

g.set_xlabel("Mean Accuracy")

g.set_title("Cross Validation Scores")

voting_classifier=VotingClassifier(estimators=[("DT", Best_estimator[0]),

                                                ("RF", Best_estimator[2])],

                                                 voting="soft",n_jobs=-1)



voting_classifier=voting_classifier.fit(x_train,y_train)

print(accuracy_score(voting_classifier.predict(x_test),y_test))



test_survived=pd.Series(voting_classifier.predict(test), name="Survived").astype(int)

results=pd.concat([test_pass_id,test_survived],axis=1)

results.to_csv("Titanic.csv",index=False)