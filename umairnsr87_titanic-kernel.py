import pandas as pd

import numpy as np

from collections import Counter
df_train=pd.read_csv("../input/titanic/train.csv")

df_test=pd.read_csv("../input//titanic/test.csv")
print("The shape of the training data is"+str(df_train.shape))

print("The shape of the test data is"+str(df_test.shape))

#lets see what out data contains

df_train.head(10)
df_train.info()
#lets check the number of missing values 

#Missing values in the data train is

df_train.isna().sum()
#Missing values in the data test is

df_test.isna().sum()
#we will be changing the types of column of the training and test dataset

df_train["Sex"]=np.where(df_train["Sex"]=="male",1,0)
#we will be changing the types of column of the training and test dataset

df_test["Sex"]=np.where(df_test["Sex"]=="male",1,0)
df_train["Title"]=df_train["Name"].str.extract('([A-Za-z]+)\.',expand=False)

df_test["Title"]=df_test["Name"].str.extract('([A-Za-z]+)\.',expand=False)
df_train["Title"].value_counts()
df_test["Title"].value_counts()
mapping={"Mr":0,"Miss":1,"Mrs":2,"Master":3,"Dr":3,"Rev":3,"Mlle":3,"Col":3,"Major":3,

         "Capt":3,"Sir":3,"Ms":3,"Jonkheer":3,"Mme":3,"Countess":3,

         "Don":3,"Lady":3}

for values in df_test:

    df_train["title"]=df_train["Title"].map(mapping)



    

for values in df_test:

    df_test["title"]=df_test["Title"].map(mapping)

    
df_train.head(5)
#now lets see the title relationship with the survived data

import seaborn as sns

my_colors = ['g']

import matplotlib.pyplot as plt

plt.figure(figsize=(10,8))

plt.hist(df_train["Survived"],color=my_colors)

plt.title("Title vs graph")

#by the means of this count plot we have seen that the survivors are in domination in female 

plt.figure(figsize=(15,6))

sns.countplot(df_train["Title"], hue=df_train['Survived'])
#lets drop the names now

df_train.drop(["Name"],axis=1,inplace=True)

df_test.drop(["Name"],axis=1,inplace=True)
x=df_train[df_train["Sex"]==0]

x=x[x["Survived"]==1]
y=df_train[df_train["Sex"]==1]

y=y[y["Survived"]==1]
#lets check the male and female survivors

sns.countplot(df_train["Sex"],hue=df_train["Survived"])

plt.xlabel("The survived females are"+str(x.shape[0]) +" and the total was"+str(df_train["Sex"].value_counts()[0]))

plt.ylabel("The survived males are"+str(y.shape[0]) +" and the total was"+str(df_train["Sex"].value_counts()[1]))



plt.title("Total number of females are"+str(df_train["Sex"].value_counts()[0])+" " +"males are"+str(df_train["Sex"].value_counts()[1]) )

#lets drop the outliers for now

sns.boxplot(df_train["Fare"])
#getting the quantiles for 25 and 75 percentile

#Quantile1=np.percentile(df_train["Fare"],25)

#Quantile3=np.percentile(df_train["Fare"],75)

#IQR

#IQR=Quantile3-Quantile1

#Outliers_steps=1.5*IQR

#Outlier_indices=df_train[df_train["Fare"]<(Quantile1-Outliers_steps)]

#Outlier_indices1=df_train[df_train["Fare"]>(Quantile1+Outliers_steps)]

#print(Outliers_steps)
def outlier_detection(dataframe_to_check,number_of_total_outliers,list_of_features):

    index_of_outliers=[]

    for col in list_of_features:

        Quantile1=np.percentile(dataframe_to_check[col],25)

        Quantile3=np.percentile(dataframe_to_check[col],75)

        IQR=Quantile3-Quantile1

        outlier_factor=1.5*IQR

        outlier_list = dataframe_to_check[(dataframe_to_check[col] < Quantile1 - outlier_factor) | (dataframe_to_check[col] > Quantile3 + outlier_factor)].index

        #outlier_list=dataframe_to_check[dataframe_to_check[col]<(Quantile1-outlier_factor)| 

         #                               dataframe_to_check[col]>(Quantile3+outlier_factor)].index

        index_of_outliers.extend(outlier_list)

    

    index_of_outliers = Counter(index_of_outliers)        

    multiple_outliers = list( k for k, v in index_of_outliers.items() if v > number_of_total_outliers )

    return multiple_outliers



#Quantile1=np.percentile(Outlier_indices1["Fare"],25)

#Quantile3=np.percentile(Outlier_indices1["Fare"],75)

#IQR

#IQR=Quantile3-Quantile1

#Outliers_steps1=1.5*IQR

#Outliers_steps1

#Outlier_indices11=Outlier_indices1[Outlier_indices1["Fare"]<(Quantile1-Outliers_steps)]

#Outlier_indices111=Outlier_indices1[Outlier_indices1["Fare"]>(Quantile1+Outliers_steps)]
#seeing the outliers by row numbers

#for col in Outlier_indices111:

 #   print(df_train.loc[col])
#dropping the rows for outliers

#df_train.drop(inplace=True,index=Outlier_indices111)
#this is the fare plot i.e the fare is maximum for 1-100 for normal ones

sns.kdeplot(df_train["Fare"])

#by this graph we can see that the maximum crew on titanic was from the age band 

plt.figure(figsize=(10,8))

sns.lmplot(data=df_train,x="Age",y="Fare")

plt.show()


#by this graph we can see that the passengers are mostly man who have paid between 0-100$ and the ones who are survived

plt.figure(figsize=(15,5))

sns.swarmplot(df_train["Title"],df_train["Fare"],hue=df_train["Survived"])
#lets check the dependency of fare vs age

plt.figure(figsize=(10,10))

sns.scatterplot(df_train["Age"],df_train["Fare"],hue=df_train["Survived"])
#checking the outliers for the age

sns.boxplot(df_train["Age"])
#checking the outliers in the age

#Quantile=np.percentile(df_train["Age"],25)



#Quantile3=np.percentile(df_train["Age"],75)

#IQR=Quantile3-Quantile1

#Outlier_factor=1.5*IQR



#outliers_lower=df_train[df_train["Age"]<(Quantile-Outlier_factor)]

#outliers_upper=df_train[df_train["Age"]>(Quantile3-Outlier_factor)]
df_train.info()
outliers_df1=outlier_detection(df_train,2,["Age","SibSp","Parch","Fare"])

outliers_df1
#rows of the outliers

df_train.loc[outliers_df1]
#now lets drop these rows which have more 2 or more than 2 outliers

df_train.drop(index=outliers_df1,inplace=True)
plt.figure(figsize=(10,8))

df_train.boxplot()
#lets see how our numeric data is related to other data

print(df_train[["Parch","Survived","SibSp","Age","Fare"]].corr())

#fare is the most effecting feature on the survival chances
sns.catplot(x="SibSp",y="Survived",data=df_train,kind="bar")
sns.catplot(x="Parch",y="Survived",data=df_train,kind="bar")
#for multiple plots 

g=sns.FacetGrid(df_train,col="Survived")

g=g.map(sns.distplot,"Age")

#the childrens have high chance of survival
g=sns.kdeplot(df_train["Age"][(df_train["Survived"]==1)& (df_train["Age"].notnull())],shade=True)

g=sns.kdeplot(df_train["Age"][(df_train["Survived"]==0)& (df_train["Age"].notnull())],ax=g,shade=True)

g.legend(["Not Survived","Survived"])
df_train["data"],df_test["data"]="train","test"

final_df=pd.concat(objs=[df_train,df_test],axis=0).reset_index(drop=True)

final_df.shape
final_df["Fare"].isnull().sum()

final_df["Fare"] = final_df["Fare"].fillna(final_df["Fare"].median())
sns.kdeplot(final_df["Fare"],color='g',label="skewness is : "+str(final_df["Fare"].skew()))
#as we can see that the distribution is pareto so the scaling wont work for it

#lets transform it using log distribution

final_df["Fare"]=final_df["Fare"].map(lambda x: np.log(x) if x>0 else 0)
#now lets check the distribution of the Fare column

sns.kdeplot(final_df["Fare"])
df_train.isnull().any()
#now lets see how pclass effect the SUrvived



sns.catplot(y="Survived",x="Pclass",data=df_train,kind='bar',height=6)

#by seeing this graph we can clearly see that the class 1 passengers have high probability of survival
#lets check the passsenger class and the survival vs sex

sns.catplot(data=df_train,x="Pclass",y="Survived",hue="Sex",kind="bar")

#we can see here too even the female from lower classes have high chances of survival
#filled the na of the embarked column of the final dataframe

final_df["Embarked"]=final_df["Embarked"].fillna('S')
#lets plot the embarked with survived

sns.countplot(df_train["Embarked"],hue=df_train["Survived"])

#we can see that the Embarked 'C' class has the more survivors than the people who have not survived
#lets impute the missing values

#lets check the most co related features with the age

df_train.corr()
#INDEX	PassengerId	Survived	Pclass	Sex	Age	SibSp	Parch	Fare	title

#Age	0.034172	-0.076867	-0.374495	0.092778	1.000000	-0.307129	-0.186457	0.110219	-0.149322

#we will check the most co related features first

#1-Pclass- (-.3744)

#2-SibSp-(-0.307129)

#3-Parch-(-0.186457)

#now we will see the graph

g=sns.factorplot(kind="box",x="Pclass",y="Age",data=df_train)

g=sns.factorplot(data=df_train,x="SibSp",y="Age",kind='box')

g=sns.factorplot(data=df_train,x="Parch",y="Age",kind='box')
# Filling missing value of Age 



## Fill Age with the median age of similar rows according to Pclass, Parch and SibSp

# Index of NaN age rows

index_NaN_age = list(final_df["Age"][final_df["Age"].isnull()].index)





for i in index_NaN_age :

    age_med = final_df["Age"].median()

    age_pred = final_df["Age"][((final_df['SibSp'] == final_df.iloc[i]["SibSp"]) & (final_df['Parch'] == 

                                                                                    final_df.iloc[i]["Parch"]) & 

                                (final_df['Pclass'] == final_df.iloc[i]["Pclass"]))].median()

    if not np.isnan(age_pred) :

        final_df['Age'].iloc[i] = age_pred

    else :

        final_df['Age'].iloc[i] = age_med
sns.factorplot(x="Survived",y="Age",data=df_train,kind='box')

#the people of young age has high chance of survival
#feature engineering

# Create a family size descriptor from SibSp and Parch

final_df["Fsize"] = final_df["SibSp"] + final_df["Parch"] + 1
sns.factorplot(x="Fsize",y="Survived",data=final_df)
#creating some of the columns for which size of the family

final_df["Single"]=final_df["Fsize"].map(lambda s:1 if s==1 else 0)



final_df["SmallF"]=final_df["Fsize"].map(lambda s:1 if s==2 else 0)



final_df["MedF"]=final_df["Fsize"].map(lambda s:1 if 3 <= s <= 4 else 0)



final_df["Large"]=final_df["Fsize"].map(lambda s:1 if s>=5 else 0)
g = sns.factorplot(x="Single",y="Survived",data=final_df,kind="bar")

g = g.set_ylabels("Survival Probability")

g = sns.factorplot(x="SmallF",y="Survived",data=final_df,kind="bar")

g = g.set_ylabels("Survival Probability")

g = sns.factorplot(x="MedF",y="Survived",data=final_df,kind="bar")

g = g.set_ylabels("Survival Probability")

g = sns.factorplot(x="Large",y="Survived",data=final_df,kind="bar")

g = g.set_ylabels("Survival Probability")
final_df = pd.get_dummies(final_df, columns = ["Embarked"], prefix="Em")
final_df.head(5)
final_df["Cabin"].describe()
#filling thenull values with X as this might be indicating the people who do not have any cabin alloted

final_df["Cabin"]=final_df["Cabin"].fillna("X")
#we are extracting the string first element that will be the cabin code

final_df["Cabin"]=final_df["Cabin"].str[0]

final_df["Cabin"].value_counts()
#plotting the cabins with survived

sns.factorplot(x="Cabin",y="Survived",data=final_df,kind='bar')
final_df.columns
final_df["title"]=final_df["title"].fillna(0)
final_df=pd.get_dummies(final_df,columns=["Cabin"],prefix="Cabin")
final_df.columns
final_df["Ticket"]=final_df["Ticket"].str[0]
final_df=pd.get_dummies(final_df,columns=["Ticket"],prefix="Ticket_")
final_df.info()
final_df.drop(["Title"],inplace=True,axis=1)
final_df["data"].value_counts()
df_training=final_df[final_df["data"]=="train"]

df_testing=final_df[final_df["data"]=="test"]



#x_train=df_training.drop(["PassengerId","Survived","data"],axis=1)

#y_train=df_training["Survived"]

x_test=df_testing.drop(["PassengerId","Survived","data"],axis=1)
#x_train.shape,y_train.shape,x_test.shape

#x_train.reset_index(inplace=True,drop=True)



#y_train.reset_index(inplace=True,drop=True)

#x_test.reset_index(inplace=True,drop=True)
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
final_train,final_validation=train_test_split(df_training,test_size=.2,random_state=43)
x_final_train=final_train.drop(["PassengerId","Survived","data"],axis=1)

y_final_train=final_train["Survived"]

x_final_validation=final_validation.drop(["PassengerId","Survived","data"],axis=1)

y_final_validation=final_validation["Survived"]
#model=LogisticRegression()

#model.fit(x_final_train,y_final_train)

from sklearn.model_selection import KFold,StratifiedKFold
# Cross validate model with Kfold stratified cross val

kfold = StratifiedKFold(n_splits=10)
from sklearn.metrics import auc

from sklearn.model_selection import GridSearchCV,RandomizedSearchCV

from sklearn.linear_model import SGDClassifier

from sklearn.ensemble import AdaBoostClassifier,ExtraTreesClassifier,GradientBoostingClassifier,RandomForestClassifier

from xgboost import XGBClassifier,XGBRFClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
# Modeling step Test differents algorithms 

random_state = 2

classifiers = []

classifiers.append(SVC(random_state=random_state))

classifiers.append(DecisionTreeClassifier(random_state=random_state))

classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),random_state=random_state,learning_rate=0.1))

classifiers.append(RandomForestClassifier(random_state=random_state))

classifiers.append(ExtraTreesClassifier(random_state=random_state))

classifiers.append(GradientBoostingClassifier(random_state=random_state))

classifiers.append(MLPClassifier(random_state=random_state))

classifiers.append(KNeighborsClassifier())

classifiers.append(LogisticRegression(random_state = random_state))

classifiers.append(LinearDiscriminantAnalysis())



cv_results = []

for classifier in classifiers :

    cv_results.append(cross_val_score(classifier, x_final_train, y = y_final_train, scoring = "accuracy", cv = kfold, n_jobs=4))



cv_means = []

cv_std = []

for cv_result in cv_results:

    cv_means.append(cv_result.mean())

    cv_std.append(cv_result.std())



cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":["SVC","DecisionTree","AdaBoost",

"RandomForest","ExtraTrees","GradientBoosting","MultipleLayerPerceptron","KNeighboors","LogisticRegression","LinearDiscriminantAnalysis"]})



g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})

g.set_xlabel("Mean Accuracy")

g = g.set_title("Cross validation scores")
# Adaboost

DTC = DecisionTreeClassifier()



adaDTC = AdaBoostClassifier(DTC, random_state=7)



ada_param_grid = {"base_estimator__criterion" : ["gini", "entropy"],

              "base_estimator__splitter" :   ["best", "random"],

              "algorithm" : ["SAMME","SAMME.R"],

              "n_estimators" :[1,2],

              "learning_rate":  [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3,1.5]}



gsadaDTC = GridSearchCV(adaDTC,param_grid = ada_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)



gsadaDTC.fit(x_final_train,y_final_train)



ada_best = gsadaDTC.best_estimator_
ada_best
gsadaDTC.best_score_
#ExtraTrees 

ExtC = ExtraTreesClassifier()





## Search grid for optimal parameters

ex_param_grid = {"max_depth": [None],

              "max_features": [1, 3, 10],

              "min_samples_split": [2, 3, 10],

              "min_samples_leaf": [1, 3, 10],

              "bootstrap": [False],

              "n_estimators" :[100,300],

              "criterion": ["gini"]}





gsExtC = GridSearchCV(ExtC,param_grid = ex_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)



gsExtC.fit(x_final_train,y_final_train)



ExtC_best = gsExtC.best_estimator_



# Best score

gsExtC.best_score_
# RFC Parameters tunning 

RFC = RandomForestClassifier()





## Search grid for optimal parameters

rf_param_grid = {"max_depth": [None],

              "max_features": [1, 3, 10],

              "min_samples_split": [2, 3, 10],

              "min_samples_leaf": [1, 3, 10],

              "bootstrap": [False],

              "n_estimators" :[100,300],

              "criterion": ["gini"]}





gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)



gsRFC.fit(x_final_train,y_final_train)



RFC_best = gsRFC.best_estimator_



# Best score

gsRFC.best_score_
# Gradient boosting tunning



GBC = GradientBoostingClassifier()

gb_param_grid = {'loss' : ["deviance"],

              'n_estimators' : [100,200,300],

              'learning_rate': [0.1, 0.05, 0.01],

              'max_depth': [4, 8],

              'min_samples_leaf': [100,150],

              'max_features': [0.3, 0.1] 

              }



gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)



gsGBC.fit(x_final_train,y_final_train)



GBC_best = gsGBC.best_estimator_



# Best score

gsGBC.best_score_
### SVC classifier

SVMC = SVC(probability=True)

svc_param_grid = {'kernel': ['rbf'], 

                  'gamma': [ 0.001, 0.01, 0.1, 1],

                  'C': [1, 10, 50, 100,200,300, 1000]}



gsSVMC = GridSearchCV(SVMC,param_grid = svc_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)



gsSVMC.fit(x_final_train,y_final_train)



SVMC_best = gsSVMC.best_estimator_



# Best score

gsSVMC.best_score_
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,

                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):

    """Generate a simple plot of the test and training learning curve"""

    plt.figure()

    plt.title(title)

    if ylim is not None:

        plt.ylim(*ylim)

    plt.xlabel("Training examples")

    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(

        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()



    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="r")

    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",

             label="Training score")

    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",

             label="Cross-validation score")



    plt.legend(loc="best")

    return plt



#GradientBoosting and Adaboost classifiers tend to overfit the training set. 

#According to the growing cross-validation curves GradientBoosting and Adaboost could perform better with more training examples.

#SVC and ExtraTrees classifiers seem to better generalize the prediction 

#since the training and cross-validation curves are close together.

g = plot_learning_curve(gsRFC.best_estimator_,"RF mearning curves",x_final_train,y_final_train,cv=kfold)

g = plot_learning_curve(gsExtC.best_estimator_,"ExtraTrees learning curves",x_final_train,y_final_train,cv=kfold)

g = plot_learning_curve(gsSVMC.best_estimator_,"SVC learning curves",x_final_train,y_final_train,cv=kfold)

g = plot_learning_curve(gsadaDTC.best_estimator_,"AdaBoost learning curves",x_final_train,y_final_train,cv=kfold)

g = plot_learning_curve(gsGBC.best_estimator_,"GradientBoosting learning curves",x_final_train,y_final_train,cv=kfold)
#Feature importance of tree based classifiers

#In order to see the most informative features for the prediction of passengers survival,

#Displayed the feature importance for the 4 tree based classifiers which are performing well and generalizing well

nrows = ncols = 2

fig, axes = plt.subplots(nrows = nrows, ncols = ncols, sharex="all", figsize=(15,15))



names_classifiers = [("AdaBoosting", ada_best),("ExtraTrees",ExtC_best),("RandomForest",RFC_best),("GradientBoosting",GBC_best)]



nclassifier = 0

for row in range(nrows):

    for col in range(ncols):

        name = names_classifiers[nclassifier][0]

        classifier = names_classifiers[nclassifier][1]

        indices = np.argsort(classifier.feature_importances_)[::-1][:40]

        g = sns.barplot(y=x_final_train.columns[indices][:40],x = classifier.feature_importances_[indices][:40] , orient='h',ax=axes[row][col])

        g.set_xlabel("Relative importance",fontsize=12)

        g.set_ylabel("Features",fontsize=12)

        g.tick_params(labelsize=9)

        g.set_title(name + " feature importance")

        nclassifier += 1
#getting the result on test data for the results

test_Survived_RFC = pd.Series(RFC_best.predict(x_test), name="RFC")

test_Survived_ExtC = pd.Series(ExtC_best.predict(x_test), name="ExtC")

test_Survived_SVMC = pd.Series(SVMC_best.predict(x_test), name="SVC")

test_Survived_AdaC = pd.Series(ada_best.predict(x_test), name="Ada")

test_Survived_GBC = pd.Series(GBC_best.predict(x_test), name="GBC")





# Concatenate all classifier results

ensemble_results = pd.concat([test_Survived_RFC,test_Survived_ExtC,test_Survived_AdaC,test_Survived_GBC, test_Survived_SVMC],axis=1)





g= sns.heatmap(ensemble_results.corr(),annot=True)
x_train_full=df_training.drop(["PassengerId","data","Survived"],axis=1)

y_train_full=df_training["Survived"]

x_train_full.shape,y_train_full.shape
#I choosed a voting classifier to combine the predictions coming from the 5 classifiers.

votingC = VotingClassifier(estimators=[('rfc', RFC_best), ('extc', ExtC_best),

('svc', SVMC_best), ('adac',ada_best),('gbc',GBC_best)], voting='soft', n_jobs=4)

#traing the algorithm on the basis of the full data

votingC = votingC.fit(x_train_full,y_train_full)
df_test.head(5)
test_Survived = pd.Series(votingC.predict(x_test), name="Survived")

results = pd.concat([df_test["PassengerId"],test_Survived],axis=1)

results.to_csv("ensemble_python_voting.csv",index=False)