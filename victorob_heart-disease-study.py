#Main libraries to work with the data

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



#set plots to show without the need of plt.show()

%matplotlib inline



#setting seaborn's plots styles

sns.set_style("darkgrid")

sns.set_palette("colorblind")



#avoid showing warnings

import warnings

warnings.filterwarnings('ignore')
data = pd.read_csv("../input/heart.csv")

data.head() #5 first rows
data["sex_s"] = data["sex"].map({0: "female", 1: "male"})

data["cp_s"] = data["cp"].map({0: "typical angina", 1: "atypical angina", 2: "non-anginal pain", 3: "asymptomatic"})

data["fbs_s"] = data["fbs"].map({0: "<= 120 mg/dl", 1: "> 120 mg/dl"})

data["restecg_s"] = data["restecg"].map({0: "normal", 1: "abnormal", 2: "dangerous"})

data["exang_s"] = data["exang"].map({0: "no", 1: "yes"})

data["slope_s"] = data["slope"].map({0: "upsloping", 1: "flat", 2:"downsloping"})

data["target_s"] = data["target"].map({0: "healthy", 1: "sick"})



data["is_sick"] = data["target"]

data.drop(["target"],axis=1,inplace=True)
data.head(3)
#Columns infos

data.info()
#General description of the numerical values, such as mean, median, std etc.

data.describe()
#Correlation matrix

#More about correlation:

#https://stats.stackexchange.com/questions/18082/how-would-you-explain-the-difference-between-correlation-and-covariance

plt.figure(figsize=(16,8))

sns.heatmap(data=data.corr(),annot=True,cmap="viridis")

plt.title("Correlation Matrix")
#Checking target distribution

sns.countplot(x="target_s",data=data)
fig, ax = plt.subplots(3,2,figsize=(16,12))

sns.boxplot(x="cp_s",y="age",data=data,ax=ax[0][0])

sns.boxplot(x="cp_s",y="age",hue="target_s",data=data,ax=ax[0][1])

sns.boxplot(x="cp_s",y="trestbps",data=data,ax=ax[1][0])

sns.boxplot(x="cp_s",y="thalach",data=data,ax=ax[2][0])

sns.boxplot(x="cp_s",y="trestbps",hue="target_s",data=data,ax=ax[1][1])

sns.boxplot(x="cp_s",y="thalach",hue="target_s",data=data,ax=ax[2][1])
fig, ax = plt.subplots(1,2,figsize=(16,3))

sns.countplot(x="cp_s",hue="target_s",data=data,ax=ax[0])

sns.barplot(x="cp_s",y="is_sick",data=data,ax=ax[1])
#Engineering two new features

group0 = ["typical angina"]

def group_pain(pain):

    return int(pain not in group0)



data["cp_typ_x_rest"] = data["cp_s"].apply(group_pain)

group0.append("asymptomatic")

data["cp_typ_&_asymp_x_rest"] = data["cp_s"].apply(group_pain)
data.head(2) #checking if the two new columns were added
fig, ax = plt.subplots(2,2,figsize=(16,8))

sns.distplot(data["thalach"],ax=ax[0][0])

ax[0][0].set_title("Distribution over the dataset")

sns.kdeplot(data[data["target_s"] == "sick"]["thalach"],ax=ax[0][1],color="red",label="sick")

sns.kdeplot(data[data["target_s"] == "healthy"]["thalach"],ax=ax[0][1],color="green",label="healthy")

ax[0][1].set_title("Distribution over the dataset separated by sick and healthy people")

sns.distplot(data[data["sex_s"] == "female"]["thalach"],ax=ax[1][0],color="orange")

ax[1][0].set_title("Distribution for women")

sns.distplot(data[data["sex_s"] ==   "male"]["thalach"],ax=ax[1][1],color="blue")

ax[1][1].set_title("Distribution for men")

plt.tight_layout()
fig, ax = plt.subplots(1,2,figsize=(16,4))

sns.countplot(x="exang",data=data,ax=ax[0]) #0 - not induced / #1 - induced

ax[0].set_title("Distribution over the dataset")

sns.countplot(x="target_s",hue="exang",data=data,ax=ax[1])

ax[1].set_title("Distribution over the dataset separated by sick and healthy people")
sns.kdeplot(data[data["target_s"] ==    "sick"]["oldpeak"],color="red",label="sick")

sns.kdeplot(data[data["target_s"] == "healthy"]["oldpeak"],color="green",label="healthy")
#Age is a continuous value, so a histogram is appropriate.

sns.distplot(data["age"])

data["age"].describe()
data.loc[data["age"] == 29]
sns.countplot(x="sex_s",hue="target_s",data=data,palette="magma")

print("Males   in dataset: {}".format(data.loc[data["sex_s"] == "male",:].shape[0]))

print("Females in dataset: {}".format(data.loc[data["sex_s"]=="female",:].shape[0]))
fig, ax = plt.subplots(2,2,figsize=(16,8))

#0,0

sns.kdeplot(

    data[(data["sex_s"] == "female") & (data["target_s"] == "sick")]["age"],

    color="yellow",shade=True,ax=ax[0][0])

sns.kdeplot(

    data[(data["sex_s"] == "female") & (data["target_s"] == "healthy")]["age"],

    color="violet",shade=True,ax=ax[0][0])

ax[0][0].legend(labels=("female_sick","female_healthy"))

#0,1

sns.kdeplot(

    data[(data["sex_s"] == "male") & (data["target_s"] == "sick")]["age"],

    color="red",shade=True,ax=ax[0][1])

sns.kdeplot(

    data[(data["sex_s"] == "male") & (data["target_s"] == "healthy")]["age"],

    color="blue",shade=True,ax=ax[0][1])

ax[0][1].legend(labels=("male_sick","male_healthy"))

#1,0

data["age_cats"] = pd.cut(data["age"],bins=[28,40,50,60,100])

sns.countplot(x="age_cats",data=data,ax=ax[1][0])

#1,1

sns.barplot(x="age_cats",y="is_sick",data=data,ax=ax[1][1])

ax[1][1].set_title("Number of sick and healthy people for each age category")
from sklearn.ensemble import ExtraTreesClassifier



print("List of all features: {}".format(data.columns))
def feature_importances(dataset,features_list,target,test_size=.25,random_state=14):

    """

    Wrap-up function to train an ExtraTreesClassifier and return a descending ordered feature importance list

    """

    etc = ExtraTreesClassifier(n_estimators=50)

    etc.fit(dataset[features_list],dataset[target])

    fi = pd.DataFrame(data=etc.feature_importances_,index=X,columns=["Feature Importance"])

    return fi.sort_values(by="Feature Importance",ascending=False)
#Using the original 13 features

X = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']

y = 'is_sick'



feature_importances(data,X,y)
#Using cp as the 1st engineered feature

X = ['age','sex','cp_typ_x_rest','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']

y = 'is_sick'



feature_importances(data,X,y)
X = ['age','sex','cp_typ_&_asymp_x_rest','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope',

     'ca','thal']

y = 'is_sick'



feature_importances(data,X,y)
X = ['age','sex','cp','cp_typ_x_rest','cp_typ_&_asymp_x_rest','trestbps','chol','fbs','restecg','thalach','exang',

     'oldpeak','slope','ca','thal']

y = 'is_sick'



feature_importances(data,X,y)
#Base algorithms, no ensembling for now

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB



#Cross validation

from sklearn.model_selection import cross_validate, GridSearchCV, train_test_split



#Scaling

from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler



#Metrics to evaluate models

#General metrics for classifiers

from sklearn.metrics import classification_report, confusion_matrix 



#Metrics for precision/recall trade-off (more of this later in this notebook)

from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, precision_score, recall_score



import time #built-in library to measure time
def autotrain(X,y,scoring="accuracy",cv_split=5,title=""):

    """

    Performs cross validation of defined base models and presents results as a dataframe sorted by best test scores.

    Adapted from LD Freeman's kernel:

    https://www.kaggle.com/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy

    """

    #define base training models.

    models = [KNeighborsClassifier(), SVC(gamma="auto"), LogisticRegression(solver="liblinear"), 

              DecisionTreeClassifier(), GaussianNB()]

    

    #create a dataframe to store training information and display after all iterations are finished.

    results = pd.DataFrame(columns=["Algorithm","Base Estimator",

                                    "Train Time","Train Score", "Test Score","Scaling Method"])

    

    print(title) #title for the resulting dataframe

    for i,model in enumerate(models):

        #define scalers to try

        scalers = [StandardScaler(),MinMaxScaler(),MaxAbsScaler(),RobustScaler()]

        results.loc[i,"Algorithm"] = model.__class__.__name__

        training = cross_validate(model,X,y,cv=cv_split,scoring="accuracy",return_train_score=True) #

        results.loc[i,"Base Estimator"] = str(model)

        results.loc[i,"Train Time"] = training["fit_time"].sum()

        results.loc[i,"Train Score"] = training["train_score"].mean()

        results.loc[i,"Test Score"] = training["test_score"].mean()

        results.loc[i,"Scaling Method"] = "Unscaled"

        #print("Model: {}".format(model.__class__.__name__))

        #print("Testing Score (unscaled): {}".format(training["test_score"].mean()))    

        for scaler in scalers:

            X_scaled = scaler.fit_transform(X)

            training = cross_validate(model,X_scaled,y,cv=cv_split,scoring="accuracy",return_train_score=True)

            #print("Testing Score ({}): {}".format(scaler.__class__.__name__,training["test_score"].mean()))

            if training["test_score"].mean() > results.loc[i,"Test Score"]:

                results.loc[i,"Train Score"] = training["train_score"].mean()

                results.loc[i,"Test Score"] = training["test_score"].mean()

                results.loc[i,"Scaling Method"] = scaler.__class__.__name__

        #print("*"*50)

        

    return results.sort_values(by="Test Score",ascending=False)
X = data[['age','sex','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','cp']]

y = data['is_sick']

autotrain(X,y,title="Model features with all 13 original features")
X = data[['age','sex','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal',

          'cp_typ_x_rest']]



autotrain(X,y,title="Model features including engineered feature for chest pain typical angina X rest")
X = data[['thalach','exang','slope','ca','thal','cp']]



autotrain(X,y,title="Model features with the best features (according to ExtraTreesClassifier) already existing")
X = data[['thalach','slope', 'ca','thal','cp_typ_x_rest']]



autotrain(X,y,title="Model features with the best features (according to ExtraTreesClassifier) and engineered cp")
X_svm1 = RobustScaler().fit_transform(data[['thalach','exang','slope','ca','thal','cp']])

X_svm2 = RobustScaler().fit_transform(data[['thalach','slope', 'ca','thal','cp_typ_x_rest']])

X_lreg = StandardScaler().fit_transform(data[['age','sex','trestbps','chol','fbs','restecg','thalach','exang',

                                                'oldpeak','slope','ca','thal','cp']])



y = data['is_sick']



svm1_X_train, svm1_X_test, y_train, y_test = train_test_split(X_svm1, y, test_size=.3, random_state=14)

svm2_X_train, svm2_X_test, y_train, y_test = train_test_split(X_svm2, y, test_size=.3, random_state=14)

lreg_X_train, lreg_X_test, y_train, y_test = train_test_split(X_lreg, y, test_size=.3, random_state=14)



svm1 = SVC(gamma="auto")

svm2 = SVC(gamma="auto")

lreg = LogisticRegression(solver="liblinear")



svm1.fit(svm1_X_train,y_train)

svm2.fit(svm2_X_train,y_train)

lreg.fit(lreg_X_train,y_train)



print("Support-vector classifiers and logistic regression trained with 70% of the dataset randomly chosen")

print("Number of instances for training: {}".format(lreg_X_train.shape[0]))

print("Number of instances for testing: {}".format(lreg_X_test.shape[0]))
fig, ax = plt.subplots(1,3,figsize=(16,4))



svm1_predict = svm1.predict(svm1_X_test)

svm1_conf_matrix = confusion_matrix(y_test,svm1_predict)



svm2_predict = svm2.predict(svm2_X_test)

svm2_conf_matrix = confusion_matrix(y_test,svm2_predict)



lreg_predict = lreg.predict(lreg_X_test)

lreg_conf_matrix = confusion_matrix(y_test,lreg_predict)



labels = ("healthy","sick")



sns.heatmap(svm1_conf_matrix,   annot=True,cmap="coolwarm_r",xticklabels=labels,yticklabels=labels,ax=ax[0])

ax[0].set_title("Support-Vector Classifier #1")

ax[0].set_ylabel("Actual Values", fontsize=16)

ax[0].set_xlabel("Predicted Values", fontsize=16)

#(with RobustScaler & only best existing features)



sns.heatmap(svm2_conf_matrix,   annot=True,cmap="coolwarm_r",xticklabels=labels,yticklabels=labels,ax=ax[1])

ax[1].set_title("Support-Vector Classifier #2")

ax[1].set_xlabel("Predicted Values", fontsize=16)

#(with RobustScaler & engineered cp and best features)



sns.heatmap(lreg_conf_matrix, annot=True,cmap="coolwarm_r",xticklabels=labels,yticklabels=labels,ax=ax[2])

ax[2].set_title("Logistic Regressor Classifier")

ax[2].set_xlabel("Predicted Values", fontsize=16)

#(with StandardScaler & all 13 existing features)
print("Classification report for Support-Vector Classifier #1".upper())

print("-"*60)

print(classification_report(y_test,svm1_predict,target_names=labels))
#SVM Tuning



params_svm = {

    "kernel": ["rbf", "linear"],

    "C": np.logspace(-5,3,9),

    "gamma": np.logspace(-4,-1,4),

    "decision_function_shape": ["ovo", "ovr"],

    "random_state": [41],

}



# In each fold, the dataset will be splitted 75/25

grid1 = GridSearchCV(SVC(probability=True),iid=False,param_grid=params_svm,cv=4) 



start = time.perf_counter()

grid1.fit(X_svm1,y)

end = time.perf_counter()



print("SVM tuning amount of seconds elapsed: {:.2f}".format(end-start))

print("Best parameters found for this model: {}".format(grid1.best_params_))
svm_tuned = grid1.best_estimator_



svm_tuned_predict = svm_tuned.predict(X_svm1) #checking overall performance for the tuned model

svm_tuned_conf_matrix = confusion_matrix(y,svm_tuned_predict)



sns.heatmap(svm_tuned_conf_matrix,annot=True,cmap="coolwarm_r",xticklabels=labels,yticklabels=labels,fmt="1")

plt.xlabel("Predicted Values")

plt.ylabel("Actual Values")

plt.title("SVM after hyperparameters tuning")
print("Classification report for Support-Vector Classifier (Tuned) #1".upper())

print("-"*60)

print(classification_report(y,svm_tuned_predict,target_names=labels))
#LogReg Tuning



params_lreg = {

    "C": np.logspace(-5,3,9),

    "solver": ("liblinear","lbfgs","sag","saga","newton-cg"),

    "fit_intercept": (True,False),

    "random_state": [41],

}



# In each fold, the dataset will be splitted 75/25

grid2 = GridSearchCV(LogisticRegression(),iid=False,param_grid=params_lreg,cv=4)



start = time.perf_counter()

grid2.fit(X_lreg,y)

end = time.perf_counter()



print("LogReg tuning amount of seconds elapsed: {:.2f}".format(end-start))

print("Best parameters found for this model: {}".format(grid2.best_params_))
print("Classification report for Logistic Regression".upper())

print("-"*60)

print(classification_report(y_test,lreg_predict,target_names=labels))

print("\n")

lreg_tuned = grid2.best_estimator_

lreg_tuned_predict = lreg_tuned.predict(X_lreg)

print("Classification report for Logistic Regression (Tuned) #1".upper())

print("-"*60)

print(classification_report(y,lreg_tuned_predict,target_names=labels))
#SVC

y_preds = svm_tuned.predict(svm1_X_test)

y_scores_svm = svm_tuned.predict_proba(svm1_X_test)[:,1]



pred_and_proba = pd.DataFrame(data={"Final Prediction": y_preds, "Proba": y_scores_svm})
#Checking if threshold is .5

pred_and_proba.loc[(pred_and_proba["Proba"] > .4) & (pred_and_proba["Proba"] < .6),:]
svm_prec, svm_rec, svm_t = precision_recall_curve(y_test,y_scores_svm)

y_scores_lreg = lreg_tuned.predict_proba(lreg_X_test)[:,1]

lreg_prec, lreg_rec, lreg_t = precision_recall_curve(y_test,y_scores_lreg)



fig, ax = plt.subplots(1,2,figsize=(16,3))

plt.sca(ax[0])

plt.step(svm_rec,svm_prec,where="post",alpha=.5,color="r")

plt.fill_between(svm_rec,svm_prec,step="post",alpha=.2,color="r")

plt.xlim(.79,1.001)

plt.xlabel("Recall",fontsize=14)

plt.ylabel("Precision",fontsize=14)

plt.title("Recall vs. Precision",fontsize=18)

plt.sca(ax[1])

plt.step(lreg_rec,lreg_prec,where="post",color="b")

plt.fill_between(lreg_rec,lreg_prec,step="post",alpha=.2,color="b")

plt.xlim(.79,1.001)

plt.xlabel("Recall",fontsize=14)

plt.ylabel("Precision",fontsize=14)

plt.title("Recall vs. Precision",fontsize=18)
plt.plot(np.arange(0,svm_t.shape[0]),svm_t,color="r")

plt.plot(np.arange(0,lreg_t.shape[0]),lreg_t,color="b")
#SVM

for threshold in np.arange(0,1.05,.05):

    y_adj = [1 if y >= threshold else 0 for y in y_scores_svm]

    print("SVM: For threshold of {:.2f}, precision is {:.3f} and recall is {:.3f}".format(

    threshold, precision_score(y_test,y_adj), recall_score(y_test,y_adj)))
plt.figure(figsize=(8, 8))

plt.title("Precision and Recall Scores as a function of the decision threshold")

plt.plot(svm_t, svm_prec[:-1], "b--", label="Precision")

plt.plot(svm_t, svm_rec[:-1], "g-", label="Recall")

plt.ylabel("Score",fontsize=14)

plt.xlabel("Decision Threshold",fontsize=14)

plt.legend(loc='best')
for threshold in np.arange(.4,.46,.01):

    y_adj = [1 if y >= threshold else 0 for y in y_scores_svm]

    print("SVM: For threshold of {:.2f}, precision is {:.3f} and recall is {:.3f}".format(

    threshold, precision_score(y_test,y_adj), recall_score(y_test,y_adj)))
#Confusion matrix for threshold = .42

threshold = .42

y_adj = [1 if y >= threshold else 0 for y in y_scores_svm]

sns.heatmap(confusion_matrix(y_test,y_adj),annot=True,fmt="1")
#Classification Report for threshold = .42

print(classification_report(y_test,y_adj))