#importing all required libraries

import numpy as np # linear algebra

import pandas as pd # data manipulation and analysis

import matplotlib.pyplot as plt #plots

import re

import seaborn as sns

from collections import Counter

from sklearn.ensemble import RandomForestRegressor

from sklearn.multioutput import MultiOutputRegressor

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import learning_curve

from keras.models import Sequential

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import StratifiedKFold

from keras.layers import Dense

# from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import cross_val_score

import numpy

# fix random seed for reproducibility

numpy.random.seed(7)

%matplotlib inline

#to show matplotlib graphics to show up inline
inputDataFrame = pd.read_csv('../input/survey.csv')

inputDataFrame.head(n=5)
#Removing timestamp data, as it dosent makes have any importance in the problem we are dealing with

inputDataFrame.drop("Timestamp",1, inplace=True)

#Dropping textual comments as the data is very less to extract any useful information

print("Out of 1259 data we have only %i textual comments"%(len(inputDataFrame.comments.unique())))

inputDataFrame.drop("comments",1,inplace=True)

#First lets check the types of data available

inputDataFrame.dtypes
# Quick exploration of data

print("We are dealing with %s rows and %s columns"%(inputDataFrame.shape[0],inputDataFrame.shape[1]))

inputDataFrame.isnull().sum()
inputDataFrame.self_employed.mode()[0]
#Since only 18/1259 missing values in self employed, we will

modeSelfEmployed=inputDataFrame.self_employed.mode()[0]

inputDataFrame.self_employed.fillna(modeSelfEmployed, inplace=True)
#Handling missing values in state

#Checking on an average how many missing states are there in a country

inputDataFrame.groupby('Country')['state'].apply(lambda x: x.isnull().mean())
#Dropping the state information, for data set as a whole, its biased towards US states and wont be a

#strong feature foing forward

inputDataFrame.drop("state",1,inplace=True)
inputDataFrame.work_interfere.unique()

#We will deal with it later when we convert it into category

#Dropping work interfere, since its not a predictor that will effect treatment ,

# it describes what if you have mental health problem

inputDataFrame.drop("work_interfere",1,inplace=True)
#Using random forest for predicting, but scikit dosent support string data in rf ,

#will implement later, as of now, using mode

# notnans = inputDataFrame.work_interfere.notnull()

# inputDataFrame_notnan = inputDataFrame[notnans]

# inputDataFrame_notnan_features=inputDataFrame_notnan.iloc[:, inputDataFrame_notnan.columns != "work_interfere"]

# labels=inputDataFrame_notnan.work_interfere

# # Split into 75% tarain and 25% test

# X_train, X_test, y_train, y_test = train_test_split(inputDataFrame_notnan_features, labels,

#                                                     train_size=0.75,test_size=0.25,

#                                                     random_state=4)

# regr_multirf = MultiOutputRegressor(RandomForestRegressor(max_depth=30,

#                                                           random_state=0))

# regr_multirf.fit(X_train, y_train)



#cleaned up data

inputDataFrame.isnull().sum()
#Converting object datatype to String.

dataType = {}

for f in inputDataFrame.columns:

    dtype = str(inputDataFrame[f].dtype)

    if dtype not in dataType.keys():

        dataType[dtype] = [f]

    else:

        dataType[dtype] += [f]

categories = dataType["object"]

def toString(s):

    return str(s)

for category in categories:

    inputDataFrame[category] = inputDataFrame[category].apply(toString)

# inputDataFrame['Country'].value_counts()

prob = inputDataFrame['Country'].value_counts()

threshold =5

mask = prob > threshold

tail_prob = prob.loc[~mask].sum()

prob = prob.loc[mask]

prob['other'] = tail_prob

prob.plot(kind='bar')

plt.xticks(rotation=90)

plt.show()
prob = inputDataFrame['no_employees'].value_counts()

prob.plot(kind='bar')

plt.xticks(rotation=45)

plt.show()
def renameCol(col,name):

    df1=inputDataFrame.rename(columns={col:name})

    return df1[name]

    
treatmentDf=renameCol("treatment",'Have you sought treatment for a mental health condition?')
#benefits: Does your employer provide mental health benefits?

benfitsdf=renameCol("benefits",'Does your employer provide mental health benefits?')

tab=pd.crosstab(benfitsdf,treatmentDf,normalize = "index")

tab
anonymitydf=renameCol("anonymity",' Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources?')

tab=pd.crosstab(anonymitydf,treatmentDf,normalize = "index")

tab
leavedf=renameCol("leave",'How easy is it for you to take medical leave for a mental health condition?')

tab=pd.crosstab(leavedf,treatmentDf,normalize = "index")

tab
#cleaning gender data

def cleanGender(Gender):

    gender=str(Gender).lower();



    if ("female" in gender or gender[0]=='f') and gender!="fluid":

        return "female"

    elif "male" in gender or gender[0]=='m':

        return "male"

    else:

        return "others"

#Apply function to all values in Age column

inputDataFrame['Gender'] = inputDataFrame['Gender'].apply(cleanGender)

print("After cleaning %i different values in Gender, such as \n %s"%(len(inputDataFrame.Gender.unique()),sorted(inputDataFrame.Gender.unique())))
Genderdf=renameCol("Gender",'What is your gender')

tab=pd.crosstab(Genderdf,treatmentDf,normalize = "index")

tab
family_historydf=renameCol("family_history",'Do you have a family history of mental illness?')

tab=pd.crosstab(family_historydf,treatmentDf,normalize = "index")

tab
care_optionsdf=renameCol("care_options",'Do you know the options for mental health care your employer provides?')

tab=pd.crosstab(care_optionsdf,treatmentDf,normalize = "index")

tab
inputDataFrame['Age'] = pd.to_numeric(inputDataFrame['Age'],errors='coerce')

def clean(age):

    if age>=0 and age<=100:

        return int(age)

    else:

        return np.nan

#Apply function to all values in Age column

inputDataFrame['Age'] = inputDataFrame['Age'].apply(clean)

inputDataFrame=inputDataFrame.dropna()

print("Number of invalid Ages %i"%(inputDataFrame.Age.isnull().sum()))

print("Valid Ages ",sorted(inputDataFrame.Age.unique()))

inputDataFrame.Age.unique()
fig,ax = plt.subplots(figsize=(8,6))

sns.distplot(inputDataFrame['Age'].dropna(),ax=ax)

plt.title('Age Distribution')

plt.ylabel('Freq')

s="Mean Age : " +str(inputDataFrame['Age'].describe())

plt.text(40,0.05,s);
# inputDataFrame = pd.read_csv('../input/survey.csv')
df1=inputDataFrame[["Age","treatment"]]
df1.Age.unique()
df1.boxplot(column="Age",by="treatment")
print("Total %i different values in Gender, such as \n %s"%(len(inputDataFrame.Gender.unique()),sorted(inputDataFrame.Gender.unique())))
prob = inputDataFrame['Gender'].value_counts()

ax=prob.plot.bar(width=.7) 

for i, v in prob.reset_index().iterrows():

    ax.text(i, v.Gender + 0.2 , v.Gender, color='red')

plt.xticks(rotation=40)

plt.show()

for features in inputDataFrame.select_dtypes(include=['object']):

    answers_count = len(inputDataFrame[features].unique()) 

    if answers_count == 2:

        first = list(inputDataFrame[features].unique())[-1]

        inputDataFrame[features] = (inputDataFrame[features] == first).astype(int)

        print(features," & ",end="")
inputDataFrame.benefits=LabelEncoder().fit_transform(inputDataFrame.benefits) 

inputDataFrame.care_options=LabelEncoder().fit_transform(inputDataFrame.care_options) 

inputDataFrame.wellness_program=LabelEncoder().fit_transform(inputDataFrame.wellness_program) 

inputDataFrame.seek_help=LabelEncoder().fit_transform(inputDataFrame.seek_help) 

inputDataFrame.anonymity=LabelEncoder().fit_transform(inputDataFrame.anonymity) 

inputDataFrame.mental_health_consequence=LabelEncoder().fit_transform(inputDataFrame.mental_health_consequence) 

inputDataFrame.phys_health_consequence=LabelEncoder().fit_transform(inputDataFrame.phys_health_consequence) 

inputDataFrame.coworkers=LabelEncoder().fit_transform(inputDataFrame.coworkers) 

inputDataFrame.mental_health_interview=LabelEncoder().fit_transform(inputDataFrame.mental_health_interview) 

inputDataFrame.phys_health_interview=LabelEncoder().fit_transform(inputDataFrame.phys_health_interview) 

inputDataFrame.Gender=LabelEncoder().fit_transform(inputDataFrame.Gender) 

inputDataFrame.supervisor=LabelEncoder().fit_transform(inputDataFrame.supervisor) 

inputDataFrame.mental_vs_physical=LabelEncoder().fit_transform(inputDataFrame.mental_vs_physical) 
#Label Encoding

# inputDataFrame["leave"]=inputDataFrame["leave"].astype('category')

# inputDataFrame["leave"]=inputDataFrame["leave"].cat.codes
# Label Encoding

cleanup_nums = {"leave":     {"Very easy": 1, "Somewhat easy": 2, "Somewhat difficult": 3,

                              "Very difficult": 4,"Don't know": 0},

                "no_employees": {"1-5": 1,"6-25": 2, "26-100": 3, "100-500": 4,

                                  "500-1000": 5, "More than 1000": 6}}

inputDataFrame.replace(cleanup_nums, inplace=True)

inputDataFrame.head(3)
def fitModel(n_splits,ep,nlayer1,nlayer2):

    # define 10-fold cross validation test harness

    #Label Encoding countries

    inputDataFrame.Country=LabelEncoder().fit_transform(inputDataFrame.Country) 

    inputDataFrame_features=inputDataFrame.iloc[:, inputDataFrame.columns != "treatment"]

    labels=inputDataFrame.treatment

    # # Split into 75% tarain and 25% test

    X_train, X_test, y_train, y_test = train_test_split(inputDataFrame_features, labels,

                                                        train_size=0.75,test_size=0.25,

                                                        random_state=4)

    #22 features

    # split into input (X) and output (Y) variables

    X=X_train.as_matrix()

    Y=y_train.as_matrix()

    Xt=X_test.as_matrix()

    Yt=y_test.as_matrix()

    kfold = StratifiedKFold(n_splits, shuffle=True, random_state=15)

    cvscores = []

    trainscores=[]

    iterations=[]

    count=1

    batchsize=100

    for train, test in kfold.split(X, Y):

        # create model

        model = Sequential()

        model.add(Dense(nlayer1, input_dim=22, activation='relu'))

        model.add(Dense(nlayer2, input_dim=22, activation='relu'))

        model.add(Dense(8, activation='relu'))

        model.add(Dense(1, activation='sigmoid'))

        # Compile model

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        #fit the model

        model.fit(X[train], Y[train], batchsize,ep)

        # evaluate the model

        scores=model.evaluate(X[train], Y[train],verbose=0)

        trainscores.append(scores[1] * 100)

        scores=model.evaluate(X[test], Y[test],verbose=0)

        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

        cvscores.append(scores[1] * 100)

        iterations.append(count)

        count+=1;

    print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))

    return iterations,cvscores,trainscores,Xt,Yt,model
#Method to plot train scores and cv scores for analysis

def plot(iterations,trainscores,cvscores):

    plt.plot(iterations, trainscores, 

             color='blue', marker='o', 

             markersize=5, label='training accuracy')

    plt.plot(iterations, cvscores, 

             color='green', linestyle='--', 

             marker='s', markersize=5, 

             label='validation accuracy')

    plt.grid()

    plt.xlabel('Number of iterations')

    plt.ylabel('Accuracy')

    plt.legend(loc='lower right')

    # plt.ylim([0.4, 1.0])

    plt.tight_layout()

    # plt.savefig('./figures/learning_curve.png', dpi=300)

    plt.show()
#first try

iterations,cvscores,trainscores,Xt,Yt,model=fitModel(5,15,80,80)

plot(iterations,trainscores,cvscores)

#Accuracy : 56.81% (+/- 6.39%)

#split=5

#epoch=15

#nlayer1=80

#epoch=80

#Seems model suffers from underfitting

#Need to train Longer and denser layer
#Second try : Better performance but still high bias

iterations,cvscores,trainscores,Xt,Yt,model=fitModel(15,100,100,80)

plot(iterations,trainscores,cvscores)

#Accuracy : 66.28% (+/- 3.26%)

#split=15

#epoch=100

#nlayer1=100

#Seems model suffers from underfitting

#Need to train Longer and denser layer
#third try : Decreased accuracy with greater overfitting

iterations,cvscores,trainscores,Xt,Yt,model=fitModel(30,100,500,30)

plot(iterations,trainscores,cvscores)

#Accuracy : 64.72% (+/- 10.42%)

#split=30

#epoch=100

#nlayer1=500

#nlayer2=30

#Still Overfitting
#fourth try : 

iterations,cvscores,trainscores,Xt,Yt,model=fitModel(5,800,500,30)

plot(iterations,trainscores,cvscores)

#Accuracy : 64.72% (+/- 10.42%)

#split=15

#epoch=800

#nlayer1=100

#nlayer2=10

#Still Overfitting
#Usinf random forest classifier to get most important features



labels=inputDataFrame.treatment

# # Split into 75% tarain and 25% test

X_train, X_test, y_train, y_test = train_test_split(inputDataFrame_features, labels,

                                                    train_size=0.75,test_size=0.25,

                                                    random_state=4)

clf = RandomForestClassifier(max_depth=40,min_samples_split=10, n_estimators=50, random_state=1)

clf.fit(X_train,y_train)

feature_Import=list(zip(X_train,clf.feature_importances_))

sorted(feature_Import, key=lambda x: x[1], reverse=True)