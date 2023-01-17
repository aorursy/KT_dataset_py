import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import os
%matplotlib inline
print(os.listdir("../input"))

inputData = pd.read_csv(r"../input/diabetes.csv");
print(inputData.dtypes)
print(inputData.columns)
print("Data shape:",inputData.shape)
print(inputData.head())
print(inputData.describe())
print(inputData.info())
print(inputData.isnull().sum())

print ("***************************************")
print ("VISUALIZATIONS IN DATA SET")
print ("***************************************")

fig = plt.figure(figsize = (10,10))
ax = fig.gca()
sns.heatmap(inputData.corr(), annot=True, fmt=".2f")
plt.title("Correlation",fontsize=5)
plt.show()

fig = plt.figure(figsize = (10,10))
ax = fig.gca()
sns.scatterplot(x="Age", y="Glucose", hue="Outcome",data=inputData)
plt.title("Age vs Glucose",fontsize =10)
plt.show()


sns.pairplot(data=inputData,hue="Outcome")
plt.title("Skewness",fontsize =10)
plt.show()

fig = plt.figure(figsize=(15,15))
ax = fig.gca()
inputData['Age'].value_counts().sort_values(ascending=False).plot.bar(width=0.5,edgecolor='k',align='center',linewidth=1)
plt.xlabel('Age',fontsize=10)
plt.ylabel('Counts',fontsize=10)
ax.tick_params(labelsize=10)
plt.title('Age records',fontsize=10)
plt.grid()
plt.ioff()


plt.figure( figsize=(10,10))
inputData['Outcome'].value_counts().plot.pie(autopct="%1.1f%%")
plt.title("Data division on Outcome of diabetes",fontsize=10)
plt.show()

length  = len(inputData.columns[:-1])
colors  = ["r","g","b","m","y","c","k","orange"] 

print ("***************************************")
print ("DISTIBUTION OF VARIABLES IN DATA SET")
print ("***************************************")
plt.figure(figsize=(13,20))
# Leavout the last column of Outcome
for i,j,k in itertools.zip_longest(inputData.columns[:-1],range(length),colors):
    plt.subplot(length/2,length/4,j+1)
    sns.distplot(inputData[i],color=k)
    plt.title(i)
    plt.subplots_adjust(hspace = .3)
    plt.axvline(inputData[i].mean(),color = "k",linestyle="dashed",label="MEAN")
    plt.axvline(inputData[i].std(),color = "b",linestyle="dotted",label="STANDARD DEVIATION")
    plt.legend(loc="upper right")
plt.show()    


print ("***************************************")
print ("MODEL FUNCTION")
print ("***************************************")

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,roc_curve,auc

def model(algorithm,dtrain_x,dtrain_y,dtest_x,dtest_y,of_type):
    
    print ("*****************************************************************************************")
    print ("MODEL - OUTPUT")
    print ("*****************************************************************************************")
    algorithm.fit(dtrain_x,dtrain_y)
    predictions = algorithm.predict(dtest_x)
    
    print (algorithm)
    print ("\naccuracy_score :",accuracy_score(dtest_y,predictions))
    
    print ("\nclassification report :\n",(classification_report(dtest_y,predictions)))
        
    plt.figure(figsize=(13,10))
    plt.subplot(221)
    sns.heatmap(confusion_matrix(dtest_y,predictions),annot=True,fmt = "d",linecolor="k",linewidths=3)
    plt.title("CONFUSION MATRIX",fontsize=20)
    predicting_probabilites = algorithm.predict_proba(dtest_x)[:,1]
    fpr,tpr,thresholds = roc_curve(dtest_y,predicting_probabilites)
    plt.subplot(222)
    plt.plot(fpr,tpr,label = ("Area_under the curve :",auc(fpr,tpr)),color = "r")
    plt.plot([1,0],[1,0],linestyle = "dashed",color ="k")
    plt.legend(loc = "best")
    plt.title("ROC - CURVE & AREA UNDER CURVE",fontsize=20)
    if  of_type == "feat":
        
        dataframe = pd.DataFrame(algorithm.feature_importances_,dtrain_x.columns).reset_index()
        dataframe = dataframe.rename(columns={"index":"features",0:"coefficients"})
        dataframe = dataframe.sort_values(by="coefficients",ascending = False)
        plt.subplot(223)
        ax = sns.barplot(x = "coefficients" ,y ="features",data=dataframe,palette="husl")
        plt.title("FEATURE IMPORTANCES",fontsize =20)
        for i,j in enumerate(dataframe["coefficients"]):
            ax.text(.011,i,j,weight = "bold")
        plt.show()
    elif of_type == "coef" :
        
        dataframe = pd.DataFrame(algorithm.coef_.ravel(),dtrain_x.columns).reset_index()
        dataframe = dataframe.rename(columns={"index":"features",0:"coefficients"})
        dataframe = dataframe.sort_values(by="coefficients",ascending = False)
        plt.subplot(223)
        ax = sns.barplot(x = "coefficients" ,y ="features",data=dataframe,palette="husl")
        plt.title("FEATURE IMPORTANCES",fontsize =20)
        for i,j in enumerate(dataframe["coefficients"]):
            ax.text(.011,i,j,weight = "bold")
        plt.show()    
    elif of_type == "none" :
        plt.show()
        return (algorithm)

def run_classifiers(train_X,train_Y,test_X,test_Y):
    print ("***************************************")
    print ("RANDOM FOREST CLASSIFIER")
    print ("***************************************")
    
    from sklearn.ensemble import RandomForestClassifier
    rf =RandomForestClassifier()
    model(rf,train_X,train_Y,test_X,test_Y,"feat")
    
    print ("***************************************")
    print ("EXTRA TREE CLASSIFIER")
    print ("***************************************")
    
    from sklearn.tree import ExtraTreeClassifier
    etc = ExtraTreeClassifier()
    model(etc,train_X,train_Y,test_X,test_Y,"feat")
    
    print ("***************************************")
    print ("GRADIENT BOOST CLASSIFIER")
    print ("***************************************")
    
    from sklearn.ensemble import GradientBoostingClassifier
    gbc = GradientBoostingClassifier()
    model(gbc,train_X,train_Y,test_X,test_Y,"feat")
    
    print ("***************************************")
    print ("GAUSSIAN NAIVES BAYES CLASSIFIER")
    print ("***************************************")
    
    from sklearn.naive_bayes import GaussianNB
    nb = GaussianNB()
    model(nb,train_X,train_Y,test_X,test_Y,"none")
    
    print ("***************************************")
    print ("K-NEAREST NEIGHBOUT CLASSIFIER")
    print ("***************************************")
    
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier()
    model(knn,train_X,train_Y,test_X,test_Y,"none")
    
    print ("***************************************")
    print ("ADA-BOOST CLASSIFIER")
    print ("***************************************")
    
    from sklearn.ensemble import AdaBoostClassifier
    ada = AdaBoostClassifier()
    model(ada,train_X,train_Y,test_X,test_Y,"feat")
from sklearn.model_selection import train_test_split
splitRatio = 0.2
train , test = train_test_split(inputData,test_size = splitRatio,random_state = 123)

plt.figure(figsize=(12,6))
plt.subplot(121)
train["Outcome"].value_counts().plot.pie(labels = ["1","0"],
                                              autopct = "%1.0f%%",
                                              shadow = True,explode=[0,.1])
plt.title("proportion of target class in train data")
plt.ylabel("")
plt.subplot(122)
test["Outcome"].value_counts().plot.pie(labels = ["1","0"],
                                             autopct = "%1.0f%%",
                                             shadow = True,explode=[0,.1])
plt.title("proportion of target class in test data")
plt.ylabel("")
plt.show()


print ("************************")
print ("RAW DATA AND  PREDICTION")
print ("************************")
#Seperating Predictor and target variables
train_X = train[[x for x in train.columns if x not in ["Outcome"]]]
train_Y = train[["Outcome"]]
test_X  = test[[x for x in test.columns if x not in ["Outcome"]]]
test_Y  = test[["Outcome"]]

run_classifiers(train_X,train_Y,test_X,test_Y)
print ("******************************************************")
print ("ROWISE UNIT NORM NORMALIZATION OF DATA AND  PREDICTION")
print ("******************************************************")
original = inputData.copy();
from sklearn import preprocessing
v = inputData.loc[:, inputData.columns != 'Outcome']
v = preprocessing.normalize(v, norm='l2',axis =1)
inputData.loc[:, inputData.columns != 'Outcome'] = v;
train , test = train_test_split(inputData,test_size = splitRatio,random_state = 123)
#Seperating Predictor and target variables
train_X = train[[x for x in train.columns if x not in ["Outcome"]]]
train_Y = train[["Outcome"]]
test_X  = test[[x for x in test.columns if x not in ["Outcome"]]]
test_Y  = test[["Outcome"]]
run_classifiers(train_X,train_Y,test_X,test_Y)
print ("********************************************")
print ("MINMAX NORMALIZATION OF DATA AND  PREDICTION")
print ("********************************************")
inputData = original.copy()
v = inputData.loc[:, inputData.columns != 'Outcome']
v = v.apply(lambda x:(x.astype(float) - min(x))/(max(x)-min(x)), axis = 1)
inputData.loc[:, inputData.columns != 'Outcome'] = v;
train , test = train_test_split(inputData,test_size = splitRatio,random_state = 123)
#Seperating Predictor and target variables
train_X = train[[x for x in train.columns if x not in ["Outcome"]]]
train_Y = train[["Outcome"]]
test_X  = test[[x for x in test.columns if x not in ["Outcome"]]]
test_Y  = test[["Outcome"]]
run_classifiers(train_X,train_Y,test_X,test_Y)


print ("***************************")
print ("PCA OF DATA AND  PREDICTION")
print ("***************************")
from sklearn.decomposition import PCA
inputData = original.copy()
n_components = 2;
pca = PCA(n_components = n_components).fit(inputData.loc[:, inputData.columns != 'Outcome'])
pcaTransformedData = pca.transform(inputData.loc[:, inputData.columns != 'Outcome'])
pcaDictionary = {}
# Create the dictionarry
for i in range(n_components):
    key = "PCA_component_"+str(i)
    value = pcaTransformedData[:,i]
    pcaDictionary[key] = value
p = pd.DataFrame.from_dict(pcaDictionary)
p["Outcome"] = inputData.Outcome
train , test = train_test_split(p,test_size = splitRatio,random_state = 123)
#Seperating Predictor and target variables
train_X = train[[x for x in train.columns if x not in ["Outcome"]]]
train_Y = train[["Outcome"]]
test_X  = test[[x for x in test.columns if x not in ["Outcome"]]]
test_Y  = test[["Outcome"]]
run_classifiers(train_X,train_Y,test_X,test_Y)


