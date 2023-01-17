# importing all the necessary libraries

import numpy as np

import pandas as pd

# plotly for data visualization

import plotly.offline as pyo

pyo.init_notebook_mode()

import plotly.graph_objects as go



from sklearn.feature_selection import chi2  

from sklearn.feature_selection import f_classif

from sklearn.feature_selection import SelectKBest

from sklearn.preprocessing import MinMaxScaler



from sklearn import model_selection

from sklearn import metrics



from sklearn import tree

from sklearn import linear_model

from sklearn import ensemble



# sweetviz a popular EDA library 

!pip install sweetviz

import sweetviz

# pandas_profiling another popular EDA library

from pandas_profiling import ProfileReport
def createFolds():

    # read the train CSV

    df = pd.read_csv("../input/framingham-heart-study-dataset/framingham.csv")

    # adding a kfold column with value -1

    df['kfold'] = -1

    # Random Sample the data with df.sample

    # Reset the indices and the drop the Index column 

    df = df.sample(frac=1).reset_index(drop=True)

    # Using Stratified K fold

    # Stratified ensures equal distribution of all classes in each fold

    kf =  model_selection.StratifiedKFold(n_splits=5,shuffle=False,random_state=42)



    for fold,(trainId,valId) in enumerate(kf.split(X=df,y=df.TenYearCHD.values)):

        df.loc[valId,'kfold'] = fold

    #save the new csv file     

    df.to_csv('framinghamFolds.csv',index=False)

createFolds()
# read the csv file and display the head 

df = pd.read_csv("framinghamFolds.csv")

df.head()
report = sweetviz.analyze([df,"Complete DataSet"],target_feat='TenYearCHD')
# savinf the EDA.html file

report.show_html("EDA_SweetViz.html")
profile = ProfileReport(df,title="Framingham Kaggle Data EDA Report")

profile.to_widgets()
profile.to_file("EDA_PandasProfling.html")
targetColumn = 'TenYearCHD'

numericalColumns = ['age',

                   'cigsPerDay',

                    'totChol',

                    'sysBP',

                    'diaBP',

                    'BMI',

                    'heartRate',

                    'glucose']

categoricalColumns = [ column for column in df.columns if column not in numericalColumns + ['kfold', targetColumn]] 
# dropping column education

# df = df.drop(['education'], axis=1)

# df.head()
# for all the numerical variables fill the missing values with the mean

# make a list of features we are interested in  

# kfold and targetColumn is something we should not alter  

features = [x for x in df.columns if x not in ['kfold', targetColumn]]

for feat in features:

    if feat in numericalColumns:

        df[feat] = df[feat].replace(np.NaN,df[feat].mean())

df.isnull().sum()
# Now we can see there are no null values in our data

df.dropna(inplace = True)

df.isnull().sum()
# create a Dataframe only for categorical variables

categoricalDF = df[categoricalColumns]

# select only Top 3 variables 

selector = SelectKBest(chi2,k=5)

# give the targetcolumn and the rest of the data to the scalar to fit

selector.fit(categoricalDF,df[targetColumn])

# get the indicies of the selected columns

cols = selector.get_support(indices=True)



# For display purpose Only

dfscores = pd.DataFrame(selector.scores_)

dfcolumns = pd.DataFrame(categoricalDF.columns)



#concat two dataframes for better visualization 

featureScores = pd.concat([dfcolumns,dfscores],axis=1)

featureScores.columns = ['Features','Score']  #naming the dataframe columns

featureScores = featureScores.sort_values(by='Score', ascending=False)

print(featureScores)

# plotting in plotly

# defining data

trace = go.Bar(x=featureScores['Features'],y=featureScores['Score'])

data=[trace]

# defining layout

layout = go.Layout(title='Chi-Score TEST For Categorical Features',xaxis=dict(title='Feature Name'),

                  yaxis=dict(title='Score'),hovermode='closest')

# defining figure and plotting

figure = go.Figure(data=data,layout=layout)

pyo.iplot(figure)
# create a new dataframe from the selected columns

selectedCategoricalDF = categoricalDF.iloc[:,cols]

selectedCategoricalDF.head()
finalDF = pd.concat([selectedCategoricalDF,df[numericalColumns]],axis=1)

finalDF
scaler = MinMaxScaler(feature_range=(0,1)) 



normalizedDF = pd.DataFrame(scaler.fit_transform(finalDF), 

                         columns=finalDF.columns)

# add the target and kfold column to the new dataframe

normalizedDF[targetColumn] = df[targetColumn].to_list()

normalizedDF['kfold'] = df['kfold'].to_list()

normalizedDF.head()
models = {  "decision_tree_gini": tree.DecisionTreeClassifier(criterion="gini"),  

          "decision_tree_entropy": tree.DecisionTreeClassifier(criterion="entropy"),

          "logistic_regression" : linear_model.LogisticRegression(), 

          "random_forest" : ensemble.RandomForestClassifier(n_jobs=-1)

         }
def run(fold,df,model):

    

    # get the Training and validation data for this fold

    # training data is where the kfold is not equal to the fold

    # validation data is where the kfold is equal to the fold

    trainDF = df[df.kfold != fold].reset_index(drop=True)

    valDF = df[df.kfold==fold].reset_index(drop=True)

    

    # drop the kfold and TenYearCHD column    

    # convert it into a numpy array

    xTrain = trainDF.drop(['kfold','TenYearCHD'],axis=1).values

    yTrain = trainDF.TenYearCHD.values

    

    # perform the same for validation

    xVal = valDF.drop(['kfold','TenYearCHD'],axis=1).values

    yVal = valDF.TenYearCHD.values

    

    # fetch the model from the model dispatcher

    clf = models[model]

    

    #fit the model on the training data

    clf.fit(xTrain,yTrain)

    

    # create probabilities for validation samples

    preds = clf.predict_proba(xVal)[:,1]



    # get roc auc score

    auc = metrics.roc_auc_score(yVal,preds)

    

    print(f"Fold={fold}, AUC SCORE={auc}") 
# select a model from dispacther

model = 'decision_tree_entropy'

# give the fold number , the dataframe and the model

print(model)

for fold in range(5):

    run(fold,normalizedDF,model)
# select a model from dispacther

model = 'logistic_regression'

# give the fold number , the dataframe and the model

print(model)

for fold in range(5):

    run(fold,normalizedDF,model)
# select a model from dispacther

model = 'random_forest'

# give the fold number , the dataframe and the model

print(model)

for fold in range(5):

    run(fold,normalizedDF,model)
# select a model from dispacther

model = 'logistic_regression'

print(model)



# train the model on a single fold

# record all the metrics for different threshold.

fold = 0



trainDF = normalizedDF[normalizedDF.kfold != fold].reset_index(drop=True)

valDF = normalizedDF[normalizedDF.kfold==fold].reset_index(drop=True)



xTrain = trainDF.drop(['kfold','TenYearCHD'],axis=1).values

yTrain = trainDF.TenYearCHD.values



xVal = valDF.drop(['kfold','TenYearCHD'],axis=1).values

yVal = valDF.TenYearCHD.values



clf = models[model]

clf.fit(xTrain,yTrain)

preds = clf.predict_proba(xVal)[:,1]

auc = metrics.roc_auc_score(yVal,preds) 

# we will run for only one fold.

# as the objective here is to look at ROC CURVE.

fpr,tpr,thresholds = metrics.roc_curve(yVal, preds)

print(f"Fold={fold}, AUC SCORE={auc}") 

#draw the ROC Curve.     

# plotting in plotly

# defining data

trace = go.Scatter(x=fpr,y=tpr,text=thresholds)

data=[trace]

# defining layout

layout = go.Layout(title='ROC CURVE',xaxis=dict(title='FPR'),

                  yaxis=dict(title='TPR'),hovermode='closest')

# defining figure and plotting

figure = go.Figure(data=data,layout=layout)

pyo.iplot(figure)

# create blank lists

tnList=[]

fpList=[]

fnList=[]

tpList=[]

precisionList =[]

recallList = []

# lets make a metics dataframe for all the thresholds

for threshold in thresholds:

    #since our predictions are in probability

    # convert them to 0 or 1 based on the threshold     

    tempPred = [1 if x >= threshold else 0 for x in preds] 

    tn, fp, fn, tp = metrics.confusion_matrix(yVal,tempPred).ravel()

    precision = metrics.precision_score(yVal,tempPred)

    recall = metrics.recall_score(yVal,tempPred)

    #append all the values to the appropriate list

    tnList.append(tn)

    fpList.append(fp)

    fnList.append(fn)

    tpList.append(tp)

    precisionList.append(precision)

    recallList.append(recall)



metricsDF = pd.DataFrame(

{'Threshold': thresholds,

 'TN': tnList,

 'FP': fpList,

 'FN' : fnList,

 'TP' : tpList,

 'Precision' : precisionList,

 'Recall' : recallList,

 'TPR' : tpr,

 'FPR' : fpr

})

print(metricsDF)
trace = go.Scatter(x=metricsDF['FP'],y=metricsDF['FN'],marker=dict(color=thresholds,

                                                                   colorscale='viridis',

                                                                   showscale = True),text=thresholds,mode='lines+markers')

data=[trace]

# defining layout

layout = go.Layout(title='False Positive vs False Negative',xaxis=dict(title='False Positive'),

                  yaxis=dict(title='False Negative'),hovermode='closest')

# defining figure and plotting

figure = go.Figure(data=data,layout=layout)

pyo.iplot(figure)
trace = go.Scatter(x=metricsDF['Precision'],y=metricsDF['Recall'],marker=dict(color=thresholds,

                                                                   colorscale='viridis',

                                                                   showscale = True),text=thresholds,mode='lines+markers')

data=[trace]

# defining layout

layout = go.Layout(title='Precision vs Recall',xaxis=dict(title='Precision'),

                  yaxis=dict(title='Recall'),hovermode='closest')

# defining figure and plotting

figure = go.Figure(data=data,layout=layout)

pyo.iplot(figure)
metricsDF[metricsDF['Threshold'].between(0.12 , 0.16,inclusive=True).to_list()]
import joblib
# dump the model

joblib.dump(clf,"classifier.pkl")

# dump the threshold

# selecting a random value between 0.13-0.15 which was our preferred range

threshold = 0.145

joblib.dump(threshold,"threshold.pkl")

# dump the features except kfold,TenYearCHD

dumpFeatures = normalizedDF.drop(['kfold','TenYearCHD'],axis=1).columns.to_list()

joblib.dump(dumpFeatures,"features.pkl")

# dump the MinMaxScaler

joblib.dump(scaler,'scaler.pkl')
# create a Dataframe only for categorical variables

numericalDF = df[numericalColumns]

# select only Top 3 variables 

selector = SelectKBest(f_classif,k=4)

# give the targetcolumn and the rest of the data to the scalar to fit

selector.fit(numericalDF,df[targetColumn])

# get the indicies of the selected columns

cols = selector.get_support(indices=True)



# For display purpose Only

dfscores = pd.DataFrame(selector.scores_)

dfcolumns = pd.DataFrame(numericalDF.columns)



#concat two dataframes for better visualization 

featureScores = pd.concat([dfcolumns,dfscores],axis=1)

featureScores.columns = ['Features','Score']  #naming the dataframe columns

featureScores = featureScores.sort_values(by='Score', ascending=False)

print(featureScores)

# plotting in plotly

# defining data

trace = go.Bar(x=featureScores['Features'],y=featureScores['Score'])

data=[trace]

# defining layout

layout = go.Layout(title='ANOVA F-TEST For Numerical Features',xaxis=dict(title='Feature Name'),

                  yaxis=dict(title='Score'),hovermode='closest')

# defining figure and plotting

figure = go.Figure(data=data,layout=layout)

pyo.iplot(figure)
# create a new dataframe from the selected columns

selectedNumericalDF = numericalDF.iloc[:,cols]

selectedNumericalDF.head()
# combine the best numerical and categorical variables

finalDF = pd.concat([selectedCategoricalDF,selectedNumericalDF],axis=1)

finalDF
# normalize the data

scaler = MinMaxScaler(feature_range=(0,1)) 



normalizedDF = pd.DataFrame(scaler.fit_transform(finalDF), 

                         columns=finalDF.columns)

# add the target and kfold column to the new dataframe

normalizedDF[targetColumn] = df[targetColumn].to_list()

normalizedDF['kfold'] = df['kfold'].to_list()

normalizedDF.head()
# select a linear regression and see the auc score

model = 'logistic_regression'

# give the fold number , the dataframe and the model

print(model)

for fold in range(5):

    run(fold,normalizedDF,model)