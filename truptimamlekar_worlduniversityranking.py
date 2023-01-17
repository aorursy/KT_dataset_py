import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

import numpy as np

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

%matplotlib inline

from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, auc

from sklearn import model_selection

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier

from sklearn.calibration import CalibratedClassifierCV

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.svm import SVC, LinearSVC

from sklearn.linear_model import LogisticRegression, RidgeClassifier

from sklearn.naive_bayes import GaussianNB
df1=pd.read_csv('../input/world-university-rankings/cwurData.csv')

df1.head(2)
df1.describe()
df1.isnull().sum()
df1['broad_impact']=df1['broad_impact'].fillna(df1.groupby(['institution'])['broad_impact'].transform('mean'))

df1['broad_impact']=df1['broad_impact'].fillna(0)
df1.isnull().sum()
df1.shape
from scipy.stats.stats import pearsonr

features = ['quality_of_education','alumni_employment','quality_of_faculty','publications','influence','citations','broad_impact','patents']

target = 'score'

for feature in features:

    coeff = pearsonr(df1[feature], df1[target])[0]

    print('Pearson correlation for ' + feature + ' coeff: ' + str(coeff))
df_top100 = df1.iloc[:100,:]
import plotly.graph_objs as go
citation= go.Scatter(x = df_top100.world_rank,y = df_top100.citations,mode = "lines",name = "citations",marker = dict(color = 'rgba(16, 112, 2, 0.8)'),text= df_top100.institution)

teaching= go.Scatter(x = df_top100.world_rank, y = df_top100.quality_of_faculty, mode = "lines+markers", name = "teaching",marker = dict(color = 'rgba(80, 26, 80, 0.8)'),text= df_top100.institution)
from plotly.offline import init_notebook_mode, iplot

from plotly.graph_objs import *

init_notebook_mode(connected=True)  

# Create a list to add traces

data = [citation, teaching]

layout = dict(title = 'Citation and Teaching vs World Rank of Top 100 Universities',xaxis= dict(title= 'World Rank',ticklen= 5,zeroline= False))

fig1 = dict(data = data, layout = layout)

iplot(fig1)
# Plot evolution of each criteria of the ranking

df2 = df1[['year', 'quality_of_faculty', 'alumni_employment', 'patents', 'citations']].head(20)

ax = df2.plot.bar( stacked=True, x='year')

ax.legend(loc=7, bbox_to_anchor=(1.4, 0.5))
df1['world_rank'] = df1['world_rank'].astype('int64')

df1['national_rank'] = df1['national_rank'].astype('float64')

df1['broad_impact'] = df1['broad_impact'].astype('float64')

df1['score'] = df1['score'].astype('float64')

# Plot evolution of ranking

ax = df1.plot(kind='line',  x='year',  y='world_rank', xlim=(2011, 2016),  ylim=(1, 60),  xticks=range(2011, 2017))

# Have ints for the labels

ax.ticklabel_format(useOffset=False, style='plain')
sns.lmplot(x = 'broad_impact',y= 'world_rank',data = df1)

plt.title('world rank vs broad impact')

plt.show()
# Prepare Data

dff=df1.head(100)

df= dff.groupby('country').size()

# Make the plot with pandas

df.plot(kind='pie', subplots=True, figsize=(8, 8))

plt.title("Pie Chart of Various Category")

plt.ylabel("")

plt.show()
f,ax=plt.subplots(1,1,figsize=(20,4))

df1['country'].value_counts().plot(kind='bar')
f,ax=plt.subplots(1,1,figsize=(8,4))

sns.countplot(x="year",data=df1,palette="muted")
import matplotlib.pyplot as plt

new=df1.head(20)

plt.scatter(new.country,new.institution,) #scatter plot example

plt.xlabel('countries')

plt.ylabel('universities')

plt.title('rank')

plt.show()
f,ax=plt.subplots(1,3,figsize=(25,5))

box1=sns.boxplot(data=df1["quality_of_education"],ax=ax[0],color='c')

ax[0].set_xlabel('quality_of_education')

box1=sns.boxplot(data=df1["alumni_employment"],ax=ax[1],color='c')

ax[1].set_xlabel('alumni_employment')

box1=sns.boxplot(data=df1["quality_of_faculty"],ax=ax[2],color='c')

ax[2].set_xlabel('quality_of_faculty')
f,ax=plt.subplots(1,3,figsize=(25,5))

box1=sns.boxplot(data=df1["publications"],ax=ax[0],color='c')

ax[0].set_xlabel('publications')

box1=sns.boxplot(data=df1["influence"],ax=ax[1],color='c')

ax[1].set_xlabel('influence')

box1=sns.boxplot(data=df1["citations"],ax=ax[2],color='c')

ax[2].set_xlabel('citations')
corr = (df1.corr())

plt.subplots(figsize=(9, 9))

sns.heatmap(corr, vmax=.8,annot=True,cmap="viridis", square=True);
df1['country'].replace([0], 'USA', inplace=True) 

df1['country'].replace([1], 'United Kingdom', inplace=True) 

df1['country'].replace([2], 'China', inplace=True)   

df1['country'].replace([2], 'Japan', inplace=True)   

df1['country'].replace([2], 'Germany', inplace=True)   



f,ax=plt.subplots(1,1,figsize=(25,10))

sns.kdeplot(df1.loc[(df1['country']=='USA'), 'quality_of_faculty'], color='b', shade=True, Label='USA')

sns.kdeplot(df1.loc[(df1['country']=='United Kingdom'), 'quality_of_faculty'], color='g', shade=True, Label='United Kingdom')

sns.kdeplot(df1.loc[(df1['country']=='China'), 'quality_of_faculty'], color='r', shade=True, Label='China')

sns.kdeplot(df1.loc[(df1['country']=='Japan'), 'quality_of_faculty'], color='m', shade=True, Label='Japan')

sns.kdeplot(df1.loc[(df1['country']=='Germany'), 'quality_of_faculty'], color='c', shade=True, Label='Germany')

plt.xlabel('quality of faculty') 

plt.ylabel('Probability Density')
sns.pairplot(df1,vars = ['quality_of_education','alumni_employment','quality_of_faculty','publications', 'influence','citations'] )
f,axes=plt.subplots (1,1,figsize=(15,4))

sns.distplot(df1['score'],kde=True,hist=True,color="r")
df_h=df1.drop(['world_rank','institution','country','national_rank','year'],axis=1)
hist_mean=df_h.hist(bins=10, figsize=(15, 15),grid=False,)
df2014 = df1[df1.year == 2014].iloc[:100,:]

df2015 = df1[df1.year == 2015].iloc[:100,:]

df2012 = df1[df1.year == 2012].iloc[:100,:]
# creating trace for year 2014

trace_2014 = go.Scatter(x = df2014.world_rank, y = df2014.citations, mode = "markers", name = "2014", marker = dict(color = 'rgba(255, 128, 255, 0.8)'), text= df2014.institution)

# creating trace for year 2015

trace_2015 = go.Scatter(x = df2015.world_rank,y = df2015.citations,mode = "markers",name = "2015",marker = dict(color = 'rgba(255, 128, 2, 0.8)'),

                        text= df2015.institution)

# creating trace for year 2016

trace_2012 = go.Scatter(x = df2012.world_rank,y = df2012.citations, mode = "markers", name = "2016", marker = dict(color = 'rgba(0, 255, 200, 0.8)'),text= df2012.institution)
# Create a list to add traces

data = [trace_2014, trace_2015, trace_2012]



layout = dict(title = 'Citation vs world rank of top 100 universities in year 2014, 2015 and 2012',

              xaxis= dict(title= 'World Rank',ticklen= 5,zeroline= False),

              yaxis= dict(title= 'Citation',ticklen= 5,zeroline= False))

fig2 = dict(data = data, layout = layout)

iplot(fig2)
from wordcloud import WordCloud 

df2=df1['country'].to_string()

# Start with one review:

text = df2

# Create and generate a word cloud image:

wordcloud = WordCloud().generate(text)

# Display the generated image:

f,ax=plt.subplots(1,1,figsize=(25,5))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df1['country']=le.fit_transform(df1['country'])
X = df1.drop(['score','institution','year'],axis=1)

Y = df1['score']

x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.2, random_state=7)
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error

lm = LinearRegression()

model = lm.fit(X,Y)

print(f'alpha = {model.intercept_}')

print(f'betas = {model.coef_}')
Y_Pred = lm.predict(x_test)

print('Linear Regression R squared: %.2f' % lm.score(x_test, y_test))
mse = mean_squared_error(Y_Pred, y_test)

rmse = np.sqrt(mse)

print('Linear Regression RMSE: %.2f' % rmse)
model.predict(X)
new_X = [[100,50,8,8,8,9,9,0,8,7,6]]

print(model.predict(new_X))
model = LinearRegression()

model.fit(x_train, y_train)

predictions = model.predict(x_test)

sns.distplot(y_test - predictions, axlabel="Test - Prediction")

plt.show()
df1.insert(14,"chances",0,True)

df1.head(1)
df1.loc[df1['score']> 50, ['chances']] = '1'

df1.loc[df1['score']< 50, ['chances']] = '0'
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df1['country']=le.fit_transform(df1['country'])
X=X.fillna(0)
X = df1.drop(['score','institution','year','chances','broad_impact'],axis=1)

Y=df1['chances'].astype(int)

x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.2, random_state=7)
models = []

models.append(("LR",LogisticRegression()))

models.append(("GNB",GaussianNB()))

models.append(("KNN",KNeighborsClassifier()))

models.append(("DecisionTree",DecisionTreeClassifier()))

models.append(("LDA",  LinearDiscriminantAnalysis()))

models.append(("QDA",  QuadraticDiscriminantAnalysis()))

models.append(("AdaBoost", AdaBoostClassifier()))

models.append(("SVM Linear",SVC(kernel="linear")))

models.append(("SVM RBF",SVC(kernel="rbf")))

models.append(("Random Forest",  RandomForestClassifier()))

models.append(("Bagging",BaggingClassifier()))

models.append(("Calibrated",CalibratedClassifierCV()))

models.append(("GradientBoosting",GradientBoostingClassifier()))

models.append(("LinearSVC",LinearSVC()))

models.append(("Ridge",RidgeClassifier()))
results = []

for name,model in models:

    kfold = KFold(n_splits=10, random_state=0)

    cv_result = cross_val_score(model,x_train,y_train, cv = kfold,scoring = "accuracy")

# It gives you an unbiased estimate of the actual performance you will get at runtime

    results.append(tuple([name,cv_result.mean(), cv_result.std()]))

    results.sort(key=lambda x: x[1], reverse = True)    

for i in range(len(results)):

    print('{:20s} {:2.2f} (+/-) {:2.2f} '.format(results[i][0] , results[i][1] * 100, results[i][2] * 100))
ran_class=RandomForestClassifier()

ran_class.fit(x_train,y_train)

ran_predict=ran_class.predict(x_test)

print(classification_report(y_test,ran_predict))

accuracy=ran_class.score(x_test,y_test)

print(accuracy*100,'%')

cm = confusion_matrix(y_test, ran_predict)

sns.heatmap(cm, annot= True)
false_positive_rate, true_positive_rate, threshold = roc_curve(y_test, ran_predict)

roc_auc = auc(false_positive_rate, true_positive_rate)

plt.figure(figsize = (10,6))

plt.title('Receiver Operating Characteristic')

plt.plot(false_positive_rate, true_positive_rate, color = 'red', label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1], linestyle = '--')

plt.axis('tight')

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')
train_score = ran_class.score(x_train,y_train)

test_score = ran_class.score(x_test,y_test)

print(f'Training Accuracy of our model is: {train_score}')

print(f'Test Accuracy of our model is: {test_score}')
prediction = ran_class.predict(x_train.iloc[15].values.reshape(1,-1))

actual_value = y_train.iloc[15]

print(f'Predicted Value \t: {prediction[0]}')

print(f'Actual Value\t\t: {actual_value}')