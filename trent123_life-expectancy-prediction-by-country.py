import pandas as pd 

from matplotlib import pyplot as plt

import numpy as np

import math

import seaborn as sns

import sklearn

from sklearn import linear_model

from sklearn import preprocessing

import statsmodels.api as sm

import pylab 

import scipy.stats as stats

from sklearn.metrics import mean_squared_error

from sklearn.metrics import explained_variance_score

from sklearn import ensemble

from sklearn.model_selection import cross_val_score
df=pd.read_csv('LifeExpectancyData.csv')

pd.set_option('display.max_columns', None) 

df.columns=['Country', 'Year', 'Status', 'Life Expectancy', 'Adult Mortality',

       'Infant Deaths', 'Alcohol', 'Percent Expenditure', 'Hep B',

       'Measles', 'BMI', 'U-5 Deaths', 'Polio', 'Total Expenditure',

       'Diphtheria', 'HIVAIDS','GDP', 'Population', 'Thinness 10-19',

       'Thinness 5-9', 'Income Composition', 'Schooling']

#Canada and France are mislabeled as Developing

df[df['Country']=='France']['Status'].replace('Developing','Developed')

df[df['Country']=='Canada']['Status'].replace('Developing','Developed')

df.head(10)
import missingno as msno

print(msno.matrix(df))

df.describe()
#Drop Population and GDP

df=df.drop(['Population','GDP'],axis=1)



#Replace Missing Values Associated with Country Feature Mean

for column in df.columns:

    for i in range(len(df)): 

        country=df['Country'][i]

        status=df['Country'][i]

        if (df[column].isnull()[i]==True):

            df[column][i]=df[df['Country']==country][column].mean() 

        else:

             pass

#Fill Unresolved Values by Status

df1=df[(df['Status']=='Developed')].fillna(df[(df['Status']=='Developed')].mean())

df2=df[(df['Status']=='Developing')].fillna(df[(df['Status']=='Developing')].mean())

df=df2.append(df1)

print(df.shape)

print(msno.matrix(df))





###



####
#Features with Outliers 

numcol=['Life Expectancy', 'Adult Mortality',

       'Infant Deaths', 'Alcohol', 'Percent Expenditure', 'Hep B',

       'Measles', 'BMI', 'U-5 Deaths', 'Polio', 'Total Expenditure',

       'Diphtheria', 'HIV/AIDS', 'Thinness 10-19', 'Thinness 5-9',

       'Income Composition', 'Schooling']

for column in numcol:

    if df[column].quantile(.9973)<df[column].max():

        print(column)

        print('99th Percentile',df[column].quantile(.9973))

        print('Max',df[column].max())

        print('Outliers Present in Column {}'.format(column))

        print('')

    elif df[column].quantile(0)>df[column].min():

        print(column)

        print('99th Percentile',df[column].quantile(.9973))

        print('Min',df[column].min())

        print('Outliers Present in Column {}'.format(column))

        print('')

    else:

        pass
df.describe()
#Life Expectancy 

sns.distplot(df['Life Expectancy'])

plt.axvline(df['Life Expectancy'].mean(),0,.6,color='black')

plt.axvline(df['Life Expectancy'].mean()+df['Life Expectancy'].std(),0,.45,color='black',linestyle='--')

plt.axvline(df['Life Expectancy'].mean()-df['Life Expectancy'].std(),0,.45,color='black',linestyle='--')

plt.axvline(df['Life Expectancy'].mean()+2*df['Life Expectancy'].std(),0,.30,color='black',linestyle='--')

plt.axvline(df['Life Expectancy'].mean()-2*df['Life Expectancy'].std(),0,.30,color='black',linestyle='--')

plt.axvline(df['Life Expectancy'].mean()-3*df['Life Expectancy'].std(),0,.15,color='black',linestyle='--')

sns.set(rc={'figure.figsize':(10,10)})

plt.show()



#QQ plot   

stats.probplot(df['Life Expectancy'], dist="norm", plot=plt)

plt.title('Life Expectancy QQ Plot')

plt.show()

print(stats.shapiro(df['Life Expectancy']))
#Life Expectancy 

sns.distplot(df[df['Status']=='Developed']['Life Expectancy'])

sns.distplot(df[df['Status']=='Developing']['Life Expectancy'],color='y')

labels=['Developed','Developing']

plt.legend(labels=labels,bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)

sns.set(rc={'figure.figsize':(10,10)})

plt.show()



#QQ plot   

stats.probplot(df[df['Status']=='Developed']['Life Expectancy'], dist="norm", plot=plt)

plt.title('Life Expectancy Developed Countries QQ Plot')

print(stats.shapiro(df['Life Expectancy']))

plt.show()

#QQ plot   

stats.probplot(df[df['Status']=='Developing']['Life Expectancy'], dist="norm", plot=plt)

plt.title('Life Expectancy Developing Countries QQ Plot')

print(stats.shapiro(df['Life Expectancy']))

plt.show()



df['Life Expectancy'].groupby(df['Status']).describe()
bins=[36,60,78,90]

labels=[3,2,1]

df['world']=pd.cut(df['Life Expectancy'],bins=bins,labels=labels)
#Life Expectancy 

sns.distplot(df[df['world']==1]['Life Expectancy'])

sns.distplot(df[df['world']==2]['Life Expectancy'],color='y')

sns.distplot(df[df['world']==3]['Life Expectancy'],color='r')



labels=['1st World','2nd World','3rd World']

plt.legend(labels=labels,bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)

sns.set(rc={'figure.figsize':(10,10)})

plt.show()



#QQ plot   

stats.probplot(df[df['world']==1]['Life Expectancy'], dist="norm", plot=plt)

plt.title('Life Expectancy 1st World Countries QQ Plot')

print(stats.shapiro(df['Life Expectancy']))

plt.show()

#QQ plot   

stats.probplot(df[df['world']==2]['Life Expectancy'], dist="norm", plot=plt)

plt.title('Life Expectancy 2nd World Countries QQ Plot')

print(stats.shapiro(df['Life Expectancy']))

plt.show()



stats.probplot(df[df['world']==3]['Life Expectancy'], dist="norm", plot=plt)

plt.title('Life Expectancy 3rd World Countries QQ Plot')

print(stats.shapiro(df['Life Expectancy']))

plt.show()



df['Life Expectancy'].groupby(df['world']).describe()
LEcountry=df.groupby(df['Country'])['Life Expectancy'].mean().sort_values(kind="quicksort",ascending=False)

sns.pointplot(y='Country',x='Life Expectancy',hue='Status',data=df,order=LEcountry.index,join=True)

plt.title( 'Life Expectancy by Country')

plt.axvline(78,0,10,color='g')

plt.axvline(df['Life Expectancy'].mean()-df['Life Expectancy'].std(),0,10,color='r')

sns.set(rc={'figure.figsize':(20,40)})

plt.show()
sns.countplot(df['Status'])

print('Developed or Developing Country Status')

print(df.Status.value_counts()/len(df.Status))

sns.set(rc={'figure.figsize':(10,10)})

plt.show()



sns.countplot(df['world'])

print('1st,2nd,and 3rd World Countries')

print(((df.world.value_counts()/len(df.world))*193).round(0))

sns.set(rc={'figure.figsize':(10,10)})

plt.show()
def LEfactorplot(column):

    x=df[df['Status']=='Developed'][column]

    y=df[df['Status']=='Developed']['Life Expectancy']

    x1=df[df['Status']=='Developing'][column]

    y1=df[df['Status']=='Developing']['Life Expectancy']

    #Fit Lines

    z1 =np.polyfit(x,y,1)

    z2 =np.polyfit(x1,y1,1)

    z1poly = np.poly1d(z1) 

    z2poly = np.poly1d(z2)



    #Plot

    plt.scatter(x,y,alpha=1)

    plt.scatter(x1,y1,alpha=1)

    plt.plot(x,z1poly(x),linewidth=7.0)

    plt.plot(x1,z2poly(x1),linewidth=7.0,color='r')

    labels=['Developed','Developing']

    plt.legend(labels=labels,bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)

    plt.ylabel('Life Expectancy')

    plt.xlabel(column)

LEfactorplot('Year')

plt.xticks(np.arange(2000,2016,1))

plt.show()
corrmat = df.corr()

mask = np.zeros_like(corrmat, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

top_corr_features = corrmat.index

plt.figure(figsize=(15,15))

#plot heat map

sns.heatmap(df[top_corr_features].corr(),annot=True,mask=mask)

sol = (corrmat.where(np.triu(np.ones(corrmat.shape), k=1).astype(np.bool))

                 .stack().sort_values(kind="quicksort",ascending=False))

LE=pd.Series(corrmat.unstack()[18:36]).sort_values(kind="quicksort",ascending=False)

print('Correlation Values for the {} countries left after Data Cleaning:'.format(len(df['Country'].unique())))

LE[1:18]
print('Top 10 Correlated Features Pairs:')

print(sol[0:10],'\n')

print('Bottom 10 Correlated Features Pairs:')

print(sol[143:153])
columns=['Alcohol','BMI','Hep B','Measles','Polio','Diphtheria','HIVAIDS','Thinness 10-19',

         'Thinness 5-9','Adult Mortality','Infant Deaths','U-5 Deaths','Percent Expenditure'

         ,'Total Expenditure','Income Composition','Schooling']



for column,i in zip(columns,range(len(columns))):

    plt.subplot(4,4,i+1)

    sns.distplot(df[column])  

    plt.tight_layout()

    sns.set(rc={'figure.figsize':(20,20)})
for column in df.columns:

    if (column=='Country')or(column=='Status')or(column=='Life Expectancy')or(column=='world')or(column=='Year'):

        pass

    else:

        LEfactorplot(column)

        sns.set(rc={'figure.figsize':(10,10)})

        sns.set(font_scale=1.5)

        plt.show()
#Encode Country and Create copy of dataframe for regression 

df_reg=df.copy()

from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()

df_reg["country_code"] = lb_make.fit_transform(df_reg["Country"])





#Binarize Status

df_reg['Status']=np.where(df_reg['Status']=='Developing',0,1)



df_reg.columns=['Country', 'Year', 'Status', 'Life_Expectancy', 'Adult_Mortality',

       'Infant_Deaths', 'Alcohol', 'Percent_Expenditure', 'Hep_B', 'Measles',

       'BMI', 'U_5_Deaths', 'Polio', 'Total_Expenditure', 'Diphtheria',

       'HIV_AIDS', 'Thinness_10_19', 'Thinness_5_9', 'Income_Composition',

       'Schooling', 'world', 'country_code']
#remove outliers

for col in df_reg.columns:

    if (col=='world')or (col=='Country'):

        pass

    else:

        df_reg=df_reg[np.abs(df_reg[col]-df_reg[col].mean())<=(3*df_reg[col].std())]
#Developed Training Data

X_developed_train=df_reg[(df_reg['Status']==1)&(df_reg['Year']<2011)].drop('Life_Expectancy',axis=1)

Y_developed_train=df_reg[(df_reg['Status']==1)&(df_reg['Year']<2011)]['Life_Expectancy']



#Developed Testing Data

X_developed_test=df_reg[(df_reg['Status']==1)&(df_reg['Year']>2011)].drop('Life_Expectancy',axis=1)

Y_developed_test=df_reg[(df_reg['Status']==1)&(df_reg['Year']>2011)]['Life_Expectancy']



#Developing Training Data

X_developing_train=df_reg[(df_reg['Status']==0)&(df_reg['Year']<2011)].drop('Life_Expectancy',axis=1)

Y_developing_train=df_reg[(df_reg['Status']==0)&(df_reg['Year']<2011)]['Life_Expectancy']



#Developing Testing Data

X_developing_test=df_reg[(df_reg['Status']==0)&(df_reg['Year']>2011)].drop('Life_Expectancy',axis=1)

Y_developing_test=df_reg[(df_reg['Status']==0)&(df_reg['Year']>2011)]['Life_Expectancy']



#World=1 Training Data

X_world1_train=df_reg[(df_reg['world']==1)&(df_reg['Year']<2011)].drop('Life_Expectancy',axis=1)

Y_world1_train=df_reg[(df_reg['world']==1)&(df_reg['Year']<2011)]['Life_Expectancy']



#World=1 Testing Data

X_world1_test=df_reg[(df_reg['world']==1)&(df_reg['Year']>2011)].drop('Life_Expectancy',axis=1)

Y_world1_test=df_reg[(df_reg['world']==1)&(df_reg['Year']>2011)]['Life_Expectancy']



#World=2 Training Data

X_world2_train=df_reg[(df_reg['world']==2)&(df_reg['Year']<2011)].drop('Life_Expectancy',axis=1)

Y_world2_train=df_reg[(df_reg['world']==2)&(df_reg['Year']<2011)]['Life_Expectancy']



#World=2 Testing Data

X_world2_test=df_reg[(df_reg['world']==2)&(df_reg['Year']>2011)].drop('Life_Expectancy',axis=1)

Y_world2_test=df_reg[(df_reg['world']==2)&(df_reg['Year']>2011)]['Life_Expectancy']



#World=3 Training Data

X_world3_train=df_reg[(df_reg['world']==3)&(df_reg['Year']<2011)].drop('Life_Expectancy',axis=1)

Y_world3_train=df_reg[(df_reg['world']==3)&(df_reg['Year']<2011)]['Life_Expectancy']



#World=3 Testing Data

X_world3_test=df_reg[(df_reg['world']==3)&(df_reg['Year']>2011)].drop('Life_Expectancy',axis=1)

Y_world3_test=df_reg[(df_reg['world']==3)&(df_reg['Year']>2011)]['Life_Expectancy']



#Full Training Set

X_train=df_reg[df_reg['Year']<2011].drop('Life_Expectancy',axis=1)

Y_train=df_reg[df_reg['Year']<2011]['Life_Expectancy']



#Full Testing Set

X_test=df_reg[df_reg['Year']>2011].drop('Life_Expectancy',axis=1)

Y_test=df_reg[df_reg['Year']>2011]['Life_Expectancy']



#Full Set

X=df_reg.drop('Life_Expectancy',axis=1)

Y=df_reg['Life_Expectancy']



#Breakdown

Xlist=[X_developed_train,X_developed_test,X_developing_train,X_developing_test,

       X_world1_train,X_world1_test,X_world2_train,X_world2_test,X_world3_train,X_world3_test,

       X_train,X_test,X]

Ylist=[Y_developed_train,Y_developed_test,Y_developing_train,Y_developing_test,

       Y_world1_train,Y_world1_test,Y_world2_train,Y_world2_test,Y_world3_train,Y_world3_test,

       Y_train,Y_test,Y]

xlist=['X_developed_train','X_developed_test','X_developing_train','X_developing_test',

       'X_world1_train','X_world1_test','X_world2_train','X_world2_test','X_world3_train','X_world3_test',

       'X_train','X_test','X']

status=['Developed','Developed','Developing','Developing',

                   '1st World','1st World','2nd World','2nd World','3rd World','3rd World',

                   'Full Training','Full Testing','Full']
df_reg['Life_Expectancy'].describe()
from sklearn.linear_model import TheilSenRegressor

from sklearn.preprocessing import scale

from pylab import rcParams

for x,y,i,z,s in zip(Xlist,Ylist,range(len(Xlist)),xlist,status):

    x=x.drop(['Country','world','Status','Infant_Deaths','Thinness_10_19'],axis=1)

    x=scale(x)

    if i==0:

        print('Thiel {}'.format(z))

        print(z,x.shape)

        #Model

        theil = TheilSenRegressor(random_state=52).fit(x,y)

        

        #R2 

        R=theil.score(x,y)

        print('R^2 Score:{:0.4f}'.format(R))



        #Predictions

        Y_pred=theil.predict(x)

        RMSE=mean_squared_error(y, Y_pred)**0.5

        print('RMSE: {:0.3f}'.format(RMSE))

        print('Minimum LE: {:0.1f}'.format(Y_pred.min()))

        print('Maximum LE: {:0.1f}'.format(Y_pred.max()))

        print('Average Predicted LE: {:0.1f}'.format(Y_pred.mean()))

        print('LE Standard Deviation: {:0.3f}'.format(Y_pred.std()))

        print('LE Variance: {:0.3f}'.format(Y_pred.std()**2))

        

        #plot

        z1=np.polyfit(Y_pred,y,1)

        z1poly = np.poly1d(z1) 

        plt.scatter(Y_pred,y,alpha=1)

        plt.plot(Y_pred,z1poly(Y_pred),linewidth=7.0,color='r')

        plt.title('Thiel {}'.format(z))

        plt.xlabel('Y_pred')

        plt.ylabel('Y')

        rcParams['figure.figsize'] = 10, 10

        plt.show()

        

        #Result DataFrame

        results = pd.DataFrame()

        results["Method"]=['Thiel']

        results['Set']=z

        results['Status']=s

        results['Datapoint Count']=x.shape[0]*x.shape[1]

        results["RMSE"] = RMSE.round(2)

        results["R^2"] = R.round(2)

        results['LE Min']=Y_pred.min().round(1)

        results['LE Max']=Y_pred.max().round(1)

        results['Average LE']=Y_pred.mean().round(1)

        results['LE Std']=Y_pred.std().round(2)

        results['LE Var']=(Y_pred.std()**2).round(1)



    else:

        print('Thiel {}'.format(z))

        print(z,x.shape)

        #Model

        theil = TheilSenRegressor(random_state=52).fit(x,y)

        

        #R2 

        R=theil.score(x,y)

        print('R^2 Score:{:0.4f}'.format(R))



        #Predictions

        Y_pred=theil.predict(x)

        RMSE=mean_squared_error(y, Y_pred)**0.5

        print('RMSE: {:0.3f}'.format(RMSE))

        print('Minimum LE: {:0.1f}'.format(Y_pred.min()))

        print('Maximum LE: {:0.1f}'.format(Y_pred.max()))

        print('Average Predicted LE: {:0.1f}'.format(Y_pred.mean()))

        print('LE Standard Deviation: {:0.3f}'.format(Y_pred.std()))

        print('LE Variance: {:0.3f}'.format(Y_pred.std()**2))

        

        #plot

        z1=np.polyfit(Y_pred,y,1)

        z1poly = np.poly1d(z1) 

        plt.scatter(Y_pred,y,alpha=1)

        plt.plot(Y_pred,z1poly(Y_pred),linewidth=7.0,color='r')

        plt.title('Thiel {}'.format(z))

        plt.xlabel('Y_pred')

        plt.ylabel('Y')

        rcParams['figure.figsize'] = 10, 10

        plt.show()

        

        #Add to results

        results.loc[i] = ['Thiel',z,s,x.shape[0]*x.shape[1]

                          ,RMSE.round(3)

                          ,R.round(4)

                          ,Y_pred.min().round(1)

                          ,Y_pred.max().round(1)

                          ,Y_pred.mean().round(1)

                          ,Y_pred.std().round(3)

                          ,(Y_pred.std()**2).round(3)]
for x,y,i,z,s in zip(Xlist,Ylist,range(len(Xlist)),xlist,status):

    x=x.drop(['Country','world','Status'],axis=1)

    x=scale(x)

    print('Ridge {}'.format(z))

    print(z,x.shape)

    #Model

    ridgeregr = linear_model.Ridge(alpha=10, fit_intercept=True,solver='auto',random_state=65)

    ridge= ridgeregr.fit(x,y)



    #R2 

    R=ridge.score(x,y)

    print('R^2 Score: {:0.4f}'.format(R))



    #Predictions

    Y_pred=ridge.predict(x)

    RMSE=mean_squared_error(y, Y_pred)**0.5

    print('RMSE: {:0.3f}'.format(RMSE))

    print('Minimum LE: {:0.1f}'.format(Y_pred.min()))

    print('Maximum LE: {:0.1f}'.format(Y_pred.max()))

    print('Average Predicted LE: {:0.1f}'.format(Y_pred.mean()))

    print('LE Standard Deviation: {:0.3f}'.format(Y_pred.std()))

    print('LE Variance: {:0.3f}'.format(Y_pred.std()**2))

    

    #plot

    z1 =np.polyfit(Y_pred,y,1)

    z1poly = np.poly1d(z1) 

    plt.scatter(Y_pred,y,alpha=1)

    plt.plot(Y_pred,z1poly(Y_pred),linewidth=7.0,color='r')

    plt.title('Ridge {}'.format(z))

    plt.xlabel('Y_pred')

    plt.ylabel('Y')

    plt.show()

        

    #Add to results

    results.loc[i+13] = ['Ridge',z,s,x.shape[0]*x.shape[1]

                          ,RMSE.round(3)

                          ,R.round(4)

                          ,Y_pred.min().round(1)

                          ,Y_pred.max().round(1)

                          ,Y_pred.mean().round(1)

                          ,Y_pred.std().round(3)

                          ,(Y_pred.std()**2).round(3)]        
cols=['Year','AdultMortality', 'Infant Deaths','Alcohol', 'PercentExpenditure', 'Hep B', 'Measles', 'BMI', 'U5Deaths',

       'Polio', 'TotalExpenditure', 'Diphtheria', 'HIVAIDS', 'Thinness1019',

       'Thinness59', 'IncomeComposition', 'Schooling','country_code']

feature_importances=pd.DataFrame(index=cols)

for x,y,i,z,s in zip(Xlist,Ylist,range(len(Xlist)),xlist,status):

    x

    y

    x=x.drop(['Country','world','Status'],axis=1)

    x=scale(x)

    print('Random Forest {}'.format(z))

    print(z,x.shape)

    #Model

    params = {'n_estimators':100,'max_depth': 3}

    rf = ensemble.GradientBoostingRegressor(**params)

    rfc= rf.fit(x,y)



    #R2 

    R=rfc.score(x,y)

    print('R^2 Score: {:0.4f}'.format(R))



    #Predictions

    Y_pred=rf.predict(x)

    RMSE=mean_squared_error(y, Y_pred)**0.5

    print('RMSE: {:0.3f}'.format(RMSE))

    print('Minimum LE: {:0.1f}'.format(Y_pred.min()))

    print('Maximum LE: {:0.1f}'.format(Y_pred.max()))

    print('Average Predicted LE: {:0.1f}'.format(Y_pred.mean()))

    print('LE Standard Deviation: {:0.3f}'.format(Y_pred.std()))

    print('LE Variance: {:0.3f}'.format(Y_pred.std()**2))

        

    #plot

    z1 =np.polyfit(Y_pred,y,1)

    z1poly = np.poly1d(z1) 

    plt.scatter(Y_pred,y,alpha=1)

    plt.plot(Y_pred,z1poly(Y_pred),linewidth=7.0,color='r')

    plt.title('Gradient Boosting {}'.format(z))

    plt.xlabel('Y_pred')

    plt.ylabel('Y')

    plt.show()

    

    #Feature Importance

    feature_importances[z]=(rfc.feature_importances_*100).round(2)

    print('Top 5 Features\n',feature_importances[z].nlargest(5).round(2),'\n')

        

    #Add to results

    results.loc[i+26] = ['Gradient Boosting',z,s,x.shape[0]*x.shape[1]

                          ,RMSE.round(3)

                          ,R.round(4)

                          ,Y_pred.min().round(1)

                          ,Y_pred.max().round(1)

                          ,Y_pred.mean().round(1)

                          ,Y_pred.std().round(3)

                          ,(Y_pred.std()**2).round(3)] 
feature_importances
results
regsum=results.groupby(['Status','Method']).mean()

regsum
df_cl=df_reg.copy()

from sklearn.preprocessing import scale

from sklearn import metrics

from sklearn.cluster import MeanShift, estimate_bandwidth

Y = df_cl['world']

X_unscaled = df_cl.drop(['Country','world','Status'],1)

X=scale(X_unscaled)



# Here we set the bandwidth. This function automatically derives a bandwidth

# number based on an inspection of the distances among points in the data.

bandwidth = estimate_bandwidth(X, quantile=0.2,n_samples=200)



# Declare and fit the model.

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)

ms.fit(X)

y_pred=ms.predict(X)



# Extract cluster assignments for each data point.

labels = ms.labels_

df_cl['MS Cluster']=labels



# Coordinates of the cluster centers.

cluster_centers = ms.cluster_centers_



# Count our clusters.

n_clusters_ = len(np.unique(labels))



print("Number of estimated clusters: {}".format(n_clusters_))

print('Cal Harabaz Score: {}'.format(metrics.calinski_harabaz_score(X, ms.labels_)/1000))

print('Silhouette Score: {}'.format(metrics.silhouette_score(X, labels, metric='euclidean')))                            

print('Homogenity Score:',metrics.homogeneity_score(y_pred,Y))

print('Completeness Score:',metrics.completeness_score(y_pred,Y))

print('Adjusted Rand Score:',metrics.adjusted_rand_score(y_pred,Y))



print('Cluster Percentage')

((df_cl['MS Cluster'].value_counts()/len(df_cl['MS Cluster'])).round(3))*100
LEcountry=df_cl.groupby(df_cl['Country'])['Life_Expectancy'].mean().sort_values(kind="quicksort",ascending=False)

sns.pointplot(y='Country',x='Life_Expectancy',hue='MS Cluster',data=df_cl,order=LEcountry.index,join=False)

plt.title('Life Expectancy by Country')

plt.axvline(78,0,10,color='g')

plt.axvline(df_cl['Life_Expectancy'].mean()-df_cl['Life_Expectancy'].std(),0,10,color='r')

sns.set(rc={'figure.figsize':(20,40)})

plt.show()
from sklearn.cluster import KMeans

#Scores

complete = []

homogenity = []

silhouette=[]

calinski=[]

adrs=[]



#Cluster Range

ns = list(range(2,15))



#Inputs and Ground Truth

Y = df_cl['world']

X_unscaled = df_cl.drop(['Country','world','Status'],1)

X=scale(X_unscaled)



for n in ns:

    km=KMeans(n_clusters=n, random_state=42)

    km.fit(X)

    y_pred=km.predict(X)

    cal=(metrics.calinski_harabaz_score(X, km.labels_)/1000)

    calinski.append(cal) 

    sil=metrics.silhouette_score(X, km.labels_, metric='euclidean')

    silhouette.append(sil)

    comp = metrics.completeness_score(y_pred,Y)

    complete.append(comp)

    homog = metrics.homogeneity_score(y_pred,Y)

    homogenity.append(homog)

    ar=metrics.adjusted_rand_score(y_pred,Y)

    adrs.append(ar)

    

#Plot

plt.plot(ns,calinski)

plt.plot(ns,silhouette)

plt.plot(ns, complete)

plt.plot(ns,homogenity)

plt.plot(ns,adrs)

plt.title('World Ground Truth')

plt.xlabel('K Values')

plt.ylabel('Score')

plt.legend(['Calinski','Silhouette','Completeness', 'Homogeneity','ARI'],bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.1)

plt.xticks(np.arange(2,15,1))

plt.show()
Y = df_cl['world']

X_unscaled = df_cl.drop(['Country','world','Status'],1)

X=scale(X_unscaled)

ncluster=3

km=KMeans(n_clusters=ncluster, random_state=42)

km.fit(X)

y_pred=km.predict(X)



# Extract cluster assignments for each data point.

labels = km.labels_

df_cl['KM Cluster']=labels



# Coordinates of the cluster centers.

cluster_centers = km.cluster_centers_



print("Number of estimated clusters: {}".format(ncluster))

print('Cal Harabaz Score: {}'.format(metrics.calinski_harabaz_score(X, km.labels_)/1000))

print('Silhouette Score: {}'.format(metrics.silhouette_score(X, labels, metric='euclidean')))  

print('Homogenity Score:',metrics.homogeneity_score(y_pred,Y))

print('Completeness Score:',metrics.completeness_score(y_pred,Y))



print('Cluster Percentage')

((df_cl['KM Cluster'].value_counts()/len(df_cl['KM Cluster'])).round(3))*100
len(df_cl['Country'].unique())
LEcountry=df_cl.groupby(df_cl['Country'])['Life_Expectancy'].mean().sort_values(kind="quicksort",ascending=False)

sns.pointplot(y='Country',x='Life_Expectancy',hue='KM Cluster',data=df_cl,order=LEcountry.index,join=False)

plt.title('Life Expectancy by Country')

plt.axvline(78,0,10,color='g')

plt.axvline(df_cl['Life_Expectancy'].mean()-df_cl['Life_Expectancy'].std(),0,10,color='r')

sns.set(rc={'figure.figsize':(20,40)})
def Clusterplot(column):

    if (column=="world") or (column=='Status'):

        x=df_cl[df_cl['KM Cluster']==0][column]

        y=df_cl[df_cl['KM Cluster']==0]['Life_Expectancy']

        x1=df_cl[df_cl['KM Cluster']==1][column]

        y1=df_cl[df_cl['KM Cluster']==1]['Life_Expectancy']

        x2=df_cl[df_cl['KM Cluster']==2][column]

        y2=df_cl[df_cl['KM Cluster']==2]['Life_Expectancy']

        #Plot

        plt.scatter(x,y,alpha=1,color='b')

        plt.scatter(x1,y1,alpha=1,color='r')

        plt.scatter(x2,y2,alpha=1,color='g')

        labels=['0','1','2']

        plt.legend(labels=labels,bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)

        plt.ylabel('Life Expectancy')

        plt.xlabel(column)

   

    else:

        x=df_cl[df_cl['KM Cluster']==0][column]

        y=df_cl[df_cl['KM Cluster']==0]['Life_Expectancy']

        x1=df_cl[df_cl['KM Cluster']==1][column]

        y1=df_cl[df_cl['KM Cluster']==1]['Life_Expectancy']

        x2=df_cl[df_cl['KM Cluster']==2][column]

        y2=df_cl[df_cl['KM Cluster']==2]['Life_Expectancy']

        #Plot

        plt.scatter(x,y,alpha=1,color='b')

        plt.scatter(x1,y1,alpha=1,color='r')

        plt.scatter(x2,y2,alpha=1,color='g')

        #Fit Lines

        z1 =np.polyfit(x,y,1)

        z2 =np.polyfit(x1,y1,1)

        z3=np.polyfit(x2,y2,1)

        z1poly = np.poly1d(z1) 

        z2poly = np.poly1d(z2)

        z3poly= np.poly1d(z3)



        plt.plot(x,z1poly(x),linewidth=7.0,color='b')

        plt.plot(x1,z2poly(x1),linewidth=7.0,color='r')

        plt.plot(x2,z3poly(x2),linewidth=7.0,color='g')

        labels=['0','1','2']

        plt.legend(labels=labels,bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)

        plt.ylabel('Life Expectancy')

        plt.xlabel(column)
cols=['Year','world','Status','Adult_Mortality','Infant_Deaths', 'Alcohol', 'Percent_Expenditure', 'Hep_B', 'Measles',

       'BMI', 'U_5_Deaths', 'Polio', 'Total_Expenditure', 'Diphtheria',

       'HIV_AIDS', 'Thinness_10_19', 'Thinness_5_9', 'Income_Composition',

       'Schooling']

for column in cols:

    Clusterplot(column)

    sns.set(rc={'figure.figsize':(10,10)})

    sns.set(font_scale=1.5)

    plt.show()
#Adjust Year

years=[2001, 2000, 2015, 2014, 2013, 2012, 2011, 2010, 2009, 2008, 2007,

       2006, 2005, 2004, 2003, 2002]

years.sort()



for year,i in zip(years,range(len(years))):

    df_reg['Year'][df_reg['Year'] == year]=i+1
# Function to calculate the intraclass correlation

def ICC(fittedmodel):

    between_var= fittedmodel.cov_re.iloc[0,0]

    resid=fittedmodel.scale

    icc=between_var/(between_var+resid)

    return icc
import statsmodels.api as sm

import statsmodels.formula.api as smf

from statsmodels.regression.mixed_linear_model import MixedLMParams

features=['Year','Country', 'Status', 'Adult_Mortality',

       'Infant_Deaths', 'Alcohol', 'Percent_Expenditure', 'Hep_B', 'Measles',

       'BMI', 'U_5_Deaths', 'Polio', 'Total_Expenditure', 'Diphtheria',

       'HIV_AIDS', 'Thinness_10_19', 'Thinness_5_9', 'Income_Composition',

       'Schooling', 'world']

for col,i in zip(features,range(len(features))):

    if i==0:

        # Model to use for calculating the ICC

        print('Model {}'.format(col))

        model = smf.mixedlm("Life_Expectancy ~ 1",data=df_reg,groups=df_reg[col])

        result = model.fit()

        print(result.summary())

        print('The Intraclass Correlation is: {:.3f}'.format(ICC(result)))

        print('Group Var: {:.2f}\n'.format(result.cov_re.iloc[0,0]))

        

        #DataFrame Columns

        LMresults=pd.DataFrame(index=range(len(features)))

        LMresults['Column']=col

        LMresults['ICC']=ICC(result).round(3)

        LMresults['Group Var']=result.cov_re.iloc[0,0].round(2)

    else:

        # Model to use for calculating the ICC

        print('Model {}'.format(col))

        model = smf.mixedlm("Life_Expectancy ~ 1",data=df_reg,groups=df_reg[col])

        result = model.fit()

        print(result.summary())

        print('The Intraclass Correlation is: {:.3f}'.format(ICC(result)))

        print('Group Var: {:.2f}\n'.format(result.cov_re.iloc[0,0]))

        

        #Add to DataFrame 

        LMresults.loc[i]=[col,ICC(result).round(3),result.cov_re.iloc[0,0].round(2)]     
LMresults
cols=['Year', 'Status', 'Adult_Mortality','Infant_Deaths', 'Alcohol', 'Percent_Expenditure', 'Hep_B', 'Measles',

       'BMI', 'U_5_Deaths', 'Polio', 'Total_Expenditure', 'Diphtheria',

       'HIV_AIDS', 'Thinness_10_19', 'Thinness_5_9', 'Income_Composition',

       'Schooling', 'world']

re=['-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-']



Y=df_reg['Life_Expectancy']

for col,r,i in zip(cols,re,range(len(re))): 

    if i==0:

        print('Running Random Intercepts')

        model = smf.mixedlm("Life_Expectancy~{}".format(col),data=df_reg,groups=df_reg['Country'])

        riresult = model.fit()

        print(riresult.summary())

        print('FE: {}'.format(col))

        print('The Intraclass Correlation is: {}'.format(ICC(riresult)))

        print('The Likelihood is: {}'.format(riresult.llf))

        print('RE Variance is {}'.format(r))

        

        #Predicted Values

        Y_pred = riresult.fittedvalues

        RMSE=mean_squared_error(Y, Y_pred)**0.5

        var=Y_pred.std()**2

        print('RMSE: {:0.3f}'.format(RMSE))

        print('Minimum LE: {:0.1f}'.format(Y_pred.min()))

        print('Maximum LE: {:0.1f}'.format(Y_pred.max()))

        print('Average Predicted LE: {:0.1f}'.format(Y_pred.mean()))

        print('LE Standard Deviation: {:0.3f}'.format(Y_pred.std()))

        print('LE Variance: {:0.3f}'.format(var),'\n')

        

        #DataFrame Columns

        MXLMresults=pd.DataFrame(index=range(11))

        MXLMresults['Model']='RI'

        MXLMresults['Fixed Effect']=col

        MXLMresults['Random Effect']=r

        MXLMresults['RE VAR']=riresult.cov_re.iloc[0,0].round(2)     

        MXLMresults['RMSE']=RMSE.round(3)

        MXLMresults['ICC']=ICC(riresult).round(3)

        MXLMresults['Likelihood']=riresult.llf.round(1)

        MXLMresults['LE pred Min']=Y_pred.min().round(1)

        MXLMresults['LE pred Max']=Y_pred.max().round(1)

        MXLMresults['LE pred Mean']=Y_pred.mean().round(1)

        MXLMresults['LE pred Std']=Y_pred.std().round(3)

        MXLMresults['LE pred Var']=var.round(3)

    else:  

        print('Running Random Intercepts Model')

        model = smf.mixedlm("Life_Expectancy~{}".format(col),data=df_reg,groups=df_reg['Country'])

        riresult = model.fit()

        print(riresult.summary())

        print('FE: {}'.format(col))

        print('The Intraclass Correlation is: {}'.format(ICC(riresult)))

        print('The Likelihood is: {}'.format(riresult.llf))

        print('RE Variance is {}'.format(r))



        #Predicted Values

        Y_pred = riresult.fittedvalues

        RMSE=mean_squared_error(Y, Y_pred)**0.5

        var=Y_pred.std()**2

        print('RMSE: {:0.3f}'.format(RMSE))

        print('Minimum LE: {:0.1f}'.format(Y_pred.min()))

        print('Maximum LE: {:0.1f}'.format(Y_pred.max()))

        print('Average Predicted LE: {:0.1f}'.format(Y_pred.mean()))

        print('LE Standard Deviation: {:0.3f}'.format(Y_pred.std()))

        print('LE Variance: {:0.3f}'.format(var),'\n')   

   

        #Add to results

        MXLMresults.loc[i]=['RI',col,r,riresult.cov_re.iloc[0,0].round(2)

                              ,RMSE.round(3),ICC(riresult).round(3),riresult.llf.round(1)

                              ,Y_pred.min().round(1),Y_pred.max().round(1),Y_pred.mean().round(1),Y_pred.std().round(3),var.round(3)]         
cols=['Status','world','Year','Status','world','Status','world','Year','Status','world','Year','Status'

      ,'world','Year','Status','world','Year']

re=['Year','Year','Status','Schooling','Schooling','HIV_AIDS','HIV_AIDS','HIV_AIDS','Income_Composition','Income_Composition','Income_Composition'

    ,'Alcohol','Alcohol','Alcohol','U_5_Deaths','U_5_Deaths','U_5_Deaths']



Y=df_reg['Life_Expectancy']

for col,r,i in zip(cols,re,range(len(re))): 

        print('Running Random Slopes Model')

        model = smf.mixedlm("Life_Expectancy~{}".format(col),data=df_reg,groups=df_reg['Country'],re_formula="~0+{}".format(r))

        rsresult = model.fit()

        print(rsresult.summary())

        print('FE: {}'.format(col))

        print('The Intraclass Correlation is: {}'.format(ICC(rsresult)))

        print('The Likelihood is: {}'.format(rsresult.llf))

        print('{} Variance is {}'.format(r,rsresult.cov_re.iloc[0,0].round(2)))



        #Predicted Values

        Y_pred = rsresult.fittedvalues

        RMSE=mean_squared_error(Y, Y_pred)**0.5

        var=Y_pred.std()**2

        print('RMSE: {:0.3f}'.format(RMSE))

        print('Minimum LE: {:0.1f}'.format(Y_pred.min()))

        print('Maximum LE: {:0.1f}'.format(Y_pred.max()))

        print('Average Predicted LE: {:0.1f}'.format(Y_pred.mean()))

        print('LE Standard Deviation: {:0.3f}'.format(Y_pred.std()))

        print('LE Variance: {:0.3f}'.format(var),'\n')   

   

        #Add to results

        MXLMresults.loc[i+19]=['RS',col,r,rsresult.cov_re.iloc[0,0].round(2)

                              ,RMSE.round(3),ICC(rsresult).round(3),rsresult.llf.round(1)

                              ,Y_pred.min().round(1),Y_pred.max().round(1),Y_pred.mean().round(1),Y_pred.std().round(3),var.round(3)]         
cols=['Status','world','Year','Status','world','Status','world','Year','Status','world','Year','Status'

      ,'world','Year','Status','world','Year']

re=['Year','Year','Status','Schooling','Schooling','HIV_AIDS','HIV_AIDS','HIV_AIDS','Income_Composition','Income_Composition','Income_Composition'

    ,'Alcohol','Alcohol','Alcohol','U_5_Deaths','U_5_Deaths','U_5_Deaths']



Y=df_reg['Life_Expectancy']

for col,r,i in zip(cols,re,range(len(re))): 

    print('Running Random Slopes+Intercepts Model')

    model = smf.mixedlm("Life_Expectancy~{}".format(col),data=df_reg,groups=df_reg['Country'],re_formula="~{}".format(r))

    risresult = model.fit()

    print(risresult.summary())

    print('FE: {}'.format(col))

    print('The Intraclass Correlation is: {}'.format(ICC(risresult)))

    print('The Likelihood is: {}'.format(risresult.llf))

    print('{} Variance is {}'.format(r,risresult.cov_re.iloc[1,1].round(2)))



    #Predicted Values

    Y_pred = risresult.fittedvalues

    RMSE=mean_squared_error(Y, Y_pred)**0.5

    var=Y_pred.std()**2

    print('RMSE: {:0.3f}'.format(RMSE))

    print('Minimum LE: {:0.1f}'.format(Y_pred.min()))

    print('Maximum LE: {:0.1f}'.format(Y_pred.max()))

    print('Average Predicted LE: {:0.1f}'.format(Y_pred.mean()))

    print('LE Standard Deviation: {:0.3f}'.format(Y_pred.std()))

    print('LE Variance: {:0.3f}'.format(var),'\n')   

   

    #Add to results

    MXLMresults.loc[i+36]=['RIS',col,r,risresult.cov_re.iloc[1,1].round(2)

                              ,RMSE.round(3),ICC(risresult).round(3),risresult.llf.round(1)

                              ,Y_pred.min().round(1),Y_pred.max().round(1),Y_pred.mean().round(1),Y_pred.std().round(3),var.round(3)]            

      
MXLMresults
Y=df_reg['Life_Expectancy']

print('Running Random Intercepts Model')

model = smf.mixedlm("Life_Expectancy~Year",data=df_reg,groups=df_reg['Country'])

riresult = model.fit()

print(riresult.summary())

print('The Intraclass Correlation is: {:.3f}'.format(ICC(riresult)),'\n')



print('Running Random Slopes Model')

model = smf.mixedlm("Life_Expectancy~Status",data=df_reg,groups=df_reg['Country'],re_formula="~0+Income_Composition")

rsresult = model.fit()

print(rsresult.summary())

print('The Intraclass Correlation is: {:.3f}'.format(ICC(rsresult)),'\n')



print('Running Random Slopes+Intercepts Model')

model = smf.mixedlm("Life_Expectancy~Status",data=df_reg,groups=df_reg['Country'],re_formula="~Year")

risresult = model.fit()

print(risresult.summary())

print('The Intraclass Correlation is: {:.3f}'.format(ICC(risresult)),'\n')



from scipy.stats import chi2

# Double Check Model Selection

def likelihood_ratio_test(bigmodel, smallmodel):

    likelihoodratio=2*(bigmodel.llf-smallmodel.llf)

    f=bigmodel.df_modelwc-smallmodel.df_modelwc

    p=chi2.sf(likelihoodratio, f)

    return p



lrt=likelihood_ratio_test(risresult,rsresult)

print('The p-value for the likelihood ratio test of the random slope and random intercept/slope models is: {:.4f}'.format(lrt))



lrt=likelihood_ratio_test(risresult,riresult)

print('The p-value for the likelihood ratio test of the random intercept and random intercept/slope models is: {:.4f}'.format(lrt))
# Use as 2nd hue

df_reg['Cluster']=df_cl['KM Cluster']
df_reg['residual']=risresult.resid



# Are residuals normally distributed?

plt.hist(df_reg['residual'])

sns.set(rc={'figure.figsize':(10,10)})

print(stats.shapiro(df_reg['residual']))

plt.show()



# Is variance constant for all values of the outcome?

sns.scatterplot(x='Life_Expectancy',y='residual',hue='Status',data=df_reg)

plt.title('Residuals by raw LE values with cluster labels')

sns.set(rc={'figure.figsize':(10,10)})

plt.show()



# Is variance constant for all values of the outcome?

sns.scatterplot(x='Life_Expectancy',y='residual',hue='Cluster',data=df_reg,palette='Set2')

plt.title('Residuals by raw LE values with cluster labels')

sns.set(rc={'figure.figsize':(10,10)})

plt.show()





# Is variance constant for all values of the predictors?

sns.scatterplot(x='Year',y='residual',hue='Status',data=df_reg)

plt.title('Residuals by Year')

sns.set(rc={'figure.figsize':(10,10)})

plt.show()





# Is variance constant for all values of the predictors?

sns.scatterplot(x='Year',y='residual',hue='Cluster',data=df_reg,palette='Set2')

plt.title('Residuals by Year with Cluster Labels')

sns.set(rc={'figure.figsize':(10,10)})

plt.show()

#plot out best with Cluster Labels

#Predictions

Y_pred=rf.predict(scale(df_reg.drop(['Life_Expectancy','Country','world','Status','residual','Cluster'],1)))

Y=df_reg['Life_Expectancy']                 

#plot

z1 =np.polyfit(Y_pred,Y,1)

z1poly = np.poly1d(z1) 

plt.scatter(Y_pred,Y,alpha=1,c=df_reg['Cluster'],cmap='Set2')

plt.plot(Y_pred,z1poly(Y_pred),linewidth=7.0,color='r')

plt.title('Gradient Boost {}'.format(z))

plt.xlabel('Y_pred')

plt.ylabel('Y')

plt.show()
print(risresult.summary())

Y=df_reg['Life_Expectancy']

Y_pred = risresult.fittedvalues

RMSE=mean_squared_error(Y, Y_pred)**0.5

print('RMSE: {:0.3f}'.format(RMSE))

print('Minimum LE: {:0.1f}'.format(Y_pred.min()))

print('Maximum LE: {:0.1f}'.format(Y_pred.max()))

print('Average Predicted LE: {:0.1f}'.format(Y_pred.mean()))

print('LE Standard Deviation: {:0.3f}'.format(Y_pred.std()))

print('LE Variance: {:0.3f}'.format(Y_pred.std()**2))

        

#plot

z1 =np.polyfit(Y_pred,Y,1)

z1poly = np.poly1d(z1)

plt.scatter(Y_pred,Y,c=df_reg['Cluster'],alpha=1,cmap='Set2')

plt.plot(Y_pred,z1poly(Y_pred),linewidth=7.0,color='r')

plt.title('Mixed Effect RIS')

plt.xlabel('Y_pred')

plt.ylabel('Y')

plt.show()            
df_reg['Life_Expectancy'].describe()