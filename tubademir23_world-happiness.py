import pandas as pd 
import os 
column_dictionary={'Country or region':'Country','Country':'Country',
'Happiness.Rank':'Rank','Overall rank':'Rank','Happiness Rank':'Rank',
'Region':'Region', 'Family':'Family',
'Happiness.Score':'Score', 'Happiness Score': 'Score', 'Score':'Score', 'Generosity':'Generosity','Standard Error':'Standard_Error',
'Economy..GDP.per.Capita.':'GDP', 'Economy (GDP per Capita)':'GDP', 'GDP per capita':'GDP', 
'Health (Life Expectancy)':'Healthy_life_expectancy', 'Health..Life.Expectancy.':'Healthy_life_expectancy','Healthy life expectancy':'Healthy_life_expectancy',
'Dystopia.Residual':'Dystopia_Residual', 'Dystopia Residual':'Dystopia_Residual','Whisker.high':'Whisker.high', 'Whisker.low':'Whisker.low', 'Social support':'Social_Support',
'Freedom to make life choices':'Freedom', 'Freedom':'Freedom', 'Lower Confidence Interval':'LCI', 'Upper Confidence Interval':'UCI',
'Perceptions of corruption':'TGC','Trust (Government Corruption)':'TGC', 'Trust..Government.Corruption.':'TGC'
}
      
from pathlib import Path
datas=[]
years=[]
csv_files = [csvfile for csvfile in Path("/kaggle/input/world-happiness/").iterdir() if csvfile.is_file() and csvfile.suffix == '.csv']
for file in csv_files:
    directory, filename = os.path.split(file)
    data = pd.read_csv(file)
    data.columns = [column_dictionary[x] for x in data.columns]
    year=int( filename.split('.')[0])
    data['Year']=year  
    years.append(year)
    datas.append(data)
years.sort()
df=pd.concat(datas).sort_values(by=['Year','Rank'])
df.set_index(['Country','Year'])

df_turkey=df[df.Country=='Turkey']

features=['Freedom','GDP', 'Generosity','Healthy_life_expectancy','TGC']
predicted_feature=['Score']
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
sns.set_palette("RdBu")

correlation=df_turkey[features+predicted_feature].corr()
sns.heatmap(correlation, annot=True)
plt.show()
sns.pairplot(df_turkey[features+predicted_feature], kind='reg')
import statistics
fig, ax = plt.subplots(1,2, figsize=(20, 6))
median_year=statistics.median(years)

correlation_first_half=df_turkey[df_turkey.Year<=median_year][features+predicted_feature].corr()
ax[0].set_title(f'[{years[0]}-{median_year}] years')
sns.heatmap(correlation_first_half, annot=True, ax=ax[0])

correlation_second_half=df_turkey[df_turkey.Year>=median_year][features+predicted_feature].corr()
ax[1].set_title(f'[{median_year}-{years[-1]}] years')
sns.heatmap(correlation_second_half, annot=True, ax=ax[1])
plt.show()
from sklearn import linear_model

X = df_turkey[features] 
y = df_turkey[predicted_feature]

regr = linear_model.LinearRegression()
regr.fit(X, y)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

fig=sns.barplot(y=features, x=regr.coef_[0])
plt.show()
import matplotlib.pyplot as plt
colors=["#648FFF","#DC267F","#FFB000","#BBFF00","#00B0FF","#00B000"]
print(features)
print(years)
def plot_features(df_, features_, regr_, x_):
    fig, axs = plt.subplots(1,len(features_), figsize=(25, 6))
    for i in range(len(features_)):
        axs[i].plot( df_[x_],df_[features_[i]], colors[i])
        axs[i].set_title(features_[i]+"\nCoefficient : "+ "%.6f"% regr_.coef_[0][i])
        if x_!= 'Year':
            axs[i].set_xticklabels(df_[x_], rotation=60)
plot_features(df_turkey, features, regr,'Year')
year=years[-1]
df_year=df[df.Year==year]

happies_country=df_year[df_year.Rank==1][['Country','Score','Rank']]
print(f'The happiest Country in {year} is: {happies_country.Country.values} with {happies_country.Score.values} score.')
#Add Social_Support to the before features
features_year=features+['Social_Support']
sns.set_palette("RdBu")

correlation=df_year[features_year+predicted_feature].corr()
sns.heatmap(correlation, annot=True)
plt.show()
sns.pairplot(df_year[features_year+predicted_feature], kind='reg')
from sklearn import linear_model

X = df_year[features_year]
y = df_year[predicted_feature]

regr = linear_model.LinearRegression()
regr.fit(X, y)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)
print('Score: \n', regr.score(X,y))
fig=sns.barplot(y=features_year, x=regr.coef_[0])

plt.show()
plot_features(df_year[df_year.Rank<=5], features_year, regr,'Country')
import statsmodels.api as sm
fig, ax = plt.subplots(ncols=6, sharey=True, figsize=(25,6))
i=0
for feature in features_year:
    X_f=sm.add_constant(df_year[feature])
    lm=sm.OLS(y, X_f)
    model=lm.fit()
    print(f'{feature.ljust(25)}: MSE: {model.mse_model}\t R_Squared: {model.rsquared}\t R_Squared_Adj: {model.rsquared_adj}')
    sub_function= format("%.4f" % model.params[0]) + ' + ' + feature+ "*"+format("%.4f" % model.params[1])
    #print('Lineer Function of the Score='+sub_function+'\n')
    #print(y.merge(pd.DataFrame(pd.DataFrame(model.fittedvalues)), left_index=True, right_index=True))
    g=sns.regplot(df_year[feature], y, ci=None,  ax=ax[i])
    g.set_title(sub_function)
    g.set_ylabel('Score')
    g.set_xlabel(feature)
    plt.ylim(bottom=0)
    i=i+1
 

    
import functools, operator
lm=sm.OLS(y, X)
model=lm.fit()

str_function= 'Function of the Score='+format("%.4f" % model.params[0])+' + '+' + '.join([ format( x)+"*"+ format("%.4f" % model.params[x])+' '  for x in features_year])
print('With single model Lineer '+str_function)
model.summary()