import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
plt.style.use('bmh') # Select bmh Plotting Style
plt.rcParams['figure.figsize'] = (11.0, 9.0)
cm = plt.cm.get_cmap('RdYlBu')
df = pd.read_csv("../input/database.csv")
df.head()
df.loc[df.birth_year == '530s','birth_year'] = 530
df.loc[df.birth_year == '1237?','birth_year'] = 1237
df.birth_year.replace("Unknown",'',inplace=True)
df['years_since_birth'] = (2018 - pd.to_numeric(df.birth_year))
_ = df.industry.value_counts().plot(kind='barh',title='Industry')
_ = df.domain.value_counts().plot(kind='barh',title='Domain')
_ = df.occupation.value_counts().nlargest(15).plot.barh(figsize=(11,11),title='Occupation (Top 15)')
_ = df.country.value_counts().nlargest(25).plot.barh(figsize=(11,11),title='Country (Top 25)')
df_sample = df.sample(2000)
_ = df_sample.historical_popularity_index.plot.kde(title='Distribution of Historical Popularity Index')
_ = df_sample.average_views.plot.hist(title='Distribution of Average Views',logx=True,figsize=(11,8))
_ = df_sample.plot.scatter('historical_popularity_index','average_views',loglog=True,
                    title='Historical Popularity Index by Average Views',figsize=(11,11))
def graph3way(x_col, y_col, hue_col, log='x', fsize=(14,11)):
    if log == '':
        set_logx = False
        set_logy = False
    elif log == 'x':
        set_logx = True
        set_logy = False
    elif log == 'y':
        set_logx = False
        set_logy = True
    else:
        set_logx = True
        set_logy = True
    groups = df[hue_col].unique()
    colors = [cm(i) for i in np.linspace(0.0,1.0,groups.size)]
    _d = {d:colors[i] for i,d in enumerate(groups)}
    fig, ax = plt.subplots(figsize=fsize)
    for d,row in df_sample.groupby(hue_col):
        row.plot.scatter(x=x_col,y=y_col,c=_d[d],logx=set_logx,logy=set_logy,label=d,ax=ax)
    _ = ax.legend()
    _ = ax.set_title('{} by {} and {}'.format(y_col,x_col,hue_col))
                     
graph3way('years_since_birth','historical_popularity_index','domain')
# Alternatively
import seaborn as sns
g = sns.lmplot(x="years_since_birth", y="historical_popularity_index", data=df_sample, fit_reg=False, hue='domain')
g.fig.set_size_inches(14,11)
_ = g.set(xscale="log")
graph3way('years_since_birth','historical_popularity_index','sex')
graph3way('years_since_birth','historical_popularity_index','continent')
graph3way('article_languages','historical_popularity_index','domain',log='x')
graph3way('article_languages','page_views','domain',log='xy')
# creating dummy variables for the columns that were objects
cat_attributes = ['sex','country','continent','occupation','industry','domain']
data_dummies = pd.get_dummies(df[cat_attributes])
#add numerical columns and drop "article_id(column 0)  & city (column 4) & state (column 5)
pan = df.drop(cat_attributes+['city','state','article_id'], axis=1)
pan = pd.concat([pan, data_dummies], axis=1)
pan.set_index('full_name',inplace=True)
pan.head()
_ = df_sample.article_languages.plot.kde()
df.article_languages.describe()
# Right Skewed Distribution
corr_matrix=pan.corr()
#what columns are correlated to the popularity index
tmp = corr_matrix['article_languages'].sort_values(ascending=False)
_ = tmp[np.abs(tmp) > .1].plot.barh(figsize=(14,7))
def get_fit_summary(X_train,X_test,y_train,y_test,model):
    fmt = '{:7} R**2 = {:.2%}'
    print('\n\n\nModel: {}'.format(model))
    print(fmt.format('Train',model.score(X_train,y_train)))
    print(fmt.format('Test',model.score(X_test,y_test)))

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import export_graphviz, DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso


pan = pan.dropna()
y = pan['article_languages']
X = pan.drop(['article_languages','birth_year','longitude','historical_popularity_index','latitude'], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=2018)


regr_1 = DecisionTreeRegressor(max_depth=2,min_samples_leaf=.01)
regr_2 = DecisionTreeRegressor(max_depth=4,min_samples_leaf=.01)
regr_ols = LinearRegression()
regr_enet = ElasticNet(alpha = .1)
regr_lasso = Lasso(alpha = .15)

regr_1.fit(X_train, y_train)
regr_2.fit(X_train, y_train)
regr_ols.fit(X_train, y_train)
regr_enet.fit(X_train, y_train)
regr_lasso.fit(X_train, y_train)


y_1 = regr_1.predict(X_test)
y_2 = regr_2.predict(X_test)
y_ols = regr_ols.predict(X_test)
y_enet = regr_enet.predict(X_test)
y_lasso = regr_lasso.predict(X_test)
print('REGRESSION MODELS')
get_fit_summary(X_train,X_test,y_train,y_test,regr_ols)
get_fit_summary(X_train,X_test,y_train,y_test,regr_enet)
get_fit_summary(X_train,X_test,y_train,y_test,regr_lasso)
regr_coefficients = pd.DataFrame({'ols':regr_ols.coef_,'enet':regr_enet.coef_,'lasso':regr_lasso.coef_},
                                 index = X.columns)
_ = regr_coefficients[regr_coefficients.lasso > 1e-6]['lasso'].sort_values().plot.barh(title='Lasso',figsize=(11,14))
_ = regr_coefficients[regr_coefficients.enet > 1e-6]['enet'].sort_values().plot.barh(title='Elastic Net',figsize=(11,17))
import graphviz 
dot_data = export_graphviz(regr_1, out_file=None, feature_names=X.columns) 
print('R-sq = {:.2%}'.format(metrics.r2_score(y_test,y_1)))
graphviz.Source(dot_data)
# Alternative Regression Tree with Bigger Depth
print('R-sq = {:.2%}'.format(metrics.r2_score(y_test,y_2)))
y_2_unique = pd.Series(pd.unique(y_2))
dot_data = export_graphviz(regr_2, out_file=None, feature_names=X.columns) 
graphviz.Source(dot_data)