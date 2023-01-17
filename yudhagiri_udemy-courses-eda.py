import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
import cufflinks as cf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

%matplotlib inline
sns.set_style('darkgrid')
cf.go_offline()
df = pd.read_csv('../input/udemy-courses/udemy_courses.csv')
df.head(3)
df.columns
df.shape
df.info
df.describe()
sns.heatmap(df.isnull(), yticklabels=False, cbar=False)
plt.figure(figsize=(8,6))
sns.countplot(df['subject'], palette='magma')
plt.figure(figsize=(8,6))
s = df.groupby('subject').sum()['num_subscribers']
s.plot(kind='barh', color='c')
plt.ylabel('Subject')
plt.xlabel('Num of Subscribers')
plt.ticklabel_format(axis='x', style='plain')
plt.show()
web_ratio = round((df[df['subject']=='Web Development']['num_subscribers'].sum()/df[df['subject']=='Web Development']['course_title'].count()))
bus_ratio = round((df[df['subject']=='Business Finance']['num_subscribers'].sum()/df[df['subject']=='Business Finance']['course_title'].count()))

def gcd(a,b):
    """ Greatest common divisor """
    while b!=0:
        r=a%b
        a,b=b,r
    return a

a= int(web_ratio/gcd(web_ratio,bus_ratio))
b= int(bus_ratio/gcd(web_ratio,bus_ratio))

print('Subscribers per course ratio for WebDev and Busfin:')
print(a, ':', b)
df['published_timestamp'] = pd.to_datetime(df['published_timestamp'])
subject = df['subject'].unique()
growth = df[['published_timestamp', 'subject']]
growth = growth.sort_values('published_timestamp')
time_series = growth['published_timestamp'].value_counts().reset_index()
time_series.columns = ['Date', 'Counts']
time_series['Cummulative'] = time_series['Counts'].cumsum()
dummies = pd.get_dummies(growth['subject'])

growth = growth.join(dummies)
growth['cum_busfin'] = growth['Business Finance'].cumsum()
growth['cum_grdes'] = growth['Graphic Design'].cumsum()
growth['cum_music'] = growth['Musical Instruments'].cumsum()
growth['cum_webdev'] = growth['Web Development'].cumsum()
growth_melt = growth.melt(id_vars='published_timestamp', value_vars=['cum_busfin', 'cum_grdes', 'cum_music', 'cum_webdev'])
fig = make_subplots(rows=1, cols=1 )
fig.append_trace(go.Scatter
                 (x=growth['published_timestamp'], y=growth['cum_busfin'],  
                  name='Business Finance',  line=dict(color="#345feb")
                 ), row=1, col=1)
fig.append_trace(go.Scatter
                 (x=growth['published_timestamp'], y=growth['cum_grdes'],  
                  name='Graphic Design', line=dict(color="#7deb34")
                 ), row=1, col=1)
fig.append_trace(go.Scatter
                 (x=growth['published_timestamp'], y=growth['cum_music'],  
                  name='Musical Instrument',line=dict(color="#eb5f34")
                 ), row=1, col=1)
fig.append_trace(go.Scatter
                 (x=growth['published_timestamp'], y=growth['cum_webdev'],  
                  name='Web Development', line=dict(color="#e8eb34")
                ), row=1, col=1)
fig['layout'].update(height=500, width=1000, title='Number of Courses Growth per Subject')
fig.show()
milestones =pd.DataFrame({'published_timestamp':['2010-01-01', '2011-10-01', '2012-12-01', '2013-04-01', '2014-01-01'],
                         'milestones':['Founded', 'Series A funding', 'Series B funding', 'ios App launched', 'Android App launched']})
milestones['published_timestamp'] = pd.to_datetime(milestones['published_timestamp'], utc=True)
milestones.head()
growth_merged = pd.merge(growth, milestones, on='published_timestamp', how='outer')

def milestone_values(col):
    milestone= col[0]
    
    if milestone=='Founded' or milestone=='Series A funding' or milestone=='Series B funding' or milestone=='ios App launched' or milestone=='Android App launched':
        return 0
    
growth_merged['milestone_value']=growth_merged[['milestones']].apply(milestone_values, axis=1)
fig = make_subplots(rows=1, cols=1 )
fig.append_trace(go.Scatter
                 (x=growth_merged['published_timestamp'], y=growth_merged['cum_busfin'],  
                  name='Business Finance',  line=dict(color="#345feb")
                 ), row=1, col=1)
fig.append_trace(go.Scatter
                 (x=growth_merged['published_timestamp'], y=growth_merged['cum_grdes'],  
                  name='Graphic Design', line=dict(color="#7deb34")
                 ), row=1, col=1)
fig.append_trace(go.Scatter
                 (x=growth_merged['published_timestamp'], y=growth_merged['cum_music'],  
                  name='Musical Instrument',line=dict(color="#eb5f34")
                 ), row=1, col=1)
fig.append_trace(go.Scatter
                 (x=growth_merged['published_timestamp'], y=growth_merged['cum_webdev'],  
                  name='Web Development', line=dict(color="#e8eb34")
                ), row=1, col=1)
fig.append_trace(go.Scatter
                 (x=growth_merged[growth_merged['milestones']=='Founded']['published_timestamp'], 
                  y=growth_merged[growth_merged['milestones']=='Founded']['milestone_value'],
                  mode='markers',
                  name='Founded',
                  hovertext='Udemy was founded in 2010',
                  marker_size=10,
                ), row=1, col=1)
fig.append_trace(go.Scatter
                 (x=growth_merged[growth_merged['milestones']=='Series A funding']['published_timestamp'], 
                  y=growth_merged[growth_merged['milestones']=='Series A funding']['milestone_value'],
                  mode='markers',
                  name='Series A Funding',
                  hovertext='Udemy received Series A funding',
                  marker_size=10,
                ), row=1, col=1)
fig.append_trace(go.Scatter
                 (x=growth_merged[growth_merged['milestones']=='Series B funding']['published_timestamp'], 
                  y=growth_merged[growth_merged['milestones']=='Series B funding']['milestone_value'],
                  mode='markers',
                  name='Series B Funding',
                  hovertext='Udemy received Series B funding',
                  marker_size=10,
                ), row=1, col=1)
fig.append_trace(go.Scatter
                 (x=growth_merged[growth_merged['milestones']=='ios App launched']['published_timestamp'], 
                  y=growth_merged[growth_merged['milestones']=='ios App launched']['milestone_value'],
                  mode='markers',
                  name='ios App Launched',
                  hovertext='ios app launched to make learning more portable for ios users',
                  marker_size=10,
                ), row=1, col=1)
fig.append_trace(go.Scatter
                 (x=growth_merged[growth_merged['milestones']=='Android App launched']['published_timestamp'], 
                  y=growth_merged[growth_merged['milestones']=='Android App launched']['milestone_value'],
                  mode='markers',
                  name='Android App Launched',
                  hovertext='Android app launched to make learning more portable for android users',
                  marker_size=10,
                ), row=1, col=1)

fig['layout'].update(height=500, width=1000, title='Number of Courses Growth per Subject and Udemy Milestones Event')
fig.show()
plt.figure(figsize=(8,6))
sns.boxplot(x='subject', y='price', data=df, palette='magma')
print('Top 5 most expensive WebDev course (USD):')

idx = pd.IndexSlice
web_dev_pricey = df.loc[idx[df[df['subject']=='Web Development']['price'].sort_values(ascending=False).index]][['course_title', 'price', 'num_subscribers']]
web_dev_pricey['gross income'] = web_dev_pricey['price'] * web_dev_pricey['num_subscribers']
web_dev_pricey['organic_profit'] = web_dev_pricey['price'] * web_dev_pricey['num_subscribers']*0.5
web_dev_pricey['promotion_profit'] = web_dev_pricey['price'] * web_dev_pricey['num_subscribers']*0.97
web_dev_pricey.head(5)
print('Top 5 most expensive Bus&Fin course (USD):')

idx = pd.IndexSlice
busfin_pricey = df.loc[idx[df[df['subject']=='Business Finance']['price'].sort_values(ascending=False).index]][['course_title', 'price', 'num_subscribers']]
busfin_pricey['gross income'] = busfin_pricey['price'] * busfin_pricey['num_subscribers']
busfin_pricey['organic_profit'] = busfin_pricey['price'] * busfin_pricey['num_subscribers']*0.5
busfin_pricey['promotion_profit'] = busfin_pricey['price'] * busfin_pricey['num_subscribers']*0.97
busfin_pricey.head(5)
print('Top 5 most expensive Graphic&Design course (USD):')

idx = pd.IndexSlice
grdes_pricey = df.loc[idx[df[df['subject']=='Graphic Design']['price'].sort_values(ascending=False).index]][['course_title', 'price', 'num_subscribers']]
grdes_pricey['gross income'] = grdes_pricey['price'] * grdes_pricey['num_subscribers']
grdes_pricey['organic_profit'] = grdes_pricey['price'] * grdes_pricey['num_subscribers']*0.5
grdes_pricey['promotion_profit'] = grdes_pricey['price'] * grdes_pricey['num_subscribers']*0.97
grdes_pricey.head(5)
print('Top 5 most expensive Musical course (USD):')

idx = pd.IndexSlice
music_pricey = df.loc[idx[df[df['subject']=='Musical Instruments']['price'].sort_values(ascending=False).index]][['course_title', 'price', 'num_subscribers']]
music_pricey['gross income'] = music_pricey['price'] * music_pricey['num_subscribers']
music_pricey['organic_profit'] = music_pricey['price'] * music_pricey['num_subscribers']*0.5
music_pricey['promotion_profit'] = music_pricey['price'] * music_pricey['num_subscribers']*0.97
music_pricey.head(5)
webdev_mostprofit = web_dev_pricey.loc[idx[web_dev_pricey['organic_profit'].sort_values(ascending=False).index]][['course_title', 'organic_profit']]
print('Top 5 most profitable WebDev courses:')
webdev_mostprofit.head()
busfin_mostprofit = busfin_pricey.loc[idx[busfin_pricey['organic_profit'].sort_values(ascending=False).index]][['course_title', 'organic_profit']]
print('Top 5 most profitable Business and Finance courses:')
busfin_mostprofit.head()
grdes_mostprofit = grdes_pricey.loc[idx[grdes_pricey['organic_profit'].sort_values(ascending=False).index]][['course_title', 'organic_profit']]
print('Top 5 most profitable Graphic Design courses:')
grdes_mostprofit.head()
music_mostprofit = music_pricey.loc[idx[music_pricey['organic_profit'].sort_values(ascending=False).index]][['course_title', 'organic_profit']]
print('Top 5 most profitable Musical Instrument courses:')
music_mostprofit.head()
df_profit = df
df_profit['organic_profit'] = df['num_subscribers'] * df['price']*0.5
df_profit = df_profit.loc[idx[df_profit['organic_profit'].sort_values(ascending=False).index]][['course_title', 'subject', 'organic_profit']]

f=df_profit.head()
figs = make_subplots(rows=1, cols=1)
figs.append_trace(go.Bar(
    x=f['organic_profit'],
    y=f['course_title'],
    orientation='h',
    marker = dict(color=f['organic_profit'].values, coloraxis='coloraxis')
    ), row=1, col=1)
figs['layout'].update(height=600, width=1000, title='Top 5 Profitable Courses (in USD)')
figs.show()
split = df['is_paid'].value_counts().reset_index()
split.columns = ['Is Paid', 'Counts']
fig = px.pie(split, names='Is Paid', values='Counts', width=500)
fig['layout'].update(title='Paid/Free Course Pie Chart')
fig.show()
paycorr = df[['price', 'num_subscribers', 'num_reviews', 'num_lectures', 'level']].corr()
plt.figure(figsize=(8,6))
sns.heatmap(paycorr, cmap='magma', annot=True)
popular = df.loc[idx[df['num_subscribers'].sort_values(ascending=False).index]][['course_title', 'num_subscribers', 'subject']]
mostpop = popular.head(10)
engage = df.loc[idx[df['num_reviews'].sort_values(ascending=False).index]][['course_title', 'num_reviews', 'subject']]
mosten= engage.head(10)
fig = make_subplots(rows=2 ,cols=1, subplot_titles=('Most Popular Courses', 'Most Engaging Courses'))
fig.append_trace(go.Bar(x=mostpop['num_subscribers'], y=mostpop['course_title'],
                       orientation='h', 
                       showlegend=False), row=1, col=1)
fig.append_trace(go.Bar(x=mosten['num_reviews'], y=mosten['course_title'],
                       orientation='h', 
                       showlegend=False), row=2, col=1)
fig['layout'].update(height=700, width=800)
fig.show()
df_level = df[['level', 'num_subscribers', 'course_title']]
subs_level = df_level.groupby('level').sum().reset_index()
course_level = df_level.groupby('level').count().reset_index()

fig = make_subplots(rows=2 ,cols=1, subplot_titles=('Levels v Subscribers', 'Levels v Number of Courses'))
fig.append_trace(go.Bar(x=subs_level['num_subscribers'], y=subs_level['level'],
                       orientation='h', 
                       showlegend=False), row=1, col=1)
fig.append_trace(go.Bar(x=course_level['course_title'], y=course_level['level'],
                       orientation='h', 
                       showlegend=False), row=2, col=1)
fig['layout'].update(height=700, width=800)
fig.show()
norm_level = pd.DataFrame(subs_level['num_subscribers']/course_level['course_title'])
norm_level = pd.concat([subs_level['level'], norm_level], axis=1)
norm_level.columns = ['level', 'subs per course']
fig =px.bar(data_frame=norm_level,x='subs per course', y='level', color='level', title='Subs per Course v Level')
fig.show()
fig=px.box(df, x='level', y='content_duration', color='level', title='Box Plot of Level v Content Duration')
fig.show()
def level_category(cols):
    level=cols[0]
    
    if level =='All Levels':
        return 1
    elif level == 'Beginner Level':
        return 2
    elif level == 'Intermediate Level':
        return 3
    elif level == 'Expert level':
        return 4
    
df['level_category'] = df[['level']].apply(level_category, axis=1)

def ispaid(cols):
    ispaid  = cols[0]
    
    if ispaid ==True:
        return 1
    else:
        return 0
df['paid'] = df[['is_paid']].apply(ispaid, axis=1)


def subject_cat(cols):
    subject =cols[0]
    
    if subject == 'Web Development':
        return 1
    elif subject =='Business Finance':
        return 2
    elif subject =='Graphic Design':
        return 3
    else:
        return 4

df['subject_cat'] = df[['subject']].apply(subject_cat, axis=1)

df.head()
pairdata= df.drop(['course_id', 'course_title', 'url', 'is_paid', 'level', 'subject', 'published_timestamp'], axis=1)
paircorr = pairdata.corr()
plt.figure(figsize=(10,6))
sns.heatmap(paircorr, cmap ='coolwarm', annot=True)
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
pairdata.head()
pairdata_fixed = pairdata
pairdata_fixed.dropna(inplace=True)
X = pairdata_fixed.drop(['organic_profit', 'num_subscribers'], axis=1)
y = pairdata_fixed['num_subscribers']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
linearmodel = LinearRegression()
linearmodel.fit(X_train, y_train)
linear_pred = linearmodel.predict(X_test)
linear_result = pd.DataFrame({'predicted_subs':linear_pred,
                          'actual_subs': y_test.reset_index()['num_subscribers']})
fig = px.scatter(linear_result, x='actual_subs', y='predicted_subs')
fig.show()
metrics.mean_absolute_error(y_test, linear_pred)
np.sqrt(metrics.mean_squared_error(y_test, linear_pred)) 
rfor = RandomForestRegressor(n_estimators=500, random_state=101)
scaler = StandardScaler()
X = pairdata_fixed.drop(['organic_profit', 'num_subscribers'], axis=1)
y = pairdata_fixed['num_subscribers']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
scaler = scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
rfor.fit(X_train, y_train)
rfor_pred = rfor.predict(X_test)
forest_result = pd.DataFrame({'predicted_subs':rfor_pred,
                          'actual_subs': y_test.reset_index()['num_subscribers']})
fig = px.scatter(forest_result, x='actual_subs', y='predicted_subs')
fig.show()
metrics.mean_absolute_error(y_test, rfor_pred)
np.sqrt(metrics.mean_squared_error(y_test, rfor_pred)) 
feature = pd.Series(rfor.feature_importances_, index=X.columns)
px.bar(x=feature.index, y=feature.values, labels={'x': 'Feature', 'y':'Importance Score'},
      color=feature.index,title='Feature Importance for RandomForestRegression')
print('Linear Regression Feature Importance Score:')
feature2= pd.Series(linearmodel.coef_, index=X.columns)
feature2
