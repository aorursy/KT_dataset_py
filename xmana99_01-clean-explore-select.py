import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df = pd.read_csv('../input/churn_working.csv')
df.head()
df.info()
df['Int.l.Plan'] = df['Int.l.Plan']=='yes'
df['VMail.Plan'] = df['VMail.Plan']=='yes'
df['Churn'] = df['Churn']=='yes'
df.info()
df.describe()
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import cufflinks as cf

# For Notebooks
init_notebook_mode(connected=True)

# For offline use
cf.go_offline()

# For cloropeths
import plotly.graph_objs as go
def to_int(churn): 
    if churn == True:
        return 1
    else:
        return 0

df['Churn_int'] = df['Churn'].apply(to_int)
byState = df.groupby(by='State').mean()['Churn_int']
byState = byState.reset_index()
byState.head()
data = dict(type='choropleth',
            colorscale = 'YIOrRd',
            locations = byState['State'],
            z = byState['Churn_int'],
            locationmode = 'USA-states',
            text = df['State'],
            marker = dict(line = dict(color = 'rgb(255,255,255)',width = 2)),  # borders across countries
            colorbar = {'title':"Churn Ratio"}
            ) 

layout = dict(title = 'Churn',
              geo = dict(scope='usa',
                         showlakes = True,
                         lakecolor = 'rgb(85,173,240)')
             )

choromap = go.Figure(data = [data],layout = layout)

iplot(choromap)
df = df.drop(['State','Churn_int'],axis=1)
plt.figure(figsize=(12,6))
sns.heatmap(df.corr(),annot=True,cmap='YlGn',lw=2)
df = df.drop(['VMail.Message','Day.Charge','Intl.Charge'],axis=1)
df = df.drop(['Account.Length','Area.Code','Day.Calls','Night.Calls'],axis=1)
df = df.drop('Phone',axis=1)
df.info()
sns.pairplot(data=df)
df['Churn'].value_counts()
sns.countplot(x=df['Churn'])
plt.figure(figsize=(12,6))
sns.heatmap(abs(df.corr()),annot=True,cmap='YlGn',lw=2)
sns.countplot(x='Int.l.Plan',data=df,hue='Churn')
plt.title('Total Int.l.Plan by Churn')
sns.barplot(x=df['Int.l.Plan'],y=df['Churn'])
plt.title('Mean Churn by Int.l.Plan')
df['Int.l.Plan'].value_counts()
sns.distplot(df[df['Churn']==False]['CustServ.Calls'],kde=False,label='Churn=no',bins=10)
sns.distplot(df[df['Churn']==True]['CustServ.Calls'],kde=False,label='Churn=yes',bins=10)
plt.title('CustServ.Calls by Churn')
plt.legend()
sns.barplot(x='Churn',y='CustServ.Calls',data=df,estimator=sum)
plt.title('Total CustServ.Calls by Churn')
sns.distplot(df[df['Churn']==True]['CustServ.Calls'],kde=False,label='Churn=yes', bins=10, color='green')
plt.legend()
sns.distplot(df[df['Churn']==False]['CustServ.Calls'],kde=False,label='Churn=no', bins=10, color='blue')
plt.legend()
sns.barplot(x='Churn', y='CustServ.Calls',data=df)
plt.title('Mean CustServ.Calls by Churn')
sns.barplot(x='CustServ.Calls', y='Churn',data=df,color='grey')
plt.title('Mean Churn by CustServ.Calls')
sns.distplot(df[df['Churn']==False]['Day.Mins'],label='Churn=no',bins=30)
sns.distplot(df[df['Churn']==True]['Day.Mins'],label='Churn=yes',bins=30)
plt.title('Day.Mins by Churn')
plt.legend()
sns.countplot(x='VMail.Plan',data=df,hue='Churn')
plt.title('Total VMail.Plan by Churn')
sns.barplot(x=df['VMail.Plan'],y=df['Churn'])
plt.title('Mean Churn by VMail.Plan')
sns.distplot(df[df['Churn']==False]['Intl.Mins'],label='Churn=no',bins=30)
sns.distplot(df[df['Churn']==True]['Intl.Mins'],label='Churn=yes',bins=30)
plt.title('Intl.Mins by Churn')
plt.legend()
sns.distplot(df[df['Churn']==False]['Intl.Calls'],kde=False,label='Churn=no',bins=10)
sns.distplot(df[df['Churn']==True]['Intl.Calls'],kde=False,label='Churn=yes',bins=10)
plt.title('Intl.Calls by Churn')
plt.legend()
sns.barplot(x='Churn',y='Intl.Calls',data=df,estimator=sum)
plt.title('Total Intl.Calls by Churn')
sns.distplot(df[df['Churn']==True]['CustServ.Calls'],kde=False,label='Churn=yes', bins=10, color='green')
plt.legend()
sns.distplot(df[df['Churn']==False]['CustServ.Calls'],kde=False,label='Churn=no', bins=10, color='blue')
plt.legend()
df.to_csv('../output/churn_working_clean.csv')
def process(churn):
    churn['Int.l.Plan'] = churn['Int.l.Plan']=='yes'
    churn['VMail.Plan'] = churn['VMail.Plan']=='yes'
    return churn.drop(['State','VMail.Message','Day.Charge','Intl.Charge',
                  'Account.Length','Area.Code','Day.Calls','Night.Calls','Phone'],axis=1)

churn_score = pd.read_csv('../input/churn_score.csv')
process(churn_score).to_csv('../output/churn_score_clean.csv')
