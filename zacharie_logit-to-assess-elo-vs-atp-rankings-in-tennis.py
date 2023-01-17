# Importing the various modules I will need for this analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
# input data files are available in the input directory
# reading our data
data = pd.read_csv('../input/Tennis_ATP_ELO_Rank_Augmented.csv')
data.columns
# We want to include a dummy for the Series in our analysis
# b/c different Series attract players of different calibers
data['Series'].unique()
fig, ax =plt.subplots(1,1,figsize=(12,10))
plt.subplots_adjust(hspace = 0.2, top = 0.4)

g0 = sns.countplot(x="Series", data=data)
g0.set_xticklabels(g0.get_xticklabels(),rotation=45)
data['Court'].unique()
data['Court'].value_counts()
# Some surfaces are harder than other
data['Surface'].unique()
fig, ax =plt.subplots(1,1,figsize=(12,10))
plt.subplots_adjust(hspace = 0.2, top = 0.4)

g0 = sns.countplot(x="Surface", data=data)
g0.set_xticklabels(g0.get_xticklabels(),rotation=45)
data['Best of'].value_counts()
data_dv = pd.get_dummies(data,columns=['Series','Court','Surface','Best of'])
# I will drop INTERNATIONAL, OUTDOOR, HARD, "Best of" 3 from my dummies set
# I choose those dummy variables because they are my base categories
# I choose the base categories by choosing the categories with the most data
# data_dv.drop(['Series_International','Court_Outdoor',
#                             'Surface_Hard','Best of_3'], axis=1, inplace=True)
data_dv.columns
# creating VARS with only relevant quantitative variables both dependent and independent
vars = data_dv.iloc[:,[0,29,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47]]
vars.columns
x_elo=vars.iloc[:,[1,4,5,6,7,8,9,10,11,12,13,14,15,16]]
x_atp=vars.iloc[:,[3,4,5,6,7,8,9,10,11,12,13,14,15,16]]
y_won = vars.iloc[:,0]
x_spm = vars.iloc[:,[2]]
x_atp1 = x_atp.replace(('#VALUE!'),np.nan)
x_atp1 = pd.concat([y_won,x_atp1],axis=1)
x_atp1 = x_atp1.dropna()
x_atp1.head()
x_atp1.tail()
y_won1 = x_atp1.iloc[:,0] # Creating a vector of Won games with the unranked players removed
y_won1.astype(float).head()
x_atp1 = x_atp1.iloc[:,1:]
model = LogisticRegression()
model = model.fit(x_atp1,y_won1)
model.score(x_atp1, y_won1)
model_elo = model.fit(x_elo,y_won)
model_elo.score(x_elo,y_won)
print(model_elo.intercept_, model_elo.coef_)
y_vars = vars.iloc[:,[0,1,3]]
print(y_vars.corr(method='spearman',))