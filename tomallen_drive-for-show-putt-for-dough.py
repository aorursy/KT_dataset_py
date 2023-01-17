import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
pga_df = pd.read_csv('../input/pga-tour-20102018-data/PGA_Data_Historical.csv')

pga_df
pga_df.info()
pga_df = pga_df.set_index(['Player Name','Variable','Season'])['Value'].unstack('Variable').reset_index()

pga_df.head()
col = ["Player Name","Season","Top 10 Finishes - (TOP 10)","Top 10 Finishes - (EVENTS)","Driving Distance - (AVG.)",

          "Driving Accuracy Percentage - (%)","Scoring Average (Actual) - (AVG)", "3-Putt Avoidance > 25' - (2 PUTT OR BETTER %)",

          "Putting Average - (AVG)", "Total Money (Official and Unofficial) - (MONEY)"]
pga_df=pga_df[col]
pga_df.rename(columns = {'Player Name':'PlayerName'}, inplace = True)

pga_df.rename(columns = {'Top 10 Finishes - (TOP 10)':'Top_10_Finishes'}, inplace = True)

pga_df.rename(columns = {'Top 10 Finishes - (EVENTS)':'Events'}, inplace = True)

pga_df.rename(columns = {'Driving Distance - (AVG.)':'Avg_Drive_Dist'}, inplace = True)

pga_df.rename(columns = {'Driving Accuracy Percentage - (%)':'Avg_Drive_Acc'}, inplace=True)

pga_df.rename(columns = {'Scoring Average (Actual) - (AVG)':'Avg_Score'}, inplace=True)

pga_df.rename(columns = {"3-Putt Avoidance > 25' - (2 PUTT OR BETTER %)":'Three_Putt_Avoid'}, inplace=True)

pga_df.rename(columns = {'Putting Average - (AVG)':'Avg_Putt'}, inplace=True)

pga_df.rename(columns = {'Total Money (Official and Unofficial) - (MONEY)':'Money'}, inplace=True)

pga_df = pga_df.replace({'\$':'',',':''},regex = True)
for col in pga_df.columns[2:]:

    pga_df[col] = pga_df[col].astype(float)
pga_df['Avg_Money'] = round(pga_df['Money']/pga_df['Events'])

pga_df['Avg_Finish'] = pga_df['Top_10_Finishes']/pga_df['Events']
pga_df['Avg_Finish'] = pga_df['Avg_Finish'] > pga_df['Avg_Finish'].mean()

print('Per event, the average percentage a player finishes in the top 10 is: {0:.0%}'.format(pga_df['Avg_Finish'].mean()))
original_record_count = pga_df.shape[0]

for num in pga_df['Events']:

    pga_df['Events'] = round(pga_df['Events'].fillna(value=pga_df['Money']/pga_df['Avg_Money']))
pga_df.drop(['Top_10_Finishes'],axis = 1, inplace = True)
pga_df=pga_df.dropna()
new_record_count = pga_df.shape[0]

print(f'Original record count: {original_record_count}')

print(f'New record count: {new_record_count}')

print('Number of records dropped: '+str(original_record_count - new_record_count))
pga_df.head(5)
pga_scale = pga_df.copy(deep=True)

corr = pga_scale.corr()

corr
sns.set_style('whitegrid')

f, ax = plt.subplots(1,2, figsize = (12,5))

ax[0].scatter('Avg_Drive_Dist','Money',data=pga_df,color='green')

ax[0].set_title("Driving for Money")

ax[0].set_xlabel('Driving Avg. (Yards)')

ax[0].set_ylabel('Money ($)')

ax[1].scatter('Avg_Putt','Money',data=pga_df, color = 'green')

ax[1].set_title("Putting for Money")

ax[1].set_xlabel('Putting Avg. (Putts per Hole)')

ax[1].set_ylabel('Money ($)')
corr_drive = round(corr['Avg_Money']['Avg_Drive_Dist'],4)

corr_putt = round(corr['Avg_Money']['Avg_Putt'],4)

print(f'The correlation between Average Driving Distance and Money earned is: {corr_drive}')

print(f'The correlation between Average Putts and Money earned is: {corr_putt}')
all_avg_acc = pga_df['Avg_Drive_Acc']

all_avg_dist = pga_df['Avg_Drive_Dist']

sns.set_style('dark')

sns.jointplot(x=all_avg_acc,y=all_avg_dist,data=pga_df)
pga_df[(pga_df['Avg_Drive_Dist']>310) & (pga_df['Avg_Drive_Acc']>62)]
print('The correlation between driving distance and driving accuracy is: ' + str(round(corr['Avg_Drive_Dist']['Avg_Drive_Acc'],4)))
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

Feature_columns = ['Avg_Drive_Dist','Avg_Drive_Acc', 'Avg_Score','Three_Putt_Avoid','Avg_Putt',

                   'Money','Avg_Money','Avg_Finish']

pga_scale[Feature_columns]=scaler.fit_transform(pga_scale[Feature_columns])
sns.heatmap(pga_scale.corr(),linewidth=1,linecolor='white',cmap='Greens')
from sklearn.model_selection import train_test_split

X = pga_df[Feature_columns].drop('Avg_Finish', axis=1)

y = pga_df['Avg_Finish']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.50)
from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()

dtree.fit(X_train,y_train)
predictions = dtree.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix

print(confusion_matrix(y_test,predictions))

print(classification_report(y_test,predictions))
X = pga_df[['Avg_Drive_Dist','Avg_Drive_Acc','Avg_Putt','Three_Putt_Avoid',]]

y = pga_df['Avg_Score']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
from sklearn.linear_model import LinearRegression

lm = LinearRegression() # creates an object from the LinearRegression class

lm.fit(X_train,y_train) # fits the training set to the model
coeff_df = round(pd.DataFrame(lm.coef_, X.columns, columns=['Coefficient']),3)

coeff_df
predictions = lm.predict(X_test)

plt.scatter(y_test, predictions, color = 'green')
from sklearn import metrics

MSE = round(metrics.mean_squared_error(y_test, predictions),4)

print(f'Mean Squared Error = {MSE}')
s = sns.PairGrid(pga_df, y_vars = ['Season'],x_vars = ['Avg_Drive_Dist','Money','Avg_Score'], height = 4)

s.map(sns.regplot)