import pandas as pd

data = pd.read_csv('/kaggle/input/rotten-tomato-movie-reviwe/rotten tomato movie reviwe.csv')

data.head()
def strip_whitespace(x):

    return x.lstrip().rstrip()

for column in data.columns:

    try:

        data[column] = data[column].apply(lambda x : x.lstrip().rstrip())

    except:

        pass
data['Rating'] = data['Rating'].apply(lambda x : x.split('(')[0].rstrip())

data['Runtime'] = data['Runtime'].apply(lambda x : int(x.split(' ')[0]))

data['TOMATOMETER score'] = data['TOMATOMETER score'].apply(lambda x : int(x.split('%')[0]))

data['AUDIENCE score'] = data['AUDIENCE score'].apply(lambda x : int(x.split('%')[0]))

data['AUDIENCE count'] = data['AUDIENCE count'].apply(lambda x : int(''.join(x.split(','))))
data = data.rename(columns = {'Directed By':'Director','Runtime':'Time','TOMATOMETER score':'Critic Score','TOMATOMETER Count':'Critic Count',

                              'AUDIENCE score':'Audience Score','AUDIENCE count':'Audience Count'})
data['Difference'] = data['Critic Score'] - data['Audience Score']

data['Abs Difference'] = abs(data['Critic Score'] - data['Audience Score'])
data.head()
data['Critic > Audience'] = data['Difference'] == data['Abs Difference']
import seaborn as sns

import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))

sns.set_style('whitegrid')

sns.kdeplot(data['Critic Score'],data['Audience Score'],hue=data['Critic > Audience'],size=data['Audience Count'],palette=['#ff6666','#9fdf9f'])

sns.lineplot([0,100],[0,100],color='black')

plt.xlabel('Tomatometer')

plt.ylabel('Audience Score')

plt.show()
plot1 = pd.concat([data.sort_values('Abs Difference').head(),data.sort_values('Abs Difference').tail()])
plt.figure(figsize=(20,8))

colors = ['#ffc2b3' for i in range(5)]

colors.extend(['#8cd9b3' for i in range(5)])

plt.barh(plot1['Name'],plot1['Abs Difference'],color=colors)
plt.figure(figsize=(15,7))

sns.scatterplot(data['Abs Difference'],data['Audience Count'],size=data['Critic Count'],hue=data['Critic > Audience'],palette=['#ff8080','#8cd9b3'])

plt.xlabel('Absolute Difference between Critic and Audience Score')

plt.ylabel('Number of Audience Reviews')
data[(data['Audience Count'] < 50000) & (data['Audience Count'] > 40000)]['Name']
data.describe()
data[(data['Abs Difference']<5) & (data['Audience Score'] > 90)]
data.head(1)