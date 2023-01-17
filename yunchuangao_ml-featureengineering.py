import pandas as pd
import numpy as np
import os
# data visualization
import seaborn as sns
%matplotlib inline
from matplotlib import pyplot as plt
from matplotlib import style
# Display the folders and files in current directory;
import os
for dirname, _, filenames in os.walk('/kaggle/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Load train data already pre-processed;
df_train = pd.read_csv('/kaggle/input/ml-course/train_processed.csv',index_col=0)
df_train.head()
# Combine SibSp and Parch as new feature; 
# Combne train test first;
all_data=[df_train]

for dataset in all_data:
    dataset['Family'] = dataset['SibSp'] + dataset['Parch'] + 1
df_train.head()
# Function of drawing graph;
def draw(graph):
    for p in graph.patches:
        height = p.get_height()
        graph.text(p.get_x()+p.get_width()/2., height + 5,height ,ha= "center")
# Family vs survied
plt.figure(figsize = (8, 5))
graph  = sns.countplot(x ="Family", hue ="Survived", data = df_train)
draw(graph)
df_train.describe()
# Use bin to convert ages to bins;
all_data=[df_train]

for dataset in all_data:
    dataset['Age'] = dataset['Age'].astype(int)
    dataset.loc[ dataset['Age'] <= 15, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 15) & (dataset['Age'] <= 20), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 20) & (dataset['Age'] <= 26), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 26) & (dataset['Age'] <= 28), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 28) & (dataset['Age'] <= 35), 'Age'] = 4
    dataset.loc[(dataset['Age'] > 35) & (dataset['Age'] <= 45), 'Age'] = 5
    dataset.loc[ dataset['Age'] > 45, 'Age'] = 6

df_train['Age'].value_counts()
plt.figure(figsize = (8, 5))
ag = sns.countplot(x='Age', hue='Survived', data=df_train)
draw(ag)
# Check fare vs survived;
# Create categorical of fare to plot fare vs Pclass first;
for dataset in all_data:
    dataset['Fare_cat'] = pd.cut(dataset['Fare'], bins=[0,10,50,100,550], labels=['Low_fare','median_fare','Average_fare','high_fare'])
plt.figure(figsize = (8, 5))
ag = sns.countplot(x='Pclass', hue='Fare_cat', data=df_train)
# Fare vs survived;
sns.barplot(x='Fare_cat', y='Survived', data=df_train)
corr=df_train.corr()#['Survived']

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
plt.subplots(figsize = (12,8))
sns.heatmap(corr, 
            annot=True,
            mask = mask,
            cmap = 'RdBu',
            linewidths=.9, 
            linecolor='white',
            vmax = 0.3,
            fmt='.2f',
            center = 0,
            square=True)
plt.title("Correlations Matrix", y = 1,fontsize = 20, pad = 20);