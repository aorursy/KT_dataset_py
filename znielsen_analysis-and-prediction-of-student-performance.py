# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns #For inital graphs. Super impressed with this

import matplotlib as mpl #Not sure if I need any matplotlib, possibly a prereq for seaborn.

import matplotlib.pyplot as plt

########################################################################################

#I mainly used sklearn libraries for all of my data analysis, which can be seen below. 

########################################################################################

from sklearn.cluster import KMeans 

from sklearn.decomposition import PCA #Principle component analysis. 

from sklearn.ensemble import RandomForestRegressor 

from sklearn.cross_validation import train_test_split #deprecated, should find another method

from sklearn.metrics import mean_squared_error #To judge how my model did. 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



df = pd.read_csv('../input/xAPI-Edu-Data.csv')



# Any results you write to the current directory are saved as output.
df.head(10)
sns.stripplot(x="Class", y="Discussion", data=df, jitter=True);
sns.stripplot(x="Class", y="raisedhands", data=df, jitter=True);
print(df['SectionID'].value_counts())

print(df['PlaceofBirth'].value_counts())
df1 = pd.get_dummies(df['gender'])

df2 = pd.get_dummies(df['StudentAbsenceDays'])

df3 = pd.get_dummies(df['Topic'])

df = pd.concat([df, df1, df2, df3], axis=1)

df.head()
kmeans_model = KMeans(n_clusters=3, random_state=1)

# Get only the numeric columns from games.

numericalColumns = df._get_numeric_data()

# Fit the model using the good columns.

kmeans_model.fit(numericalColumns)

# Get the cluster assignments.

labels = kmeans_model.labels_
# Create a PCA model.

pca_2 = PCA(2)

# Fit the PCA model on the numeric columns from earlier.

plot_columns = pca_2.fit_transform(numericalColumns)

# Make a scatter plot of each game, shaded according to cluster assignment.

plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=labels)

# Show the plot.

plt.show()
def map_values(row, values_dict):

    return values_dict[row]



values_dict = {'L': 1, 'M': 2, 'H': 3}



df['Grade'] = df['Class'].apply(map_values, args = (values_dict,))
train = df.sample(frac=0.8, random_state=1)

# Select anything not in the training set and put it in the testing set.

test = df.loc[~df.index.isin(train.index)]

# Print the shapes of both sets.

print(train.shape)

print(test.shape)
unwantedCols = ["gender", "NationalITy", 'Grade', "PlaceofBirth", "StageID", "GradeID", "SectionID", "Topic", "Semester", "Relation", "ParentAnsweringSurvey", "ParentschoolSatisfaction", "StudentAbsenceDays", "Class", "Grade"]

columns = df.columns.tolist()

columns = [c for c in columns if c not in unwantedCols]

target = "Grade"

# Initialize the model with some parameters.

model = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=1)

# Fit the model to the data.

model.fit(train[columns], train[target])

# Make predictions.

predictions = model.predict(test[columns])

# Compute the error.

mean_squared_error(predictions, test[target])