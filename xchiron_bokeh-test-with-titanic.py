# Importing Libraries



# pandas

import pandas as pd

from pandas import Series,DataFrame



#numpy, matplotlib, seaborn

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline



#bokeh

from bokeh.io import push_notebook, show, output_notebook

from bokeh.layouts import row

from bokeh.plotting import figure

from bokeh.charts import output_file, show, output_notebook

from bokeh.models import HoverTool, ColumnDataSource

output_notebook()



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# machine Learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB
# import train and test data frames

train_df = pd.read_csv("../input/train.csv", dtype={"Age": np.float64},)

test_df  = pd.read_csv("../input/test.csv", dtype={"Age": np.float64},)



# preview the data

train_df.head()
# view summary of the data

train_df.info()

print("-------------------------------------------")

test_df.info()
train_df.Cabin.unique()

len(train_df.Cabin.unique())



for col in train_df:

    print(col + ": " + str(len(train_df[col].unique())))
#Because I can't get CategoricalColorMapper to import correctly, I'm using the below code to add a column of color to the dataframe

train_df['color'] = np.where(train_df['Survived']==1, 'SpringGreen', 'Crimson')

#If CategoricalColorMapper I would use the following code:

#color_mapper=CategoricalColorMapper(factors=[0, 1],

#                                    palette=['Crimson', 'SpringGreen'])

#This element would be added to p.circle:  color=dict(field='Survived', transform=color_mapper)



#Define the source of data for our graphing

source = ColumnDataSource(train_df)



#Define the axis labels for the figure

p=figure(x_axis_label='Age',y_axis_label='Fare')



#x and y axis display for scatter plot, size 5 circles and color is the additional color column

p.circle('Age','Fare',source=source,size=5,alpha=0.8,color='color')



#adding the hover tool to show name, age, and fare

hover = HoverTool(tooltips=[('Name: ','@Name'),

                            ('Age: ','@Age'),

                            ('Fare: ','@Fare')])

p.add_tools(hover)



show(p, notebook_handle=True)
train_df['Sex_num'] = np.where(train_df['Sex']=='male', 1, 2)

source2 = ColumnDataSource(train_df)

p2=figure(x_axis_label='Age',y_axis_label='Sex')

#p2.circle('Age','Sex',source=source2,size=5,alpha=0.8,color='color')

p2.circle(train_df['Age'],train_df['Sex_num'],size=5,alpha=0.8,color='color')

hover2 = HoverTool(tooltips=[('Name: ','@Name'),

                            ('Age: ','@Age'),

                            ('Fare: ','@Fare')])

p2.add_tools(hover2)

show(p2, notebook_handle=True)
train_df.Sex.value_counts().plot(kind='bar')
plt.scatter(train_df['Age'],train_df['Pclass'],c=train_df['color'])