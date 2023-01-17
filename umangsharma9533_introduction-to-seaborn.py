# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()
countries_data=pd.read_csv('/kaggle/input/countries-of-the-world.csv')
countries_data.head()
# Use sns.scatterplot(x,y) to create a scatter plot

plt.figure(figsize=(10,10))

sns.scatterplot(x=countries_data['GDP ($ per capita)'],y=countries_data['Phones (per 1000)'])

plt.show()
# We can also create a count plot for the categorical data

plt.figure(figsize=(25,10))

# we can input data into y variable to rotate the plot

sns.countplot(x="Region",data=countries_data)

plt.xlabel("Region wise data")

plt.show()
# We can also create a count plot for the categorical data

plt.figure(figsize=(10,10))

# we can input column of  data into y variable to rotate the plot

sns.countplot(y="Region",data=countries_data)

plt.ylabel("Region wise data")

plt.show()
student_data=pd.read_csv('/kaggle/input/student-alcohol-consumption.csv')
sns.scatterplot(x="absences", y="G3", 

                data=student_data, 

                hue="sex")
# Changing hue order using hue_order variable

sns.scatterplot(x="absences", y="G3", 

                data=student_data, 

                hue="sex",hue_order=["M","F"])
plt.style.use('default')

# We can even change the color of hue using variable called palette as shown

palette_Colors={"M":"blue","F":"pink"}

sns.scatterplot(x="absences", y="G3", 

                data=student_data, 

                hue="sex",hue_order=["M","F"],

               palette=palette_Colors)
sns.relplot(x="absences", y="G3", 

            data=student_data,

            kind="scatter", 

            row="study_time",

            )

plt.show()
#Split graph column wise

sns.relplot(x="absences", y="G3", 

            data=student_data,

            kind="scatter", 

            col="study_time",

            )

plt.show()
# Below mentioned command will create 4 graphs ,with 4 differnt conditon

# 1) famsup = yes and schoolsup=yes means student gets support from family and school

# 2) famsup = yes and schoolsup=no means student gets support from family and no support from school

# 3) famsup = no and schoolsup=yes means student gets support from school and no support from family

# 1) famsup = no and schoolsup=no means student gets no support from family and school

sns.relplot(x="G1", y="G3", 

            data=student_data,

            kind="scatter", 

            col="schoolsup",

            col_order=["yes", "no"],

            row="famsup",

            row_order=["yes", "no"])



# Show plot

plt.show()
mpg=pd.read_csv('/kaggle/input/mpg.csv')
# Create scatter plot of horsepower vs. mpg

sns.relplot(x="horsepower", y="mpg", 

            data=mpg, kind="scatter", 

            size="cylinders",

            hue="cylinders")



# Show plot

plt.show()


sns.relplot(x="acceleration",y="mpg",data=mpg,hue="origin",style="origin",kind="scatter")



# Show plot

plt.show()
# Make the shaded area show the standard deviation

sns.relplot(x="model_year", y="mpg",

            data=mpg, kind="line",

            ci="sd")



# Show plot

plt.show()
# We can set confidence interval to None wth following code



sns.relplot(x="model_year", y="horsepower",

            data=mpg, kind="line",

            ci=None)



# Show plot

plt.show()

# Add markers and make each line have the same style

sns.relplot(x="model_year", y="horsepower", 

            data=mpg, kind="line", 

            ci=None, style="origin", 

            hue="origin",

            markers=True,

            dashes=False)



# Show plot

plt.show()
survey_data=pd.read_csv('/kaggle/input/young-people-survey-responses.csv')
#Use sns.catplot() to create a count plot using the survey_data DataFrame with "Internet usage" on the x-axis.

sns.catplot(x="Internet usage",data=survey_data,kind="count")





# Show plot

plt.show()
#Use sns.catplot() to create a count plot using the survey_data DataFrame with "Internet usage" on the y-axis.

sns.catplot(y="Internet usage", data=survey_data,

            kind="count")



# Show plot

plt.show()
# Create column subplots based on age category

plt.figure(figsize=(50,30))

sns.catplot(y="Internet usage", data=survey_data,

            kind="count",

            col="Age")



# Show plot

plt.show()
# Measuring intrest of youth based if their score is greater than or equal to 50%

survey_data["Interested in Math"]=survey_data["Mathematics"]>=(max(survey_data["Mathematics"])/2)
survey_data
# Create a bar plot of interest in math, separated by gender



sns.catplot(x="Gender",y="Interested in Math", data=survey_data,kind="bar")



# Show plot

plt.show()
# Plot the relation between student study time and their grades using barplots and arrange the bar in ascending order w.r.t to study time

# Turn off the confidence intervals

sns.catplot(x="study_time", y="G3",

            data=student_data,

            kind="bar",

            order=["<2 hours", 

                   "2 to 5 hours", 

                   "5 to 10 hours", 

                   ">10 hours"],ci=False)



# Show plot

plt.show()
#Use sns.catplot() and the student_data DataFrame to create a box plot with "study_time" on the x-axis and "G3" on the y-axis. Set the ordering of the categories to study_time_order

# Specify the category ordering

study_time_order = ["<2 hours", "2 to 5 hours", 

                    "5 to 10 hours", ">10 hours"]



# Create a box plot and set the order of the categories



sns.catplot(x="study_time",y="G3", data=student_data,kind="box",order=study_time_order)







# Show plot

plt.show()
# Create a box plot with subgroups on location column and omit the outliers

sns.catplot(x="internet",y="G3", data=student_data,kind="box",hue="location",sym="")





# Show plot

plt.show()
# Extend the whiskers to the 5th and 95th percentile

# This can be done using whis variable

# whis = 0.5 means 0.5*IQR



sns.catplot(x="romantic", y="G3",

            data=student_data,

            kind="box",

            whis=0.5)



# Show plot

plt.show()
# whis=[0,100] = means we are extending plot graph in range of min(0) to max(100) and in this case there will be no outliers in the plotting

# Set the whiskers at the min and max values

sns.catplot(x="romantic", y="G3",

            data=student_data,

            kind="box",

            whis=[0, 100])



# Show plot

plt.show()
#create a point plot with "famrel" on the x-axis and number of absences ("absences") on the y-axis

sns.catplot(x="famrel", y="absences",data=student_data,kind="box")

        

# Show plot

plt.show()
#create a point plot with "famrel" on the x-axis and number of absences ("absences") on the y-axis

sns.catplot(x="famrel", y="absences",data=student_data,kind="point")

        

# Show plot

plt.show()
# Add caps to the confidence interval

sns.catplot(x="famrel", y="absences",data=student_data,kind="point",capsize=0.2)

        

# Show plot

plt.show()
# Import median function from numpy as this data set contain many outliers  so median will be more effective as compared to mean



from numpy import median

# Plot the median number of absences instead of the mean

sns.catplot(x="romantic", y="absences",

data=student_data,

            kind="point",

            hue="school",

            estimator=median,dodge=True)



# Show plot

plt.show()
# Import median function from numpy as this data set contain many outliers  so median will be more effective as compared to mean



from numpy import median

# Plot the median number of absences instead of the mean

sns.catplot(x="romantic", y="absences",

data=student_data,

            kind="point",

            hue="school",

            ci=None,estimator=median)



# Show plot

plt.show()
# Plot the relation between student study time and their grades using barplots and arrange the bar in ascending order w.r.t to study time

# Turn off the confidence intervals

sns.catplot(x="study_time", y="G3",

            data=student_data,

            kind="bar",

            order=["<2 hours", 

                   "2 to 5 hours", 

                   "5 to 10 hours", 

                   ">10 hours"],ci=False)



# Show plot

plt.show()
# Set the color palette to "Purples"

sns.set_style("whitegrid")

sns.set_palette("Purples")

# Plot the relation between student study time and their grades using barplots and arrange the bar in ascending order w.r.t to study time

# Turn off the confidence intervals

sns.catplot(x="study_time", y="G3",

            data=student_data,

            kind="bar",

            order=["<2 hours", 

                   "2 to 5 hours", 

                   "5 to 10 hours", 

                   ">10 hours"],ci=False)



# Show plot

plt.show()
# Set the style to "darkgrid"

sns.set_style("darkgrid")

sns.set_palette(["#39A7D0","#36ADA4"])



# Set a custom color palette





# Create the box plot of age distribution by gender

sns.catplot(x="Gender", y="Age", 

            data=survey_data, kind="box")



# Show plot

plt.show()
 # Create scatter plot

sns.set_style("white")

g = sns.relplot(x="weight", 

                y="horsepower", 

                data=mpg,

                kind="scatter")



# Add a title "Car Weight vs. Horsepower"

# y variable set the distatnce of title from y-axis

g.fig.suptitle("Car Weight vs. Horsepower",y=1.03)

# Show plot

plt.show()
# Create line plot



g = sns.lineplot(x="model_year", y="mpg", 

                 data=mpg,

                 hue="origin",ci=None)



# Add a title "Average MPG Over Time"

g.set_title("Average MPG Over Time")



# Add x-axis and y-axis labels

g.set(xlabel="Car Model Year",ylabel="Average MPG")





# Show plot

plt.show()
# Create point plot

sns.catplot(x="origin", 

            y="acceleration", 

            data=mpg, 

            kind="point", 

            join=False, 

            capsize=0.1)



# Rotate x-tick labels

plt.xticks(rotation=90)



# Show plot

plt.show()