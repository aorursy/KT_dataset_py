

import numpy as np # Numpy library

import pandas as pd # Pandas library

import matplotlib.pyplot as plt # Matplotlib library for visualisation 

%matplotlib inline

import seaborn as sns # Matplotlib library for visualisation 



import warnings # Import warning liabraries to ignore standard warnings 

warnings.filterwarnings("ignore")



# Input data files are available in the "../input/" directory.



import os # os liabrary to find the directory where dataset is placed

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
mydf=pd.read_csv("../input/Loan payments data.csv") # Read the dataset CSV

mydf.head(5) # Let's find top 5 records of the dataset 
# Let's generate descriptive statistics of 

#dataframe (mydf) using describe function 

mydf.describe()
#Let's concise summary of our dataset using pandas info function

mydf.info()
#From below query we can see we have 100 null (NAN) values in paid_off_time and 300 null values in 

#past_due_days which is fine , reason - if someone pays earlier before due date these columns will not

#have values specified



mydf.isnull().sum()

# Check dataset shape [rows, columns],below query shows we have a dataset of 500 rows , 11 columns

mydf.shape
sns.set(style="whitegrid") # Lets set background of charts as white 
# First of all lets find out how many loan cases are Paid Off, Collection or Collection_PaidOff status

x = sns.countplot(x="loan_status", data=mydf )

y = sns.countplot(x="loan_status", data=mydf , hue='Gender')
x = sns.countplot(x="terms", data=mydf , hue='loan_status', palette='pastel', linewidth=5)
g = sns.catplot("loan_status", col="education", col_wrap=4,

                 data=mydf[mydf.loan_status.notnull()],

                 kind="count", height=12.5, aspect=.6)

ax = sns.barplot(x="Principal", y="age",hue="Gender" ,  data=mydf)

ax.legend(loc="upper right")
sns.set(style="whitegrid")

ax = sns.countplot(x="loan_status", hue="Gender", data=mydf ,palette='pastel' ,edgecolor=sns.color_palette("dark", 3))
fig = plt.figure(figsize=(25,5))

g = sns.catplot(x="Principal", hue="loan_status", col="Gender",palette='pastel',

                data=mydf, kind="count",

                 height=4, aspect=.7);
# Lets draw a pairplot to see data visualisation from different variables impact factor 

sns.pairplot(mydf, hue='Gender')
sns.set(style="whitegrid", palette="pastel", color_codes=True)



# Draw a nested violinplot and split the violins for easier comparison

sns.violinplot(x="Principal", y="terms", hue="Gender",

               split=True, inner="quart",

               

               data=mydf)

sns.despine(left=False)
g = sns.lmplot(x="age", y="Principal", hue="Gender",

               truncate=True, height=5, data=mydf)



# Use more informative axis labels than are provided by default

g.set_axis_labels("Age", "Principal")
# Plot miles per gallon against horsepower with other semantics

sns.relplot(x="Principal", y="age", hue="education",size="Gender",

            sizes=(40, 400), alpha=.5, palette="muted",

            height=6, data=mydf)
#(mydf.shape[0])

mydf['past_due_days'].isnull().sum()


defaultPerc=((mydf.shape[0]-mydf['past_due_days'].isnull().sum())/mydf.shape[0])*100

print(defaultPerc,"% of people paid after time")
# Pie chart, where the slices will be ordered and plotted counter-clockwise:

labels = 'After Due Date', 'Before Due Date'

sizes = [defaultPerc,100-defaultPerc]

explode = (0, 0.1)  # only "explode" the 2nd slice 



fig1, ax1 = plt.subplots()

ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



fig1.suptitle('People who paid Before Due Date or After Due Date', fontsize=16)





plt.show()
sns.boxplot(x='education', y='Principal', data=mydf)

plt.show()
sns.lmplot(x='Principal', y='age', hue = 'Gender', data=mydf, aspect=1.5, fit_reg = False)



plt.show()
sns.lmplot(x='Principal', y='age', hue = 'education', data=mydf, aspect=1.5, fit_reg = False)

plt.show()
fig = plt.figure(figsize=(15,5))

ax = sns.countplot(x="effective_date", hue="loan_status", data=mydf ,palette='pastel' ,edgecolor=sns.color_palette("dark", 3))

ax.set_title('Loan date')

ax.legend(loc='upper right')

for t in ax.patches:

    if (np.isnan(float(t.get_height()))):

        ax.annotate(0, (t.get_x(), 0))

    else:

        ax.annotate(str(format(int(t.get_height()), ',d')), (t.get_x(), t.get_height()*1.01))

plt.show();