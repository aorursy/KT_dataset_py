#Import libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
#Get the data and look at the first 5 rows

df = pd.read_csv('../input/students-performance-in-exams/StudentsPerformance.csv')

df.head()
#Get the shape of data

print(f'Shape of data: {df.shape}')

print(f'Number of rows: {df.shape[0]}')

print(f'Number of columns: {df.shape[1]}')

print(f'Number of dimensions: {df.ndim}')
#column names

print(f'Column names are: {df.columns}')
#replace ' ' and '/' with '_'

df.columns = df.columns.str.replace(' ', '_').str.replace('/', '_')

df.columns
#Look at dtypes

print(df.dtypes)

print('\n')

print(df.dtypes.value_counts())
#only show numerical dtypes

print(df.select_dtypes(include='number').head())



print('\n')



#only show non-numerical dtypes

print(df.select_dtypes(include='object').head())
#Look at the dtypes, missing values as well as memory usage at the same time.

df.info()
df.info()
#See memory usage for each column

df.memory_usage(deep=True)
#Let's look at the value counts for each.

for col in df.select_dtypes(include='object').columns:

  print(f'---Value counts for {col}---\n {df[col].value_counts()}. \n\n')
#You can simple do the same by calling this

df.select_dtypes(include='object').nunique()
#check current memory usage

df.select_dtypes(include='object').memory_usage(deep=True)
#look at how efficient it'd be when converted to category dtype

df.select_dtypes(include='object').astype('category').memory_usage(deep=True)
#covert object to categorical

non_numerical_columns = df.select_dtypes(include='object').columns

for col in non_numerical_columns:

  df[col] = df[col].astype('category')

print(df.select_dtypes(include='category').info())
df.info()
#look at numerical values again

df.select_dtypes(include='number').head()
print(f"- Min scores:\n{df.select_dtypes(include='number').min()}")

print('\n')

print(f"- Max scores:\n{df.select_dtypes(include='number').max()}")
#compare memory usage when using int64 vs int8

print(f"{df.select_dtypes(include='number').memory_usage()}\n") #int64

print(f"{df.select_dtypes(include='number').astype('int8').memory_usage()}\n") #int8
#Convert int64 to int8

numerical_columns = df.select_dtypes(include='number').columns

for col in numerical_columns:

  df[col] = df[col].astype('int8')

print(df.select_dtypes(include='number').info())
#Make sure if there is any missing values in the data.

df.isna().sum()
#show summary statistics with .describe()

df.describe(include='number').T
#.describe() on object/categorical dtypes show counts, nunique, etc.

df.describe(include='category').T
#set color palette

sns.set(palette='colorblind')
#check distribution & trend of test scores. Anything that stands out?

sns.pairplot(df)

plt.show()
#take a closer look

fig, ax = plt.subplots(figsize=(12,6))

ax = sns.regplot(df.writing_score, df.reading_score)

ax.set_title('reading score vs writing score')

plt.show()
#take a closer look at the score distribution

#show multiple plots

fig, axes = plt.subplots(1, 3, sharex=True, sharey=True)

axes[0].boxplot(df.math_score)

axes[0].set_title('math score')

axes[1].boxplot(df.reading_score)

axes[1].set_title('reading score')



axes[2].boxplot(df.writing_score)

axes[2].set_title('writing score')



plt.show()
#Make a new column with the avg of three scores

df['average_of_three_tests'] = df[['math_score', 'reading_score', 'writing_score']].mean(axis=1).round(0).astype('int8')
#check its distribution

sns.distplot(df.average_of_three_tests, bins=30)

plt.show()
df.corr()
#Heatmap to clearly show the correlation values between variables

sns.heatmap(df.corr(), cmap="YlGnBu")
#Review, there are five categorical variables

df.select_dtypes(include='category').describe().T
def autolabel(viz):

    '''For labling on bar chart'''

    for p in viz.patches:

        viz.annotate(format(p.get_height(), '.2f'), 

        (p.get_x() + p.get_width() / 2., p.get_height()), 

        ha = 'center', 

        va = 'center', 

        xytext = (0, 10), 

        textcoords = 'offset points')

    
def get_sorted_cat_values(num_column, cat_column, ascending=True):

  '''returns a list of sorted values in cat_column sorted by the mean of num_column.

     only accept by str name as in "gender." ''' 

  #----------------------------------------------------------------

  #arguments should be str

  if not (isinstance(num_column, str) and isinstance(cat_column, str)):

    raise ValueError('Enter column by its name in string!')

  #num_column should be numerical

  if not ( df[num_column].dtype.name.__contains__('int') or df[num_column].dtype.name.__contains__('float')):

    raise ValueError(f'First argument should be int or float. Your type was: {df[num_column].dtype.name}')

  #cat_column should be categorical

  if ((df[cat_column].dtype.name) != 'category'):

    raise ValueError(f'Seccond argument should be categorical. Your type was: {df[cat_column].dtype.name}')

  #----------------------------------------------------------------



  sorted_values = ( 

                    df.groupby(cat_column)[num_column]

                   .mean()

                   .sort_values(ascending=ascending)

                   .index

                   .unique()

                  )

  return sorted_values

#gender

fig = plt.figure(figsize=(12, 6))

viz = sns.barplot('gender', 'average_of_three_tests', data=df, order=get_sorted_cat_values('average_of_three_tests', 'gender', ascending=False), ci = None)

plt.title('average_of_three_tests with gender')

autolabel(viz)

plt.show()
#race

fig = plt.figure(figsize=(12, 6))

viz = sns.barplot('race_ethnicity', 'average_of_three_tests', data=df, order=get_sorted_cat_values('average_of_three_tests', 'race_ethnicity', ascending=False), ci = None)

plt.title('average_of_three_tests with race')

autolabel(viz)

plt.show()
#paret educaiton level

fig = plt.figure(figsize=(12, 6))

viz = sns.barplot(df.parental_level_of_education, df.average_of_three_tests, order=get_sorted_cat_values('average_of_three_tests', 'parental_level_of_education', False), ci = None)

plt.title('average_of_three_tests with parental_level_of_education')

autolabel(viz)

plt.show()
#lunch

fig = plt.figure(figsize=(12, 6))

viz = sns.barplot(df.lunch, df.average_of_three_tests, order=get_sorted_cat_values('average_of_three_tests', 'lunch', False), ci = None)

plt.title('average_of_three_tests with lunch')

autolabel(viz)

plt.show()
#test prep course

fig = plt.figure(figsize=(12, 6))

viz = sns.barplot(df.test_preparation_course, df.average_of_three_tests, order=get_sorted_cat_values('average_of_three_tests', 'test_preparation_course', False), ci = None)

plt.title('average_of_three_tests with test prep course')

autolabel(viz)

plt.show()