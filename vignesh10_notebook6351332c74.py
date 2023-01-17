#Please go through the comments in each cell.

import pandas as pd

import numpy as np

import scipy

import matplotlib.pyplot as plt

from sklearn.feature_selection import VarianceThreshold

import scipy.stats as stats
#Fetching the training data as a pandas dataframe for visualization

train_df = pd.read_csv("../input/train.csv")

#deleting the column 'id' from the dataframe as it is a unique and does not have any effect on the algorithm

del train_df['id']

"../input/train.csv"
#creating 2 seperate dataframes for categorical and continuous features.

train_df_cat = pd.DataFrame()#training data frame with categorical features

train_df_cont = pd.DataFrame() #training data with continuous features

cat_list = []#list of categorical features

cont_list = []#list of continuous features



#populating the created data frames for categorical and continuous features

for each_column in train_df.columns:

    if train_df[each_column].dtype == 'float':

        cont_list.append(each_column)

cont_list.remove('loss')

for each_column in train_df.columns:

    if train_df[each_column].dtype == 'object':

        cat_list.append(each_column)



for i in range(0,len(cat_list)):

    train_df_cat[i] = train_df[cat_list[i] ]

train_df_cat.columns =cat_list

for i in range(0,len(cont_list)):

    train_df_cont[i] = train_df[cont_list[i] ]

train_df_cont.columns =cont_list



#checking for missing values:

train_df.isnull().sum().sum()

#sum is 0 which indicates there are no misisng values in the training data.
#Univariate analysis:

#It is method of analysis where each variable can be analysed independently.

#For analysing the distributions of each feature, I have used  4 methods:

    #1.Histogram

    #2.Boxplot

    #3.stats.skew to measure the skewness in each continuous feature

#For the first three methods, I have used the plot() function with just changing the arguements for respective plots

#ex: df.plot(kind = 'hist') for histogram. I have customized the plot using other arguements to the plot function such

 #as (subplot = True) for plotting each column (each features) independently.



import matplotlib.pyplot as plt

%matplotlib inline



# change outlier point symbols

hist_dist = train_df_cont.plot(kind='hist', subplots=True,layout = (5,3),figsize = (15,15),sharex = False)



#On analysing the histogram of all the continuous features, the features cont1,cont5,cont7,cont8,cont9,cont13 seems to be 

#skewed. Let us confirm, if that is the case with the remaining methods.

#Method 2: Density plots

density_plot_cont = train_df_cont.plot(kind='density', subplots=True,layout = (5,3),figsize = (15,15),sharex = False)



#On analysing the density plots, it almost presents a similar picture as the histogram. The same set of features 

 #cont1,cont5,cont7,cont8,cont9,cont13 seems to be skewed, while the other features seems to be fairly symmetrical.

    
#Method 3: Box plots



color = dict(boxes='DarkGreen', whiskers='DarkOrange',medians='DarkBlue', caps='Gray')

box_plot_cont = train_df_cont.plot(kind='box', subplots=True,layout = (5,3),color=color, sym='r+',figsize = (15,15),

                                   sharex = False,showfliers=True)



#After all we are learning machine learning and avoiding redundancy is a key component hence would end the analysis

#with just stating similar results. But one key point to be highlighted in this case, is the presence of outliers 

#in the features cont 7, cont 9, cont 10.  eliminating outliers is a one of the important pre processing step before 

#applying the algorithm.
#Method 4: stats.skew

#Using the general thumb rule

#If the skewness is between -0.5 and 0.5, the data are fairly symmetrical

#If the skewness is between -1 and – 0.5 or between 0.5 and 1, the data are moderately skewed

#Hence fetching the list of features with skewness above and below the specified limit.



import scipy.stats as stats

skew_list = []

for each_column in train_df_cont.columns:

    skew_list.append(round(scipy.stats.skew(train_df_cont[each_column],bias = False),2))

skew_dict = dict(zip(train_df_cont.columns,skew_list))

print ("The skewness in each continuous features are:",skew_dict)

#Fetching the list of continuous features to be normalized

to_be_normalized = []

for keys,value in skew_dict.items():

    if  not((-0.5 < value < 0.5)):

        to_be_normalized.append(keys)



print ('\t')

print ("The continuous features to be normalized are",to_be_normalized)



print (train_df_cont.head())

    
#Feature pre processing of continuous features.

#Now that we have an idea of continuous features, let us get into preprocessing



#We are going to try Log transformation on the continuous features to reduce skewness.



for each_column in to_be_normalized:

    print ("The skew in", each_column ,"before applying the transformation:",train_df_cont[each_column].skew())

 

    train_df_cont[each_column] = np.log1p(train_df_cont[each_column])

    print ("The skew in", each_column ,"after applying the transformation:",train_df_cont[each_column].skew())

    print ('\t')

    

#You can see the reduction is skewness, but still the features 'cont5','cont7','cont9' have high skewness
#Multivariate Data Analysis:

#lets analyze and intrepret a key factor in the training data set, correlation between the features.

#Stating the general intitution, for better performance of a machine learning algorithm, the features has to be highly

#correalted with the class variable and ideally uncorrealted with the other features in the feature set.

#Lets perform the correaltion analysis to check if the above mentioned condition holds for our training data.





import matplotlib.pyplot as plt

import seaborn as sns

#creating a dataframe for storing the correaltion values between each continuous features. Could use any other data 

#structure as well, using pandas the code for analysis could be concise.

train_df_cont_corr = pd.DataFrame()

#populating the created dataframe

train_df_cont_corr = train_df_cont.corr()

plt.subplots(figsize=(21, 11))

sns.heatmap(train_df_cont_corr,annot=True)



plt.show()



#Now that we have a visual image of the correaltion between the continuous features, lets get the list of columns 

#which has correaltion above the specified limit of 0.6 in the heatmap.



for index,rows in train_df_cont_corr.iterrows():

    for each_column in train_df_cont_corr.columns:

        if rows[each_column] > 0.6 and rows[each_column] != 1:

            print ("The correaltion of continous features:", index, "and",each_column,"is", rows[each_column])



#Now we have the list of features which has inter feature correaltion above 0.6. I have added the condition '!= 1'

#beacuse it is clear from the heat map that a feature has a correlation of 1 only with itself, hence removing the 

#redundant information
#If a feature's value are all same, it cannot give us extra information. Hence finding the vatriance in data for 

#all the continuous features

vt = VarianceThreshold()

xt = vt.fit_transform(train_df_cont)

variance_dict = dict(zip(train_df_cont.columns,vt.variances_))

print (variance_dict)

print (min(variance_dict, key = variance_dict.get))



#Result: Found that the feature 'cont7' has the least variance.
#Let us analyse the feature importance of the continuous features.

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import f_regression

train_df_cont['loss'] = train_df['loss']

array = train_df_cont.values





X = array[:,0:14]

Y = array[:,14].astype(int)



test = SelectKBest(score_func=f_regression, k=4)

fit = test.fit(X, Y)

print ("The feature importance for the various continuous features are:")

d = dict(zip(train_df_cont.columns,fit.scores_))

sorted(d.items(), key=lambda x: (-x[1], x[0]))
#Now that, we have pre processed the continuous features lets analyze the categorical data.

#Intially we will see the number of categories in each categorical feature



category_list = []

for each_column in train_df_cat.columns:

    print ("The number of categories in the feature",each_column,"are:",len(train_df_cat[each_column].unique()))



#Now we know the number of categories in each categorical feature, for better performance of the algorithm 

#we will encode categories with numerical value