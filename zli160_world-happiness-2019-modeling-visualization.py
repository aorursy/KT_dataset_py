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
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current 

from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, export_graphviz
import sklearn.metrics as met
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

happy = pd.read_csv('/kaggle/input/world-happiness/2019.csv')

##############################################################################
###        Data cleaning
#  First, we check how much of our data is missing, and found there are no missing values.

#  Second, renamed the column names to make it easier to work with the data. Typing out 'GDP Per Capita' is 
#cumbersome at best so 'GDC' is far preferable.  made all the names one word long.
happy.rename(columns={ 'GDP per capita' : 'GDP',  'Social support' : 'Social', 
 'Healthy life expectancy' : 'Health',  
 'Freedom to make life choices' : 'Freedom',  'Generosity': 'Generosity', 
 'Perceptions of corruption' : 'Corruption'}, inplace=True)

#  Third, we find that the variables of the  data are all between 0 to 2. And the response value is between 0-10.  
#  Standardize the variables to see if there are any outliers within the data.
happy['zGDP'] = stats.zscore(happy['GDP'])
happy['zSocial'] = stats.zscore(happy['Social'])
happy['zHealth'] = stats.zscore(happy['Health'])
happy['zFreedom'] = stats.zscore(happy['Freedom'])
happy['zGenerosity'] = stats.zscore(happy['Generosity'])
happy['zCorruption'] = stats.zscore(happy['Corruption'])
happy.query('zGDP > 3 | zGDP < -3')
# No outliers for zGDP
happy.query('zSocial > 3 | zSocial < -3')
# Central Africa Republic has zSocial score of 4.05
oSocial = happy.query('zSocial > 3 | zSocial < -3')
# I'm saving this outlier
happy.query('zHealth > 3 |zHealth  < -3')
# Swaziland has a zHealth score of -3.005
oHealth = happy.query('zHealth > 3 |zHealth  < -3')
# I'm saving this outlier
happy.query('zFreedom > 3 | zFreedom  < -3')
# No outlier for zFreedom
happy.query('zGenerosity > 3 |zGenerosity  < -3')
# Myanmar has a zGenerosity score of 4.01 
# Indonesia has a zGenerosity score of 3.298 
oGenerosity = happy.query('zGenerosity > 3 |zGenerosity  < -3')
happy.query('zCorruption > 3 | zCorruption < -3')
# Singapore has a zCorruption score of 3.63
# Rwanda has a zCorruption score of 3.19
# Denmark has a zCorruption score of 3.18
oCorruption = happy.query('zCorruption > 3 | zCorruption < -3')
# No outlier for zFreedom
# Outliers are standardized numbers above 3 or below -3. 7 total countries across four categories fall into the outlier designation.
happy_outliers = oSocial, oHealth, oGenerosity, oCorruption
# This stores all of the outliers in one place if we want to see them
# Because our numbers deal with people's perceptions, we won't remove the outliers.


##############################################################################
#           Exploratory Data Analytics:

# Matplot / Visualization   
plt.plot(range(len(happy['Score'])),happy['Score'])
plt.ylabel("Happiness Score")
plt.title("Happiness Score Matplot")
plt.show()
plt.plot(range(len(happy['GDP'])),happy['GDP'])
plt.ylabel("GDP per Capita")
plt.title("GDP Matplot")
plt.show()

plt.plot(range(len(happy['Social'])),happy['Social'])
plt.ylabel("Social Support")
plt.title("Social Support Matplot")
plt.show()

plt.plot(range(len(happy['Health'])),happy['Health'])
plt.ylabel("Life Expectancy")
plt.title("Healthy Life Expectancy")
plt.show()

plt.plot(range(len(happy['Freedom'])),happy['Social'])
plt.ylabel("Freedom to make life choices")
plt.title("Freedom to make life choices")
plt.show()

plt.plot(range(len(happy['Generosity'])),happy['Social'])
plt.ylabel("Generosity")
plt.title("Generosity")
plt.show()

plt.plot(range(len(happy['Corruption'])),happy['Social'])
plt.ylabel("Perceptions of corruption")
plt.title("Perceptions of corruption")
plt.show()
# the matplots have the same trends with histograms and not 
# easy to read. Not show in presentation. 
# Histograms Visualization                         
#All variables
plt.hist(happy['Score'],color='green',alpha=0.35)
plt.xlabel('Happiness Score')
plt.ylabel('Frequency')
plt.title('Happiness Summary')
plt.show
##------------------------------------------------------------------

plt.hist(happy['GDP'],edgecolor='k',alpha=0.35)
plt.xlabel('GDP Value per capita')
plt.ylabel('Frequency')
plt.title('GDP per Capita')
plt.show
##------------------------------------------------------------------

plt.hist(happy['Social'],color='r',alpha=0.35)
plt.xlabel('Social Support')
plt.ylabel('Frequency')
plt.title('Social Support')
plt.show
##------------------------------------------------------------------

plt.hist(happy['Health'],color='orange',alpha=0.35)
plt.xlabel('Life Expectancy')
plt.ylabel('Frequency')
plt.title('Healthy Life Expectancy')
plt.show
##------------------------------------------------------------------

plt.hist(happy['Freedom'],edgecolor='k',alpha=0.35)
plt.xlabel('Freedom to make life choices')
plt.ylabel('Frequency')
plt.title('Freedom to make life choices')
plt.show
##------------------------------------------------------------------

plt.hist(happy['Generosity'],color='darkblue',alpha=0.35)
plt.xlabel('Generosity')
plt.ylabel('Frequency')
plt.title('Generosity')
plt.show
##------------------------------------------------------------------

plt.hist(happy['Corruption'],color='purple',alpha=0.35)
plt.xlabel('Perceptions of corruption')
plt.ylabel('Frequency')
plt.title('Perceptions of corruption')
plt.show
#Most of the respondents in the world happiness report feel their life
# at an average level of happiness which is in the range between 4.5 and 6.5.
# Corresponding the GDP level is similar. Most of the people do not take care about
# the perception of corruption while freedom to make life choices is extremely important for them.
#------------------------------------------------------


# Histograms Visualization with overlay and four groups    
#there are totally 156 countries or regions in the rank list. I divided them into
#4 groups according to the ranking.
# Rename the Column name of Overall rank and Country or region
happy_df=happy
happy_df.rename(columns={ 'Overall rank' : 'Rank',  'Country or region' : 'Area'}, inplace=True)

happy_1=happy_df[happy_df.Rank<40]
happy_2=happy_df[(happy_df.Rank>39)&(happy_df.Rank<79)]
happy_3=happy_df[(happy_df.Rank>78)&(happy_df.Rank<118)]
happy_4=happy_df[happy_df.Rank>117]
happy_1.shape
happy_2.shape
happy_3.shape
happy_4.shape

happyset1=happy_1['Score']
happyset2=happy_2['Score']
happyset3=happy_3['Score']
happyset4=happy_4['Score']
(n_ha,bins,patches)=plt.hist([happyset1,happyset2,happyset3,happyset4],bins=4,stacked=True)
plt.legend(['Rank1-39','Rank40-78','Rank79-117','Rank118-156'])
plt.xlabel('Happiness Score');plt.ylabel('Frequency')
plt.title('Histogram of Happiness Score')
plt.show()

happyset1=happy_1['GDP']
happyset2=happy_2['GDP']
happyset3=happy_3['GDP']
happyset4=happy_4['GDP']
(n_ha,bins,patches)=plt.hist([happyset1,happyset2,happyset3,happyset4],bins=4,stacked=True)
plt.legend(['Rank1-39','Rank40-78','Rank79-117','Rank118-156'])
plt.xlabel('GDP');plt.ylabel('Frequency')
plt.title('Histogram of GDP per capita')
plt.show()

happyset1=happy_1['Social']
happyset2=happy_2['Social']
happyset3=happy_3['Social']
happyset4=happy_4['Social']
(n_ha,bins,patches)=plt.hist([happyset1,happyset2,happyset3,happyset4],bins=4,stacked=True)
plt.legend(['Rank1-39','Rank40-78','Rank79-117','Rank118-156'])
plt.xlabel('Social Support');plt.ylabel('Frequency')
plt.title('Histogram of Social Support')
plt.show()

happyset1=happy_1['Health']
happyset2=happy_2['Health']
happyset3=happy_3['Health']
happyset4=happy_4['Health']
(n_ha,bins,patches)=plt.hist([happyset1,happyset2,happyset3,happyset4],bins=4,stacked=True)
plt.legend(['Rank1-39','Rank40-78','Rank79-117','Rank118-156'])
plt.xlabel('Healthy Life Expectancy');plt.ylabel('Frequency')
plt.title('Histogram of Healthy Life Expectancy')
plt.show()


happyset1=happy_1['Freedom']
happyset2=happy_2['Freedom']
happyset3=happy_3['Freedom']
happyset4=happy_4['Freedom']
(n_ha,bins,patches)=plt.hist([happyset1,happyset2,happyset3,happyset4],bins=4,stacked=True)
plt.legend(['Rank1-39','Rank40-78','Rank79-117','Rank118-156'])
plt.xlabel('Freedom to make life choices');plt.ylabel('Frequency')
plt.title('Histogram of Freedom to make life choices')
plt.show()

happyset1=happy_1['Generosity']
happyset2=happy_2['Generosity']
happyset3=happy_3['Generosity']
happyset4=happy_4['Generosity']
(n_ha,bins,patches)=plt.hist([happyset1,happyset2,happyset3,happyset4],bins=4,stacked=True)
plt.legend(['Rank1-39','Rank40-78','Rank79-117','Rank118-156'])
plt.xlabel('Generosity');plt.ylabel('Frequency')
plt.title('Histogram of Generosity')
plt.show()

happyset1=happy_1['Corruption']
happyset2=happy_2['Corruption']
happyset3=happy_3['Corruption']
happyset4=happy_4['Corruption']
(n_ha,bins,patches)=plt.hist([happyset1,happyset2,happyset3,happyset4],bins=4,stacked=True)
plt.legend(['Rank1-39','Rank40-78','Rank79-117','Rank118-156'])
plt.xlabel('Perceptions of corruption');plt.ylabel('Frequency')
plt.title('Histogram of Perceptions of corruption')
plt.show()
# The histogram can not show any rules among the groups. 

###########################################################
#Correlation between variable ：
#Option1:
happy.corr()
#It is simple to create a correlation table. However, it is hard to distinguish
#from table the weighfactors at the first glance. Therefore, a heat map was created below.
#Option2: heat map for correlation between varialbes.
corr_matrix1,ax=plt.subplots(figsize=(10, 10))
sns.heatmap(happy.corr(), ax=ax, annot=True, linewidth=0.05, fmt='.2f', cmap='magma')
plt.title("Correlations Overall Areas")
plt.show()
#In this Heat Map we can see that Happiness Score is very highly correlated with GDP, 
#Social support, and Life expectancy and somewhat related with Freedom also, 
#but has a very low relation with Generosity and Perceptions of corruption in average case.

# Created a new dataframe of top10 happiness countries and add a new index to category
top10=happy.iloc[0:10, 0:]
top10['category']='Top10'
top10.ax=plt.subplots(figsize=(10,10))
top10_d=top10.loc[lambda top10: top10['category']=='Top10']
top10_matrix=np.triu(top10_d.corr())
sns.heatmap(top10_d.corr(),cmap='ocean', annot=True)
plt.title("Correlations of Top 10 Areas")
plt.show()
#The Heat Map particularly for Top 10 areas has one more thing to add apart from Family Satisfaction, Freedom, Economy, Generosity, It is also highly correlated with Trust in Government.

 # Created a new dataframe of last10 happiest countries and add a new index to category
bottom10=happy.iloc[146:156, 0:]
bottom10['category']='Bottom10'
plt.rcParams['figure.figsize']=(10, 10)
bottom10_d=bottom10.loc[lambda bottom10: bottom10['category']=='Bottom10']
sns.heatmap(bottom10_d.corr(), cmap='Wistia', annot=True)
plt.title("Correlations of Last 10 Areas")
plt.show()

#Correlation between variable ：
#In this Heat Map we can see that Happiness Score is very highly correlated with GDP, 
#Social support, and Life expectancy and somewhat related with Freedom also, 
#but has a very low relation with Generosity and Perceptions of corruption in average case.
plt.rcParams['figure.figsize'] = (20, 15)
sns.heatmap(happy.corr(), cmap = 'copper', annot = True)
plt.show()
#the trend of the top 10 and last 10 countries and regions are distinct-different.
# People in the top 10 areas do not think happiness is related to economic level,
# while they are more focusing on social support and freedom to make life decisions.
# By contrast, people in the bottom 10 areas are still caring about the economy 
#and health which are the most important to decide their life happiness.



#####################################
#Construct Polar Chart for top 5 and bottom 5 for observation of trends and relationships

#top5 areas dataset to array
happy001=happy.iloc[0,3:9]
happy1=np.array(happy001)
happy1=happy1.tolist()
happy1=np.concatenate((happy1,[happy1[0]]))

happy002=happy.iloc[1,3:9]
happy2=np.array(happy002)
happy2=happy2.tolist()
happy2=np.concatenate((happy2,[happy2[0]]))

happy003=happy.iloc[2,3:9]
happy3=np.array(happy003)
happy3=happy3.tolist()
happy3=np.concatenate((happy3,[happy3[0]]))

happy004=happy.iloc[3,3:9]
happy4=np.array(happy004)
happy4=happy4.tolist()
happy4=np.concatenate((happy4,[happy4[0]]))

happy005=happy.iloc[4,3:9]
happy5=np.array(happy005)
happy5=happy5.tolist()
happy5=np.concatenate((happy5,[happy5[0]]))
print(happy1);print(happy2);print(happy3);print(happy4);print(happy5)
###################################
#Bottom 5 areas dataset to array
happy0152=happy.iloc[151,3:9]
happy152=np.array(happy0152)
happy152=happy152.tolist()
happy152=np.concatenate((happy152,[happy152[0]]))

happy0153=happy.iloc[152,3:9]
happy153=np.array(happy0153)
happy153=happy153.tolist()
happy153=np.concatenate((happy153,[happy153[0]]))

happy0154=happy.iloc[153,3:9]
happy154=np.array(happy0154)
happy154=happy154.tolist()
happy154=np.concatenate((happy154,[happy154[0]]))

happy0155=happy.iloc[154,3:9]
happy155=np.array(happy0155)
happy155=happy155.tolist()
happy155=np.concatenate((happy155,[happy155[0]]))

happy0156=happy.iloc[155,3:9]
happy156=np.array(happy0156)
happy156=happy156.tolist()
happy156=np.concatenate((happy156,[happy156[0]]))
print(happy152);print(happy153);print(happy154);print(happy155);print(happy156)
#############################

#set polar chart shapes and labels(same for two charts)
labels=np.array(['GDP per capita','Social Support','Healthy Life Expectancy',
                 'Freedom to make life choices', 'Generosity','Perceptions of corruption'])
happyLength=6
angles=np.linspace(0,2*np.pi, happyLength, endpoint=False)
angles=np.concatenate((angles,[angles[0]]))

#Polar chart1 for top5 areas
fig1=plt.figure()
ax1=fig1.add_subplot(111,polar=True)
ax1.plot(angles,happy1,'ro-',label='Finland',color='r',linewidth=3)
ax1.plot(angles,happy2,'ro-',label='Denmark',color='y',linewidth=3)
ax1.plot(angles,happy3,'ro-',label='Norway',color='orange',linewidth=3)
ax1.plot(angles,happy4,'ro-',label='Iceland',color='pink',linewidth=3)
ax1.plot(angles,happy5,'ro-',label='Netherlands',color='black',linewidth=3)

ax1.set_thetagrids(angles*180/np.pi,labels,fontproperties="fontprop")
ax1.set_title("Top5 Country or Region",va='bottom',fontproperties="Times New Roman")
ax1.grid(True)
plt.legend()
plt.show()

#set polar chart shapes and labels(same for two charts)
labels=np.array(['GDP per capita','Social Support','Healthy Life Expectancy',
                 'Freedom to make life choices', 'Generosity','Perceptions of corruption'])
happyLength=6

angles=np.linspace(0,2*np.pi, happyLength, endpoint=False)

angles=np.concatenate((angles,[angles[0]]))

#Polar chart2 for bottom5 areas
fig2=plt.figure()
ax2=fig2.add_subplot(111,polar=True)
ax2.plot(angles,happy152,'ro-',label='Rwanda',color='r',linewidth=3)
ax2.plot(angles,happy153,'ro-',label='Tanzania',color='y',linewidth=3)
ax2.plot(angles,happy154,'ro-',label='Afghanistan',color='orange',linewidth=3)
ax2.plot(angles,happy155,'ro-',label='Central African Republic',color='pink',linewidth=3)
ax2.plot(angles,happy156,'ro-',label='South Sudan',color='blue',linewidth=3)

ax2.set_thetagrids(angles*180/np.pi,labels,fontproperties="fontprop")
ax2.set_title("Bottom5 Country or Region",va='bottom',fontproperties="Times New Roman")
ax2.grid(True)
plt.legend()
plt.show()
# the top 5 countries have the similar influential factors while the bottom countries
# have quite different factors

##############################################################################
##############################################################################
#           Partition/ Validation/ Methodology
# we split our data set to train 78% & test 22% which is(121：35),in order to improve the performance of our small size data trainning.
happy_train2, happy_test2 = train_test_split(happy, test_size = 0.22, random_state = 7)

# The Y value of our dataset is continuous variables, so here are the models we can do if we use supervised learning:
    #Regression, Linear Regression, Decision Tree, RandomForest, GBT, AFT Survival Regression, Isotonic Regression.
# We chose three of them and made a comparison which is Random Forest Regression, Multiple Regression and DecisionTrees.

# We did a lot of tests of MAE in each model, with different partitions, 
    #for example 80% with 20%, 78% with 22%, 75%with 25%, 70% with 30%, 67% with 32%, and 65% with 35%.
#After the Horizontal and vertical contrast we found 78%with 22% is the best split. Since it has the least error of each model.

#############################################################################
#          Using three model to test/validate our Partition and Methodology

#           Random Forest Regressor
#Again, Random ForestClassifier and 'gini' are used for int data. Because Score out data is floating point (there are always decimals), we're using Random Forest Regressor
happy_train2, happy_test2 = train_test_split(happy, test_size = 0.22, random_state = 7)
#### For overall 
cols = ['GDP', 'Health', 'Freedom','Social', 'Corruption', 'Generosity']
Xtrain_2 = happy_train2[cols]
ytrain_2 = happy_train2['Score']
# Test the max_depth with 1,2,3,4,5. and 4 performance the best result.( which is less error)
rf01 = RandomForestRegressor(max_depth=4, random_state = 0)
rf01.fit(Xtrain_2, ytrain_2)

#Create Test data set:        
Xtest_2 = happy_test2[cols]
ytest_2= happy_test2['Score']

# Use the test data to make predictions
rf26_pred = rf01.predict(Xtest_2)
MSE = np.sum(np.power(np.array(ytest_2) - np.array(rf26_pred), 2)) / len(ytest_2)
#MSE:0.3564
MAE_test = met.mean_absolute_error(ytest_2, rf26_pred)
# MAE is obtained by subtracting the predict data from the actual data : 0.4360

#Random Forest Regressor performance Visualization
plt.figure()
plt.scatter(rf26_pred, ytest_2)
plt.plot([3, 7], [3, 7], 'k')
plt.title('Random Forest Regressor with overall')
plt.xlabel('Predict Happiness Score') 
plt.ylabel('Actual Happiness Score')


############################################
#          Multiple Regression model
# Since we want to compare the Multiple Regression model with others models, so we did this overall model
# Partitioned data use same partition with Random Forest Regressor
#Training data set
happy_train2, happy_test2 = train_test_split(happy, test_size = 0.22, random_state = 7)
x1 = pd.DataFrame(happy_train2[['GDP', 'Health', 'Freedom', 'Social', 'Corruption', 'Generosity']])
x1 = sm.add_constant(x1)
y1 = pd.DataFrame(happy_train2[['Score']])
model1 = sm.OLS(y1, x1).fit()
model1.summary()

# Test data set
x2 = pd.DataFrame(happy_test2[['GDP', 'Health', 'Freedom', 'Social', 'Corruption', 'Generosity']])
x2 = sm.add_constant(x2)
y2 = pd.DataFrame(happy_test2[['Score']])
model2 = sm.OLS(y2, x2).fit()
model2.summary()


y2pred = model1.predict(x2)
MAEregression_test = met.mean_absolute_error(y2, y2pred)
#0.5129530374043557

#Multiple Regression model performance Visualization
plt.figure()
plt.scatter(y2pred, y2)
plt.plot([3, 7], [3, 7], 'k')
plt.title('Multiple Regression model & Overall')
plt.xlabel('Predict Happiness Score') 
plt.ylabel('Actual Happiness Score')

###########################################
#                    DECISION TREEs
# Since we want to compare the Decision Tree model performance with others models, so we did this overall model
happy_train2, happy_test2 = train_test_split(happy, test_size = 0.22, random_state = 7)
#Train data set
x1 = pd.DataFrame(happy_train2[['GDP', 'Health', 'Freedom', 'Social', 'Corruption', 'Generosity']])
x1 = sm.add_constant(x1)
y1 = pd.DataFrame(happy_train2[['Score']])
cart01 = DecisionTreeRegressor(criterion = 'mse', max_leaf_nodes = 8).fit(x1,y1)
#Test data set
x2 = pd.DataFrame(happy_test2[['GDP', 'Health', 'Freedom', 'Social', 'Corruption', 'Generosity']])
x2 = sm.add_constant(x2)
y2 = pd.DataFrame(happy_test2[['Score']])
# Use the test data to make predictions
dt_pred = cart01.predict(x2)
#Ther MAE is 0.5613471638655461
MAE_test = met.mean_absolute_error(y2, dt_pred)

#Decision Tree model performance Visualization
plt.figure()
plt.scatter(dt_pred, y2)
plt.plot([3, 7], [3, 7], 'k')
plt.title('Decision Tree & Overall')
plt.xlabel('Predict Happiness Score') 
plt.ylabel('Actual Happiness Score')

##############################################################################
#MAEbaseline = 0.8949
MAEbaseline = np.mean(y1)
y2.shape
MAEbaseline_list = ([5.444917]*35)
MAE_test = met.mean_absolute_error(y2, MAEbaseline_list)

############################################################################
     #           Modeling and Predictive Analytics
############################################################################
#DECISION TREE
# DecisionTreeClassifier and 'gini' are used for int data. Because Score out data is floating point (there are always decimals), we're using DecisionTreeRegressor and 'mse'
# For Overall X
y = happy[['Score']]
x = happy[['GDP', 'Health','Social', 'Freedom', 'Corruption', 'Generosity']]
x_names = ['GDP', 'Health','Social', 'Freedom', 'Corruption', 'Generosity']
cart01 = DecisionTreeRegressor(criterion = 'mse', max_leaf_nodes = 8).fit(x,y)
#cannot show from here, but can show on your computer
#export_graphviz(cart01, out_file = "/Users/zli/Downloads/894_813759_bundle_archive/happy_cartall.dot", feature_names=x_names)
# Here’s a decision tree regression model with the five strongest predictors from our six variable dataset.  The first split occurs along the Social rating. Countries that have a less than or equal to a 1.206 go to the left, those that are higher go right.

# Following the True branch down, we see further splits using GDP (less than or equal to 0.4491), and Freedom (less than or equal to .024).

# Along the False (or higher Social value) branch, we see splits along GDP ( less than or equal to 1.266), Health (less than or equal to 0.857) and Social (less than or equal to 1.442) before the final split at Freedom (less than or equal to 0.405).

# There are eight boxes that represent an ending for one of the branching paths. Here they are in order of highest Happiness rating to lowest.

# Happy Rating: 7.079 (Social greater than 1.442, GDP greater than 1.266)
# Happy Rating: 6.161 (Social less than or equal to 1.442, GDP greater than 1.266)
# Happy Rating: 6.109 (Social greater than 1.206, GDP less than or equal to 1.266, Health greater than 0.857)
# Happy Rating: 5.729 (Social greater than 1.206, GDP less than or equal to 1.266, Health less than or equal to 0.857), Freedom greater than 0.0405)
# Happy Rating: 5.287 (Social greater than 1.206, GDP less than or equal to 1.266, Health less than or equal to 0.857), Freedom less than or equal to 0.0405)
# Happiness Rating 4.716 (Social less than or equal to 1.206, GDP greater than 0.491)
# Happy Rating: 4.206 (Social less than or equal to 1.206, GPD less than or equal to 0.491, Freedom greater than 0.24)
# Happy Rating: 3.572 (Social less than or equal to 1.206, GPD less than or equal to 0.491, Freedom less than or equal to 0.24)

# That’s a lot of numbers! The short explanation is that higher numbers in any of these categories equals higher happiness. 

y = happy[['Score']]
x = happy[['GDP', 'Health']]
x_names = ['GDP', 'Health']
cart01 = DecisionTreeRegressor(criterion = 'mse', max_leaf_nodes = 5).fit(x,y)
#cannot show from here, but can show on your computer
#export_graphviz(cart01, out_file = "/Users/zli/Downloads/894_813759_bundle_archive/2019happy_cart01.dot", feature_names=x_names)
# The top box shows that Health is going to be the first branching point. Countries with a health rating that's less than or equal to .65 go to the left, those who are greater than .66 go to the right. That means the healthiest people (or those with the longest life expectancy) go to the right and the unhealthiest people (or those who have a low life expectancy) go the left. 

# 51 countries fall into the less healthy category whereas 105 fall in the healthier category.

# The next split is based on GDP. One interesting thing to note is the dividing point between the two boxes. The left (less healthy) box splits at less than or equal to .501. The right (or healthier) box splits at less than or equal to 1.266. Those that life longer have significantly more money.

# The poorer, less health section doesn't split anymore. 29 of the 51 samples have a GDP score of less than or equal to .501 whereas 22 samples are above that number. The average happiness rating of the poorest, least healthy people is 4.009 while the second poorest, least healthy is above that at 4.577. The poorer, less healthy people are also less happy.

# One of the boxes in the healthier, richer section does split again so we'll focus on the one that doesn't split first. The richest people who also live the longest terminate on the third level. 33 of the 105 samples from the first split made it into this box. They have an average happiness score of 6.829.

# The other box (the second most healthy and rich) has 72 of the 105 from the first split and splits between according to GDP again. Those that have less than or equal to 1.003 GDP go to the left, the rest go to the right. Those with less money on this split number 35 from the 72 in the box above. They have a happinest average of 5.333. The slightly richer box numbers 37 and has a happiness score of 5.799.

y = happy[['Score']]
a = happy[['Social', 'Freedom']]
a_names = ['Social', 'Freedom']
cart02 = DecisionTreeRegressor(criterion = 'mse', max_leaf_nodes = 5).fit(a,y)
#cannot show from here, but can show on your computer
#export_graphviz(cart02, out_file = "/Users/zli/Downloads/894_813759_bundle_archive/2019happy_cart02.dot", feature_names=a_names)

# will show the tree:
# The top box shows that Health is going to be the first branching point. Countries with a health rating that's less than or equal to .65 go to the left, those who are greater than .66 go to the right. That means the healthiest people (or those with the longest life expectancy) go to the right and the unhealthiest people (or those who have a low life expectancy) go the left. 

# 51 countries fall into the less healthy category whereas 105 fall in the healthier category.

# The next split is based on GDP. One interesting thing to note is the dividing point between the two boxes. The left (less healthy) box splits at less than or equal to .501. The right (or healthier) box splits at less than or equal to 1.266. Those that life longer have significantly more money.

# The poorer, less health section doesn't split anymore. 29 of the 51 samples have a GDP score of less than or equal to .501 whereas 22 samples are above that number. The average happiness rating of the poorest, least healthy people is 4.009 while the second poorest, least healthy is above that at 4.577. The poorer, less healthy people are also less happy.

# One of the boxes in the healthier, richer section does split again so we'll focus on the one that doesn't split first. The richest people who also live the longest terminate on the third level. 33 of the 105 samples from the first split made it into this box. They have an average happiness score of 6.829.

# The other box (the second most healthy and rich) has 72 of the 105 from the first split and splits between according to GDP again. Those that have less than or equal to 1.003 GDP go to the left, the rest go to the right. Those with less money on this split number 35 from the 72 in the box above. They have a happinest average of 5.333. The slightly richer box numbers 37 and has a happiness score of 5.799.

y = happy[['Score']]
a = happy[['Social', 'Freedom']]
a_names = ['Social', 'Freedom']
cart02 = DecisionTreeRegressor(criterion = 'mse', max_leaf_nodes = 5).fit(a,y)
#cannot show from here, but can show on your computer
#export_graphviz(cart02, out_file = "/Users/zli/Downloads/894_813759_bundle_archive/2019happy_cart02.dot", feature_names=a_names)
# The first box uses Social as a branching point. Those who score less than or equal to 1.206 go left, those that are above that number go right.

# There are two boxes on the second row, 62 on the left and 94 on the right. The one with a lower Social rating has an average Happiness score of 4.385, the countries with the higher Social ranking have an average Happiness score of 6.081

# 62 of the 156 values go left. The box with the lower Social ranking branches once more, again using Social as a metric. Countries with a Social ranking less than or equal to 0.765 go left, the rest go right. 

# On the third line, we get the final Happiness scores for the lower Social countries. The least social have an average Happiness score of 3.776 while the slightly more social box is also happier: 4.563.

# The box on the second line with the higher social score also splits again, this time using Freedom as a metric. Countries with a Freedom rating of less than 0.515 go left, those that are higher go right.

# There are 67 countries with a happiness score above 1.206 and a Freedom score below 0.515. These countries have an average Happiness score of 5.791.

# The box on the third row that has both higher Social and Freedom rankings splits once cmore, this time using Social rating again. Those that have a Social score less than or equal to 1.47 go left, the rest go right.

# 11 samples go left. These have the second highest social score and the highest Freedom score. Their happiness rating is 6.245. 16 countries have the highest Social and Freedom score and their rating for Happiness is 7.182.

# So, the higher the Social and Freedom rankings, the happier people are. This doesn't disprove our theory that Wealth and Health make a person happy, but it does show other factors also directly contribute to a person's well being. 
y = happy[['Score']]
c = happy[[ 'Generosity', 'Corruption']]
c_names = happy[['Generosity', 'Corruption']]
cart03 = DecisionTreeRegressor(criterion = 'mse', max_leaf_nodes = 5).fit(c,y)
#cannot show from here, but can show on your computer
#export_graphviz(cart03, out_file = "/Users/zli/Downloads/894_813759_bundle_archive/2019happy_cart03.dot")
# Here's a decision tree with the final two variables: Corruption and Generosity. The trajectory is a little different. If you follow the divergent paths from top to bottom, there are five boxes that serve as ending points. All told, there are five different ending Corruption ratings and two different ending Generosity levels. 

# Here are the five ending states and their corresponding Happiness rating:

# A Corruption >0.288 has a Happiness rating of 7.378
# B Corruption <= 0.288 has a Happiness rating of 6.31
# C: Corruption <= 0.181, Generosity <=0.146 has a Happiness rating of 5.45
# D: Corruption <= 0.181, Generosity >0.146 has a Happiness rating of 4.992
# E: Corruption >0.411 has a Happiness rating of 4.798

# The most corrupt is the least happy.

#The summary shows that all independent variables have a significant impact

# When the entire dataset is used, only Generosity (.327) and Corruption (.075) are above the .05 cutoff.

# Ultimately, Corruption and Generosity have the least pull on a country’s overall Happiness level. 


#############################################################################
#MAEbaseline = 0.8949
MAEbaseline = np.mean(y1)
y2.shape
MAEbaseline_list = ([5.444917]*35)
MAE_test = met.mean_absolute_error(y2, MAEbaseline_list)



#############################################
# Random Forest Regressor

#Again, Random ForestClassifier and 'gini' are used for int data. Because Score out data is floating point (there are always decimals), we're using Random Forest Regressor
# This time we split our data set to train 78% & test 22% which is(121：35),in order to improve the performance of our small size data trainning.

happy_train2, happy_test2 = train_test_split(happy, test_size = 0.22, random_state = 7)
#### For overall 
cols = ['GDP', 'Health', 'Freedom','Social', 'Corruption', 'Generosity']
Xtrain_2 = happy_train2[cols]
ytrain_2 = happy_train2['Score']
# Test the max_depth with 1,2,3,4,5. and 4 performance the best result.( which is less error)
rf01 = RandomForestRegressor(max_depth=4, random_state = 0)
rf01.fit(Xtrain_2, ytrain_2)

#Create Test data set:        
Xtest_2 = happy_test2[cols]
ytest_2= happy_test2['Score']

# Use the test data to make predictions
rf26_pred = rf01.predict(Xtest_2)

MSE = np.sum(np.power(np.array(ytest_2) - np.array(rf26_pred), 2)) / len(ytest_2)
#MSE:0.3564

MAE_test = met.mean_absolute_error(ytest_2, rf26_pred)
# MAE is obtained by subtracting the predict data from the actual data : 0.4360

#Random Forest Regressor performance Visualization
plt.figure()
plt.scatter(rf26_pred, ytest_2)
plt.plot([3, 7], [3, 7], 'k')
plt.title('Random Forest Regressor& Overall')
plt.xlabel('Predict Happiness Score') 
plt.ylabel('Actual Happiness Score')


# We want to know the correlation of variables, so we do the test by taking out a variable, and observing whether the error is increasing or decreasing.

# Take out GDP, the MAE is: 0.4697 > overall MAE 0.4360
# Take out Health, the MAE is:0.4810  >overall MAE 0.436
# Take out Freedom, the MAE is:0.4386 > overall MAE 0.436
# Take out Social, the MAE is:0.4682 >overall MAE 0.436
# Take out Corruption, the MAE is:0.4387 >overall MAE 0.436
# Take out Generosity, the MAE is:0.4297< overall MAE 0.436

#By this comparison, the most relevant element should be the Health and GDP , after is Social & Freedom, and last is Corruption & Generosity.




###########################################################################
#  Multiple Regression model 
# Partitioned data use same partition with Random Forest Regressor
#Training data set
happy_train, happy_test = train_test_split(happy, test_size = 0.22, random_state = 7)
# No GDP
# Train data set
xtrain = pd.DataFrame(happy_train[['Health', 'Freedom', 'Social', 'Corruption', 'Generosity']])
xtrain = sm.add_constant(xtrain)
ytrain = pd.DataFrame(happy_train[['Score']])
model1 = sm.OLS(ytrain, xtrain).fit()
model1.summary()

# Test data set
xtest = pd.DataFrame(happy_test[['Health', 'Freedom', 'Social', 'Corruption', 'Generosity']])
xtest = sm.add_constant(xtest)
ytest = pd.DataFrame(happy_test[['Score']])
model2 = sm.OLS(ytest, xtest).fit()
model2.summary()

# Use the test data to make predictions
ypred = model1.predict(xtest)
MAEregression_test = met.mean_absolute_error(ytest, ypred)
# 0.4445515888853238

# No Health
# Train data set
xtrain2 = pd.DataFrame(happy_train[['GDP', 'Freedom', 'Social', 'Corruption', 'Generosity']])
xtrain2 = sm.add_constant(xtrain2)
ytrain2 = pd.DataFrame(happy_train[['Score']])
model3 = sm.OLS(ytrain2, xtrain2).fit()
model3.summary()
# test2 data set
xtest2 = pd.DataFrame(happy_test[['GDP', 'Freedom', 'Social', 'Corruption', 'Generosity']])
xtest2 = sm.add_constant(xtest2)
ytest2 = pd.DataFrame(happy_test[['Score']])
model4 = sm.OLS(ytest2, xtest2).fit()
model4.summary()

# Use the test2 data to make predictions
ypred2 = model3.predict(xtest2)
MAEregression_test2 = met.mean_absolute_error(ytest2, ypred2)
# 0.5010302403540907

# No Freedom

xtrain3 = pd.DataFrame(happy_train[['GDP', 'Health', 'Social', 'Corruption', 'Generosity']])
xtrain3 = sm.add_constant(xtrain3)
ytrain3 = pd.DataFrame(happy_train[['Score']])
model5 = sm.OLS(ytrain3, xtrain3).fit()
model5.summary()

# test3 data set
xtest3 = pd.DataFrame(happy_test[['GDP', 'Health', 'Social', 'Corruption', 'Generosity']])
xtest3 = sm.add_constant(xtest3)
ytest3 = pd.DataFrame(happy_test[['Score']])
model6 = sm.OLS(ytest3, xtest3).fit()
model6.summary()

# Use the test3 data to make predictions
ypred3 = model5.predict(xtest3)
MAEregression_test3 = met.mean_absolute_error(ytest3, ypred3)
# 0.461498861801825

# No Social

xtrain4 = pd.DataFrame(happy_train[['GDP', 'Health', 'Freedom', 'Corruption', 'Generosity']])
xtrain4 = sm.add_constant(xtrain4)
ytrain4 = pd.DataFrame(happy_train[['Score']])
model7 = sm.OLS(ytrain4, xtrain4).fit()
model7.summary()

# test4 data set
xtest4 = pd.DataFrame(happy_test[['GDP', 'Health', 'Freedom', 'Corruption', 'Generosity']])
xtest4 = sm.add_constant(xtest4)
ytest4 = pd.DataFrame(happy_test[['Score']])
model8 = sm.OLS(ytest4, xtest4).fit()
model8.summary()

# Use the test4 data to make predictions
ypred4 = model7.predict(xtest4)
MAEregression_test4 = met.mean_absolute_error(ytest4, ypred4)
# 0.4626210775277544

# No Corruption 

xtrain5 = pd.DataFrame(happy_train[['GDP', 'Health', 'Freedom', 'Social', 'Generosity']])
xtrain5 = sm.add_constant(xtrain5)
ytrain5 = pd.DataFrame(happy_train[['Score']])
model9 = sm.OLS(ytrain5, xtrain5).fit()
model9.summary()

# test5 data set
xtest5 = pd.DataFrame(happy_test[['GDP', 'Health', 'Freedom', 'Social', 'Generosity']])
xtest5 = sm.add_constant(xtest5)
ytest5 = pd.DataFrame(happy_test[['Score']])
model10 = sm.OLS(ytest5, xtest5).fit()
model10.summary()

# Use the test5 data to make predictions
ypred5 = model9.predict(xtest5)
MAEregression_test5 = met.mean_absolute_error(ytest5, ypred5)
# 0.4462581703777429

# No Generosity 
xtrain6 = pd.DataFrame(happy_train[['GDP', 'Health', 'Freedom', 'Social', 'Corruption']])
xtrain6 = sm.add_constant(xtrain6)
ytrain6 = pd.DataFrame(happy_train[['Score']])
model11 = sm.OLS(ytrain6, xtrain6).fit()
model11.summary()

# test6 data set
xtest6 = pd.DataFrame(happy_test[['GDP', 'Health', 'Freedom', 'Social', 'Corruption']])
xtest6 = sm.add_constant(xtest6)
ytest6 = pd.DataFrame(happy_test[['Score']])
model12 = sm.OLS(ytest6, xtest6).fit()
model12.summary()

# Use the test6 data to make predictions
ypred6 = model11.predict(xtest6)
MAEregression_test6 = met.mean_absolute_error(ytest6, ypred6)
0.44653839713541876

xtrain7 = pd.DataFrame(happy_train[['GDP', 'Health', 'Freedom', 'Social', 'Corruption', 'Generosity']])
xtrain7 = sm.add_constant(xtrain7)
ytrain7 = pd.DataFrame(happy_train[['Score']])
model13 = sm.OLS(ytrain7, xtrain7).fit()
model13.summary()

# test7 data set
xtest7 = pd.DataFrame(happy_test[['GDP', 'Health', 'Freedom', 'Social', 'Corruption', 'Generosity']])
xtest7 = sm.add_constant(xtest7)
ytest7 = pd.DataFrame(happy_test[['Score']])
model14 = sm.OLS(ytest7, xtest7).fit()
model14.summary()

# Use the test7 data to make predictions
ypred7 = model13.predict(xtest7)
MAEregression_test7 = met.mean_absolute_error(ytest7, ypred7)
0.4433334554668311

#Multiple Regression model performance Visualization
plt.figure()
plt.scatter(y2, y2pred)
plt.plot([3, 7], [3, 7], 'k')
plt.title('Multiple Regression model & Overall')
plt.xlabel('Predict Happiness Score') 
plt.ylabel('Actual Happiness Score')


#So our predicted values are Health and Social by Multiple Regression model
#The predicted values from Random Forest Regressor are Health and GDP

#Two model have a same result which is Health is No.1 reason of happy
# Our Hypothesis is Health and GDP, which is close.

#############################################################################