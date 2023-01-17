# Importing Packages and Libraries

#Essential
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#Ploting/Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Ignore useless warnings
import warnings
warnings.filterwarnings(action="ignore")
pd.options.display.max_seq_items = 8000
pd.options.display.max_rows = 8000

#Model
import statsmodels.formula.api as smf

#Model Evaluation
import sklearn.metrics as metrics
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif

#Dataset Path, which I uploaded to Data folder.
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        print('Copy the above path and paste this in your read_csv(), to load the dataset as pandas dataframe.')
#Reading the dataset as a DataFrame
#Lets give our dataframe name -> marketing

marketing = pd.read_csv('/kaggle/input/mktmix.csv')
#Top 5 records of the marketing DataFrame

marketing.head()
#Shape of the data that we are deaing with

#marketing.shape

print("NO. Of. Rows = %s" % marketing.shape[0])
print("NO. Of. Columns = %s" % marketing.shape[1])
#Summary Statistics

#Using pandas options to set float_format to 2 decimals after the point.
pd.options.display.float_format = '{:.2f}'.format   #This makes the table clear and easy to understand

marketing.describe()
#Useful, detailed information of the marketing dataset

marketing.info()
sns.set_style("white")
sns.set_color_codes(palette='deep')
f, ax = plt.subplots(figsize=(8, 7))

#Check the new distribution 
sns.distplot(marketing['NewVolSales'], color="darkgreen");
ax.xaxis.grid(True)
ax.yaxis.grid(True)
ax.set(ylabel="Frequency")
ax.set(xlabel="NewVolSales")
ax.set(title="NewVolSales Distribution")
sns.despine(trim=True, left=True)
plt.show();
# Skew and kurt
print("Skewness: %f" % marketing['NewVolSales'].skew())
print("Kurtosis: %f" % marketing['NewVolSales'].kurt())
#Finding numeric features

marketing.select_dtypes(include = ['float64', 'int64']).columns
#Assigning a variable name to the list of numeric cols in df

num_features = marketing.select_dtypes(include = ['float64', 'int64']).columns.tolist()

num_features
#Visualising num_cols data from marketing dataframe

marketing[num_features].hist(figsize = (12,8), bins = 10, xlabelsize = 8, ylabelsize = 8, color= 'purple');
marketing.columns
marketing = marketing.rename(columns = { 'NewVolSales' : 'NewVolSales', 
                  'Base_Price' : 'Base_Price', 
                  'Radio ':  'Radio', 
                  'InStore' : 'InStore', 
                  'NewspaperInserts' : 'NewspaperInserts', 
                  'Discount' : 'Discount' , 
                  'TV' : 'TV', 
                  'Stout' : 'Stout', 
                  'Website_Campaign ':  'Website_Campaign'})
marketing.columns
marketing.isnull().sum()
#Imputing Median for NaN in Radio

marketing.Radio = marketing.Radio.fillna(marketing['Radio'].median())
#One Hot Encoding on NewspaperInserts

marketing = pd.get_dummies(marketing, columns = ["NewspaperInserts"])
#One Hot Encoding on Website_Campaign


marketing = pd.get_dummies(marketing, columns = ["Website_Campaign"])
#Top 5 records with new columns after encoding

marketing.head()
#Total no. of. columns now, after encoding: 

marketing.shape[1]
#Column name after encoding:

marketing.columns.tolist()
#lets now check for NaN values, if there are any left untreated.

marketing.isnull().sum()
#Boxplot for Base_Price Outliers

sns.set_style("white")
sns.set_color_codes(palette='deep')
f, ax = plt.subplots(figsize=(5, 4))

sns.boxplot('Base_Price', data = marketing, orient = 'v', color = 'darkgreen')
ax.set(title="Base_Price Boxplot")
plt.show();
#Minimum Value

print("Minimum Value of 'Base_Price' is %s" % marketing['Base_Price'].quantile(0.01))

#The records in which Base_Price has outliers

marketing[marketing['Base_Price'] < marketing['Base_Price'].quantile(0.01)]
marketing.loc[(marketing['Base_Price'] < marketing['Base_Price'].quantile(0.01)), "Base_Price"]= marketing['Base_Price'].quantile(0.01)
#Boxplot for Radio Outliers

sns.set_style("white")
sns.set_color_codes(palette='deep')
f, ax = plt.subplots(figsize=(5, 4))

sns.boxplot('Radio', data = marketing, orient = 'v', color = 'yellow')
ax.set(title="Radio Boxplot")
plt.show();
marketing['Radio'].quantile(np.arange(0,1,0.05))
#Minimum Value

print("Minimum Value of 'Radio' is %s" % marketing['Radio'].quantile(0.10))

#The records in which Base_Price has outliers

marketing[marketing['Radio'] < marketing['Radio'].quantile(0.10)]
#Count of such weeks

marketing[marketing['Radio'] < marketing['Radio'].quantile(0.10)].shape[0]
marketing.columns
marketing['Online'] = marketing['Website_Campaign_Facebook']+ marketing['Website_Campaign_Twitter']+ marketing['Website_Campaign_Website Campaign ']
marketing["Offline"] = marketing['TV'] + marketing['InStore'] + marketing['Radio']
marketing.columns
corr = marketing.corr()

#Only the reltion coefficients between all other features to NewVolSales.
corr = corr.NewVolSales 

corr = corr.drop('NewVolSales')# Because we dont need the correlation NewVolSales - NewVolSales.

corr[abs(corr).sort_values(ascending = False).index] #ascenfing order irrespective of their sign
corr = marketing.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.index.values);
sns.pairplot(data = marketing)
corr = marketing.drop('NewVolSales', axis=1).corr() # We already examined SalePrice correlations
plt.figure(figsize=(15, 20))


ax = sns.heatmap(corr[(corr >= 0.6) | (corr <= -0.8)], 
            cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,
            annot=True, annot_kws={"size": 8}, square=True)

ax.xaxis.grid(True)
ax.yaxis.grid(True)
plt.show();
# relation to the target
fig = plt.figure(figsize = (12,7))
for i in np.arange(11):
    ax = fig.add_subplot(5,5,i+1)
    sns.regplot(x = marketing.iloc[:,i], y = marketing.NewVolSales, color = 'orange')

plt.tight_layout()
plt.show();
reg_model = smf.ols("NewVolSales~Base_Price+InStore+TV+Discount+Stout",data=marketing) 
#ols - Ordinary Least Square model
#OLS fits the linear regression model with Ordinary Least Squares

#Fitting the model

results = reg_model.fit()

print(results.summary())
#Prediction on marketing data

pred = results.predict(marketing)

#Actaul values of marketing.NewVolSales

actual = marketing.NewVolSales
## Actual vs Predicted plot
plt.plot(actual,"blue")
plt.plot(pred,"green")
plt.figure(figsize=(70,50));
residuals = results.resid

residuals.head()
plt.scatter(actual, residuals);
plt.scatter(pred, residuals);
for i in range(1, len(marketing.columns[:])):
    v = vif(np.matrix(marketing[:]), i)
    print("Variance inflation factor for {}: {}".format(marketing.columns[i], round(v, 2)))
MAE = metrics.mean_absolute_error(actual,pred)

MAE
np.mean(abs((pred-actual)/actual))*100