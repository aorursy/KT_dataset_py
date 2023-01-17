
## Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from pprint import pprint
## Importing Data Set
A = pd.read_csv('../input/Original_Dataset.csv')
## Displaying the Data Set's top row
A.head()
## Identifying data frame's attribute data types
A.dtypes
## Seperating numeric columns out into a seperate table
numeric_table = A._get_numeric_data()
numeric_table.head()
## Seperating all other data types out into a seperate table 
others_table = A.select_dtypes(exclude=['number'])
others_table.head()
## Writing the two seperated tables into two seperate CSV files
numeric_table.to_csv('Quantitative.csv', index=False)
others_table.to_csv('Others.csv', index=False)
## Importing Data Set
Numeric = pd.read_csv('Quantitative.csv')
Numeric.head(2)
## Summary Table w/ Descriptive Statistics for Continuous Features
Numeric.describe()
## Develop Equal-width Histograms
for name, series in Numeric.iteritems():
    plt.hist(Numeric[name], bins=100)
    plt.title(name)
    plt.show()
## Develop Horizontal Violin Plots
for name, series in Numeric.iteritems():
    plt.violinplot(Numeric[name], 
                      points=30,
                      vert=False,
                      widths=0.5, 
                      showmeans=True, 
                      showextrema=True, 
                      showmedians=True)
    plt.title(name)
    plt.show()
## Develop Scatter Plot Matrix
SPM_numeric = Numeric


sns.pairplot(SPM_numeric)
## Develop a Covariance Table & a Heat Map
covariance = SPM_numeric.cov()

sns.heatmap(covariance)
    
covariance
## Develop Correlation Table & a Heat Map
correlations = SPM_numeric.corr()

sns.heatmap(correlations)
    
correlations
## Importing Data Set
Quant_df = pd.read_csv('Quantitative.csv')
Numeric.head(2)
## IQR Methodology
 # Boxplots for identifying outliers visually
for (name, series) in Quant_df.iteritems():
    plt.boxplot(Quant_df[name], vert=False)
    plt.title(name)
    plt.show()
## Identifying outliers 
Q1 = Quant_df.quantile(0.25)
Q3 = Quant_df.quantile(0.75)
IQR = Q3 - Q1

outliers = Quant_df[((Quant_df < (Q1 - 1.5 * IQR)) | (Quant_df > (Q3 + 1.5 * IQR)))]

outliers_dictionary = {}

for name, series in outliers.iteritems():
    
    column = list(outliers[name])
    attribute_outliers = []
    
    for i in range(len(column)):
        
        if str(column[i]) != 'nan':
            
            index_value = (i, column[i])
            attribute_outliers.append(index_value)
            
    outliers_dictionary[name] = attribute_outliers

pprint(outliers_dictionary)
## Removing Outliers into a 'Filtered' Dataset
Quant_df_filtered = Quant_df[~((Quant_df < (Q1 - 1.5 * IQR)) | (Quant_df > (Q3 + 1.5 * IQR))).any(axis=1)]
## Labeling Filtered Data Set's Column Names
Quant_df_filtered.columns = [str(col) + '_ClampedValues' for col in Quant_df.columns]
## Labeling Filtered Data Set's Column Names
Norm_columns              = Quant_df.columns
Norm_columns              = [str(col) + '_ClampedNormalizedValues' for col in Norm_columns]

## Normalizing the data set via 'Min-Max' Methodolgy
scaler                    = preprocessing.MinMaxScaler()   # preprocessing is the library / minmaxscaler is the function
Norm_Quant_df_filtered    = pd.DataFrame(scaler.fit_transform(Quant_df_filtered), columns = Norm_columns)

## Generating BoxPlots on Normalized Filtered Data
for name, series in Norm_Quant_df_filtered.iteritems():
    plt.boxplot(Norm_Quant_df_filtered[name], vert = False)
    plt.title(name)
    plt.show()
# Generating Scatter Plot Matrix on Normalized Filtereed Data
sns.pairplot(Norm_Quant_df_filtered)
plt.show()
# Generating Covariance table on Normalized Filtered Data
covariances_norm = Norm_Quant_df_filtered.cov()
covariances_norm
## Generating Heatmap on Normalized Filtered Data
sns.heatmap(covariances_norm)
plt.show()
## Generating Correlation table and Heatmap on Normalized Data
correlations_norm = Norm_Quant_df_filtered.corr()
correlations_norm
## Generating Heatmap for the Normalized Filtered Data
sns.heatmap(correlations_norm)
plt.show()
## Joining Filtered data Columns and Normalized Filtered Data Columns into one Table
initial_df  = Quant_df.join(Quant_df_filtered)
final_df    = initial_df.join(Norm_Quant_df_filtered)

## Sorting Columns
filtered_columns = Quant_df_filtered.columns
original_columns = Quant_df.columns
sorted_columns   = []
for col0, col1, col2 in zip(original_columns, filtered_columns, Norm_columns):
    sorted_columns.append(col0)
    sorted_columns.append(col1)
    sorted_columns.append(col2)
final_df = final_df[sorted_columns]
final_df
# Generating a csv file of the final dataframe
final_df.to_csv('QTransferred.csv', index=False)
## Importing Data Set
other = pd.read_csv('Others.csv')
other.head(10)
## Generating Data Quality Report, Summary Table for The Categorical Features
other.describe()
## Idenitfying the Attributes' Data Types
other.dtypes
binned_df = pd.read_csv('Quantitative.csv')
binned_df
for name, series in binned_df.iteritems():
    sorted_column = list(binned_df[name].sort_values())
    x = 25
    for i in range(0, len(sorted_column), x):
        bins = list()
        for j in range(i, i + x):
            bins.append(sorted_column[j])
        avg_bin = (min(bins) + max(bins)) / 2
        for k in range(i, i + x):
            binned_df.loc[k, name + '_Binned'] = avg_bin

sorted_cols = []
cols        = binned_df.columns
length      = int(len(cols)/2)
length
for i in range(length):
    sorted_cols.append(cols[i])
    sorted_cols.append(cols[i+length])
Final_df = binned_df[sorted_cols]
Final_df
# Generating Scatter Plot Matrix on binned data set
sns.pairplot(binned_df)
plt.show()
## Develop a Covariance Table & a Heat Map
covariance = binned_df.cov()

sns.heatmap(covariance)
covariance
## Develop Correlation Table & a Heat Map
correlations = binned_df.corr()

sns.heatmap(correlations)
    
correlations
Final_df.to_csv('QuantitativeBinned.csv', index=False)
