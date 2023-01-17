# There are lot of dataset's on kaggle. To make a difference i need to create a creative dataset, this made me to think out of the box and finally got crazy idea.
# I create a dataset on kaggle datasets, which sounds wow right!
# Main mottto is to share knowledge by pratical projection. Below i created a good stuff, please have a look at it.


#Loading libraries
# linear algebra
import numpy as np 
# data processing
import pandas as pd 
#data visuvalisation
import matplotlib.pyplot as plt
import seaborn as sns

#Path to access file
import os
print(os.listdir("../input"))
#Reading the file
kaggle_df = pd.read_csv('../input/KaggelMostVotes.csv')

#To get count of rows |& columns
kaggle_rows,kaggle_cols = kaggle_df.shape[0],kaggle_df.shape[1]
# (1960, 15)

#Getting all columns
kaggle_df.columns
#Getting the information
kaggle_df.info()
#Remove Dublicates - it will remove duplicate row 
kaggle_df.drop_duplicates()
# To remove by colomn kaggle_df.drop_duplicates(subset='col_name')
# right now there are no duplicates - check it by len(kaggle_df)
#checking the NAN or null data & cleaning it
# kaggle_df.isnull()  Feel free to use this also
kaggle_df.isnull().sum()
#Right now there is no null (or) NAN data nut below show how to replace it by relevent data 

#you can  count the number of unique values from character variables by -  kaggle_df.Discussions.value_counts(sort=True)

#For Discussion
kaggle_df.Discussions.fillna('0',inplace=True)

#For Views
kaggle_df.Views.fillna('0',inplace=True)

#Now you can cross check the data 
kaggle_df.isnull().sum()
# If you see the dat some of the columns are in correct datatype like Kernels,Discussions,views
kaggle_df.info()
#You change the datatype of columns 

#Changing Version column to number datatype
  
# you can  change NAN value by pass the value in fillna function  
kaggle_df['Version'] = kaggle_df['Version'].astype(int).fillna(0)

# #Changing Views column to number datatype
kaggle_df['Discussions'] = kaggle_df['Discussions'].astype(float).fillna(0)

#You can change it to float as
# kaggle_df['Views'] = kaggle_df['Views'].astype(float).fillna(0)

kaggle_df.info()
#Below th dataype of  Version, Discussions is changed
# To simply sort by column 
kaggle_df.Version.sort_values()
#SNO version Number - below results are just displaying the verions in sorted way
#You can also group by coloumns
kaggle_df.groupby('FileType').count()
#Here we see how to by count by certain range 
version_length = [1,5,10,25,50,75,100,500,1000]
kaggle_version_ranges = pd.cut(kaggle_df.Version, version_length)
pd.value_counts(kaggle_version_ranges)
#Below result is showing the count of how many versions are bewtten - 1 to 5, 5 to 10, 10 to 25, 25 to 50 and so on....
# "correlation" refers to a mutual relationship or association between quantities
# Correlation can help in predicting one quantity from another
kaggle_df.corr()
#crosstab display the each elemnt from one  column and count how many time it occuried in second coloumn
kaggle_crosstab = pd.crosstab(kaggle_df['Updated on'],kaggle_df.FileType,margins=True)
kaggle_crosstab
# To display in percentage
# kaggle_crosstab = pd.crosstab(kaggle_df['Updated on'],kaggle_df.FileType,margins=True)/kaggle_df.shape[0]
# kaggle_crosstab

# For Total count 
# kaggle_crosstab.max() 
# Total crosstab list
kaggle_crosstab.max()
#list all columns where Discussions is greater than Version
kaggle_df.query('Discussions > Version')

#using AND condition
kaggle_df.query('500 < Votes & Discussions > Version')

#using OR condition
kaggle_df.query('Version < Discussions | 500 < Votes ')
kaggle_df.index.values
#Here we are going how the data changes from versions and votes

# to get indes of dataframe 
kaggle_SNO= kaggle_df.index.values
# kaggle_df.index.values
kaggle_fig, kaggle_axes = plt.subplots(nrows=2,ncols=1,figsize=(22, 16),) 

#NOTE
# lw -linewidth; ls - linestyle
# For more styling https://matplotlib.org/examples/lines_bars_and_markers/line_styles_reference.html

#For the first plot
kaggle_axes[0].plot(kaggle_SNO[:101],kaggle_df['Version'][:101],color='blue',lw=3,ls=':',marker='o',markersize=10,markerfacecolor='blue')
kaggle_axes[0].set_xlabel('Dataset Version')
kaggle_axes[0].set_ylabel('Datasets SNO ')
kaggle_axes[0].set_title('SNO VS VERSIONS')

#For the first plot
kaggle_axes[1].plot(kaggle_SNO[:101],kaggle_df['Votes'][:101],color='blue',lw=3,ls=':',marker='o',markersize=10,markerfacecolor='blue')
kaggle_axes[1].set_xlabel('Dataset Votes')
kaggle_axes[1].set_ylabel('Datasets SNO ')
kaggle_axes[1].set_title('SNO VS Votes')

#Finaly display
plt.show()

#Represeting version ranges in pie chart
version_length = [1,3,5,10,20,50,500]
kaggle_version_ranges = pd.cut(kaggle_df.Version, version_length)
kaggle_version_cluster=  pd.value_counts(kaggle_version_ranges)

#Getting indexs
kaggle_pie_labels = kaggle_version_cluster.index.values
#Getting data
kaggle_pie_data = kaggle_version_cluster.values

#  = 
# slices_hours = [4, 8]
# activities = ['Sleep', 'Work']
plt.pie(kaggle_pie_data, labels=kaggle_pie_labels,startangle=90, autopct='%.1f%%',radius=3)

#For more colors and styling 
plt.show()

#Heatmap to show the correlation
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(kaggle_df.corr())
plt.show()
#Display plot by number of count
sns.countplot(x='Version',data=kaggle_df.tail(20))
#Display plot by number of count and classify with filetype
sns.countplot(x='Version',data=kaggle_df.tail(20),hue='FileType')
# Thank you for reading...