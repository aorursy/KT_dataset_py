# Data analysis packages:
import pandas as pd
import numpy as np
#from datetime import datetime as dt

# Visualization packages:
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
## Reading the dataset file name:
import os
print(os.listdir("../input"))
pd.read_csv('../input/KaggleV2-May-2016.csv').head()
## Loading the dataset and printing out a few lines:
dataset = pd.read_csv('../input/KaggleV2-May-2016.csv')
dataset.head(3)
## Reading dataset general information:
dataset.info()
## Describing the numerical attributes:
dataset.describe()
## Checking the attribute data type:
type(dataset['PatientId'][0])
## Converting the values to int type and then to str type:
dataset['PatientId'] = dataset['PatientId'].apply(lambda x: str(int(x)));
## Counting how many unique patients are in the dataset:
len(dataset['PatientId'].unique())
## Checking the attribute data type:
type(dataset['AppointmentID'][0])
## Converting the values to int type and then to str type:
dataset['AppointmentID'] = dataset['AppointmentID'].apply(lambda x: str(int(x)));
## Counting how many unique patients are in the dataset:
len(dataset['AppointmentID'].unique())
dataset.set_index('AppointmentID', drop=True, inplace=True)
dataset[dataset['Age']<0]
dataset.drop('5775010',inplace=True)  #Removing the anomalous instance
# dataset.reset_index(drop=True,inplace=True)  #Reseting the dataset index
## Converting all 'Handcap' values higher than 0 to 1:
dataset['Handcap'] = np.where(dataset['Handcap']>0, 1, 0)
## Getting information of the categorical attributes:
dataset.info()
## Counting gender classes
dataset.Gender.value_counts()
## Reading again the dataset first lines to get acquainted with its content:
dataset.head(2)
## Converting the date information in string to datetime type:
dataset['ScheduledDay'] = pd.to_datetime(dataset.ScheduledDay)
dataset['AppointmentDay'] = pd.to_datetime(dataset.AppointmentDay)
## Creating a new column (attribute) containing just the scheduling time:
dataset['ScheduleTime'] = dataset.ScheduledDay.dt.time
## Normalizing the "Day" columns to keep just the date information (dropping the time info)
dataset['ScheduledDay'] = dataset.ScheduledDay.dt.normalize()
## Since both 'AppointmentDay' and 'ScheduledDay' are pandas.Timestamp type, this operation can be done directly:
dataset['WaitingDays'] = dataset['AppointmentDay'] - dataset['ScheduledDay']
def waiting_days(days):
    '''Auxiliary function to parse a date information from string type to python datetime object.
    Syntax: waiting_days(days), where:
        days = int type with the number of days considered.
    Return: a correspondent pandas._libs.tslib.Timedelta data type.
    '''
    arg = str(days) + ' days'
    return pd.tslib.Timedelta(arg)
## Checking which instances were scheduled after the appointment:
dataset[dataset['WaitingDays'] < waiting_days(0)]
## Recording the inconsistent instances index 
dropIx = dataset[dataset['WaitingDays'] < waiting_days(0)].index
## Dropping these instances from the dataset:
dataset.drop(dropIx, inplace=True)
dataset['WaitingDays'] = dataset.WaitingDays.dt.days  #Extract just the day value from the full "timedelta" object.
## Grouping by the 'WaitingDays' and 'No_show' values:
waitingdays = dataset.groupby(by=['WaitingDays','No-show'])
waitingdays = waitingdays.count()['PatientId'].unstack()
waitingdays.fillna(value=0, inplace=True)
waitingdays.reset_index(drop=False, inplace=True)
waitingdays.info()
## Defining the categories label:
categories = pd.Series(['Same day: 0', 'Short: 1-3', 'Week: 4-7', 'Fortnight: 8-15', 'Month: 16-30', 'Quarter: 31-90', 'Semester: 91-180', 'Very long: >180'])
## Applying these categories both to the auxiliary and to the working datasets:
waitingdays['WaitingDays'] = pd.cut(waitingdays.WaitingDays, bins = [-1,0,3,7,15,30,90,180, 10000], labels=categories)
dataset['WaitingCategories'] = pd.cut(dataset.WaitingDays, bins = [-1,0,3,7,15,30,90,180, 10000], labels=categories)
## Grouping the dataset by the waiting categories, returning the sum of all instances:
waitingdays = waitingdays.groupby('WaitingDays').sum()
## Creating a new attribute, "No-showing rate", relating how many patients did not show up against those who did.
waitingdays['No-showing rate'] = (waitingdays.Yes / waitingdays.No)*100
## Viewing the resulting dataset:
waitingdays
## Checking the unique neighborhood names:
neighborhood = dataset.Neighbourhood.unique()
neighborhood.sort()  #Sorting the names in alphabetical order
neighborhood  #Showing the results
dataset.drop(dataset[dataset['Neighbourhood'] == 'ILHAS OCEÂNICAS DE TRINDADE'].index, inplace=True)
## Counting again the neighborhood number:
neighborhood = dataset.Neighbourhood.unique()
neighborhood.sort()
## Counting neighborhood:
len(neighborhood)
## Plotting an histogram with the neighborhoods sorted alphabetically. 
plt.figure(figsize=(16,6))
ax = sns.countplot(x='Neighbourhood', data=dataset, order=neighborhood)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=8)
plt.title('Distribution of appointments per neighborhood', fontsize=14, fontweight='bold')
plt.show()
## Counting gender classes
dataset['No-show'].value_counts()
## Reading the dataset attributes (columns):
dataset.columns
dataset = dataset.reindex(columns=['PatientId', 'Gender', 'Age', 'Scholarship', 'Hipertension', 'Diabetes',
       'Alcoholism', 'Handcap', 'ScheduledDay', 'ScheduleTime', 'AppointmentDay', 'WaitingDays', 'WaitingCategories', 'SMS_received', 
       'Neighbourhood', 'No-show'])
## Reading again the current attribute labels:
dataset.columns
## Renaming "No-show"to "No_show"; "Handcap" to "Handicap"; and "ScheduleTime" to "ScheduledTime":
dataset.columns = ['PatientId', 'Gender', 'Age', 'Scholarship', 'Hipertension', 'Diabetes',
       'Alcoholism', 'Handicap', 'ScheduledDay', 'ScheduledTime', 'AppointmentDay', 'WaitingDays', 
       'WaitingCategories', 'SMS_received', 'Neighbourhood', 'No_show']
## Checking again the dataset information (for numerical attributes) and description (for categorical ones):
print(dataset.info())
dataset.describe()
## Visualizing few instances of the data:
dataset.head(3)
def get_statistics(data, bins=20):
    '''Prints basic statistics from the input data. 
    Syntax: get_statistics(data, bins=20), where:
        data = the input data series;
        bins = the number of bins to the histogram.
    '''
    total = data.values
    print('Mean:', np.mean(total))
    print('Standard deviation:', np.std(total))
    print('Minimum:', np.min(total))
    print('Maximum:', np.max(total))
    print('Median:', np.median(total))
    plt.hist(data, bins=bins);
def get_total(dataframe):
    '''Return the total sum of each numerical attribute of a pandas.Dataframe.'''
    return dataframe.sum(axis=1)
def df_row_normalize(dataframe):
    '''Normalizes the values of a given pandas.Dataframe by the total sum of each line.
    Algorithm based on https://stackoverflow.com/questions/26537878/pandas-sum-across-columns-and-divide-each-cell-from-that-value'''
    return dataframe.div(dataframe.sum(axis=1), axis=0)
def df_column_normalize(dataframe, percent=False):
    '''Normalizes the values of a given pandas.Dataframe by the total sum of each column.
    If percent=True, multiplies the final value by 100.
    Algorithm based on https://stackoverflow.com/questions/26537878/pandas-sum-across-columns-and-divide-each-cell-from-that-value'''
    if percent:
        return dataframe.div(dataframe.sum(axis=0), axis=1)*100
    else:
        return dataframe.div(dataframe.sum(axis=0), axis=1)
get_statistics(dataset.WaitingDays)
## Showing the data again:
waitingdays
## Adjusting the dataframe:
eda_waitingDays = waitingdays.copy()  #Copying the dataframe from Section 2.3.3
eda_waitingDays.reset_index(drop=False, inplace=True)  #Making the index as a column in order to be plotted.
eda_waitingDays.drop(7, inplace=True)  #Droppping the last row, since it's empty.

## Adding new columns:
#Transforming the 'No-showing rate' into strings with the percentual values:
eda_waitingDays['No-show percentual'] = eda_waitingDays['No-showing rate'].apply(lambda x: '{0:.2f}%'.format(x))
#Multiplying the rate values by 500 times in order to be plotted in the same scale:
eda_waitingDays['No-showing rate (500x)'] = eda_waitingDays['No-showing rate']*500

## Showing the adjusting dataframe:
eda_waitingDays
## Setting the graph parameters:
fig1, ax = plt.subplots(figsize=[12,6])  #Defines the graph window size
fig1.subplots_adjust(top=0.92)
plt.suptitle('Appointments distribution by waiting time categories', fontsize=14, fontweight='bold')

colors = ['tab:blue', 'tab:green', 'tab:red']  #Defines the colors to be used

ax.set_ylabel('Number of occurences', color=colors[0], fontsize=12)  #Set the y-axis color and label
ax.tick_params(axis='y', labelcolor=colors[0])

## Plotting the line chart:
eda_waitingDays[['WaitingDays', 'No-showing rate (500x)']].plot(x='WaitingDays', linestyle='-', marker='o', ax=ax, color=colors[2])
#Setting the line chart marker labels
x = ax.get_xticks()  #Getting the x-axis ticks to plot the label
for a,b,c in zip(x,eda_waitingDays['No-showing rate (500x)'], eda_waitingDays['No-show percentual']):
    plt.text(a,b+1500,c, color='red', fontsize=14)
    
## Plotting the bar chart:
eda_waitingDays[['WaitingDays', 'No', 'Yes']].plot(x='WaitingDays', kind='bar', ax=ax, color=colors[0:2])

ax.set_xlabel('Waiting time categories', fontsize=12)  #Set the y-axis color and label

plt.show()
## Group I - Describing the numerical attributes for the same day appointments:
group_I = dataset[dataset['WaitingCategories'] == 'Same day: 0'].describe()
group_I
## Group II - Describing the numerical attributes for the semester appointments:
group_II = dataset[dataset['WaitingDays']>90].describe()
group_II
def find_differences(serie1, serie2, pct_diff):
    '''Given two data series [serie1, serie2], compare those attributes and return 
    those who difference among them is higher than pct_diff (e.g. 50% must be entered as 0.5).
    The index of both series must be identical.
    '''
    try:
        if (serie1.index.all() == serie2.index.all()):
            ## Calculating the differences
            testA = serie1 / serie2
            testB = serie2 / serie1
            checkA = [x for x in testA if (x > pct_diff)&(x<1)]
            checkB = [x for x in testB if (x > pct_diff)&(x<1)]
            
            ## Showing which attributes in serie1 are less than "pct_diff" of those in serie1:
            print('Attributes in "Serie I" whose values are less than {0:.1f}% of those in "Serie II":'.format(pct_diff*100))
            for item in checkA:
                print('\t{0}: {1:.1f}%'.format(testA[testA == item].index[0], item*100))
            
            ## Showing which attributes of serie2 are "pct_diff" higher in serie1:
            print('Attributes in "Serie II" whose values are less than {0:.1f}% of those in "Serie I":'.format(pct_diff*100))
            for item in checkB:
                print('\t{0}: {1:.1f}%'.format(testB[testB == item].index[0], item*100))
    except ValueError:
        print('The series must have same index and length!')
    return    
find_differences(group_I.loc['mean'], group_II.loc['mean'], 0.30)
## Using the pandas.groupby() method to generate a pivot table:
neighbors_I = dataset.groupby(by='Neighbourhood').No_show.value_counts().sort_index()
## Manipulating the data:
neighbors_I = neighbors_I.unstack()  #Converting the groupby object into a dataset
neighbors_I.fillna(value=0, inplace=True)  #Replacing NaN values by zero
print(neighbors_I.head(3))
## Normalizing the data using a predefined function:
normalNeighbor = df_row_normalize(neighbors_I)
print(normalNeighbor.head(3))
## Getting the normalized data statistics:
normalNeighbor.describe()
## Adding a total column:
neighbors_I['Total'] = get_total(neighbors_I)
normalNeighbor['Total'] = get_total(normalNeighbor)
#Reseting the 'neighbourhood' index and making it as a column:
neighbors_I.reset_index(inplace=True)  
normalNeighbor.reset_index(inplace=True)
## Initialize the matplotlib figure:
fig2, (ax1, ax2) = plt.subplots(1,2, figsize=(12,16), sharey=False)
fig2.tight_layout()  #When working with 'tight_layout', the subplot must be adjusted [https://stackoverflow.com/questions/7066121/how-to-set-a-single-main-title-above-all-the-subplots-with-pyplot]
fig2.subplots_adjust(top=0.96)  #Adjusting the space for the superior title

## Plot the relative absence by neighborhood
#Total appointments
sns.set_color_codes("pastel")
sns.barplot(x="Total", y="Neighbourhood", data=normalNeighbor, label="Total", color="b", ax=ax1)
#Attended appointments
sns.set_color_codes("muted")
sns.barplot(x="No", y="Neighbourhood", data=normalNeighbor, label="Attended", color="b", ax=ax1)
## Add a legend and informative axis label
ax1.legend(ncol=2, loc="lower left", frameon=True)
ax1.set(xlim=(0, 1), ylabel="", xlabel="Relative attended appointments by neighborhood")
sns.despine(left=True, bottom=True,ax=ax1)

## Plot the absolute absence by neighborhood
#Total appointments
sns.set_color_codes("pastel")
sns.barplot(x="Total", y="Neighbourhood", data=neighbors_I, label="Total", color="b",ax=ax2)
#Attended appointments
sns.set_color_codes("muted")
sns.barplot(x="No", y="Neighbourhood", data=neighbors_I, label="Attended", color="b", ax=ax2)
## Add a legend and informative axis label
ax2.legend(ncol=2, loc="lower right", frameon=True)
ax2.set(xlim=(0, 7720), ylabel="", xlabel="Absolute attended appointments by neighborhood")  #The xlim value comes from the maximum value in the dataset.
ax2.set_yticklabels([''])
sns.despine(left=True, bottom=True, ax=ax2)

plt.suptitle('Attended appointments by neighborhood', fontsize=14, fontweight='bold')
plt.show()
## Using the pandas.groupby() method to produce a pivot table:
neighbors_II = dataset.groupby(by=['Neighbourhood','No_show']).WaitingCategories.value_counts().sort_index()
## Manipulating the data:
neighbors_II = neighbors_II.unstack(1).unstack()  #Converting the groupby object into a dataset
neighbors_II.fillna(value=0, inplace=True)  #Replacing NaN values by zero
neighbors_II = df_row_normalize(neighbors_II)  #Normalizing its values by the total of each row
neighbors_II = neighbors_II['Yes']  #Keeping only the values related to the absence
## Converting the normalized float values to percentual int values:
neighbors_II = (neighbors_II*100).astype('int64')
neighbors_II = neighbors_II.reindex(columns=['Same day: 0', 'Short: 1-3', 'Week: 4-7', 'Fortnight: 8-15', 'Month: 16-30', 'Quarter: 31-90', 'Semester: 91-180'])
# Drawing a heatmap with the numeric values in each cell
fig3, ax = plt.subplots(figsize=(10, 25))
fig3.subplots_adjust(top=.965)
plt.suptitle('Relative absence distributed by neighborhood and waiting categories', fontsize=14, fontweight='bold')

cbar_kws = {'orientation':"horizontal", 'pad':0.08, 'aspect':50}
sns.heatmap(neighbors_II, annot=True, fmt='d', linewidths=.3, ax=ax, cmap='RdPu', cbar_kws=cbar_kws);
## Defining a new dataframe from the attributes of interest:
patients = dataset[['Gender','Age','Scholarship','Hipertension','Diabetes',
                    'Alcoholism','Handicap','WaitingCategories','SMS_received','No_show']]
## Obtaining an statistical overview of all the attributes:
patients.groupby(by=['No_show','WaitingCategories']).describe()
## Grouping by classes and waiting categories and calculating the instances sum:
patients_sum = patients.groupby(by=['No_show','WaitingCategories']).sum()
## Grouping by classes and waiting categories and calculating the instances sum:
patients_mean = patients.groupby(by=['No_show','WaitingCategories']).mean()
## Adjusting the 'Age' attribute to have the mean instead of sum values:
patients = patients_sum.copy()
patients['Age'] = patients_mean['Age']
## Normalizing data using the predefined function
patients = df_column_normalize(patients, percent=True)
# Drawing a heatmap with the numeric values in each cell
fig4, ax = plt.subplots(figsize=(12, 10))
fig4.subplots_adjust(top=.94)
plt.suptitle('Distribution of patients attributes by waiting categories and no-showing classes', fontsize=14, fontweight='bold')

ax.set_yticklabels(ax.get_yticklabels(), ha="right", fontsize=12, weight='bold')
ax.set_xticklabels(ax.get_xticklabels(), fontsize=12, weight='bold')

cbar_kws = {'orientation':"horizontal", 'pad':0.05, 'aspect':50}
sns.heatmap(patients, annot=True, fmt='.2f', linewidths=.3, ax=ax, cmap='RdPu', cbar_kws=cbar_kws);
fig1 ##This chart was generated in Section 3.1.1
fig4 ##This chart was generated in Section 3.2
fig2 ##This chart was generated in Section 3.1.3
fig3 ##This chart was generated in Section 3.1.3