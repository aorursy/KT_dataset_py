# import the pandas module
import pandas as pd

# copy data from a file called heathrow-2015.csv and store it in pandas as a dataset called heathrow_2015_data
heathrow_2015_data = pd.read_csv("../input/ldsedexcel/heathrow-2015.csv")

# show the first records from the data
heathrow_2015_data.head()
heathrow_2015_data.head(6)
heathrow_2015_data.shape
heathrow_2015_data.dtypes
heathrow_2015_data['Daily Mean Temperature'].describe()
heathrow_2015_data['Daily Mean Temperature'].describe()
# this imports the plotting library - you only have to do this once in the notebook
# this box has no output
import matplotlib.pyplot as plt
# generate the box plot for the Daily Mean Temperature column
heathrow_2015_data.boxplot(column = ['Daily Mean Temperature'])
plt.show()
print(heathrow_2015_data['Mean Cardinal Direction'].describe())
print(heathrow_2015_data['Mean Cardinal Direction'].value_counts())
heathrow_2015_data = pd.read_csv("../input/ldsedexcel/heathrow-2015.csv")
heathrow_2015_data['Daily Mean Temperature'].describe()
heathrow_2015_data['Daily Total Rainfall'].describe()
# replace any instances of 'tr' with 0.025
heathrow_2015_data['Daily Total Rainfall'] = heathrow_2015_data['Daily Total Rainfall'].replace({'tr': 0.025})
# change the data type to float
heathrow_2015_data['Daily Total Rainfall'] = heathrow_2015_data['Daily Total Rainfall'].astype('float')

# get a summary for the field
heathrow_2015_data['Daily Total Rainfall'].describe()
# The dataset needs to be imported again to overwrite it with the original with "tr" values in the rainfall column
heathrow_2015_data = pd.read_csv("../input/ldsedexcel/heathrow-2015.csv")

# You can edit the following three lines to change the value "tr" is replaced by
heathrow_2015_data['Daily Total Rainfall'] = heathrow_2015_data['Daily Total Rainfall'].replace({'tr': 0.025})
heathrow_2015_data['Daily Total Rainfall'] = heathrow_2015_data['Daily Total Rainfall'].astype('float')
heathrow_2015_data['Daily Total Rainfall'].describe()
# import the data
heathrow_2015_data = pd.read_csv("../input/ldsedexcel/heathrow-2015.csv")

# find the statistics for a numerical column 
print(heathrow_2015_data['Daily Mean Temperature'].describe())

# draw a boxplot 
heathrow_2015_data.boxplot(column = ['Daily Mean Temperature'])
plt.show()

# count the values in a non-numerical column
print(heathrow_2015_data['Mean Cardinal Direction'].value_counts())