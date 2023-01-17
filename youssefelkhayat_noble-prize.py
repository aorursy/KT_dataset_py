%matplotlib inline
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
#Inspecting the data
nobel_data = pd.read_csv("../input/archive/archive.csv")
nobel_data.sample(10)
#print number of rows by number of columns
nobel_data.shape
#print the count of unique values in each column
nobel_data.nunique()
#based on the ID column, drop duplicates keeping only the first occurence
nobel_no_dup = nobel_data.drop_duplicates(subset='Laureate ID', keep='first')

#check that the number of records is now the same as the number of unique IDs
nobel_no_dup.shape
#Check the missing values in the dataset
# get the number of missing data points per column
missing_values = nobel_no_dup.isnull().sum()

# We want to check the percentage of the missing values.
#Therefore we get the number of missing values and number of total cells to calculate %
total_cells = np.product(nobel_no_dup.shape) 
total_missing_cells = missing_values.sum()

# percent of data that is missing
percentage_missing_values = (total_missing_cells/total_cells) * 100
percentage_missing_values = '%.2f' % percentage_missing_values
print("Missing Data Percentage: " + str(percentage_missing_values) + "%")
#check the different values in the column along with their counts
nobel_no_dup['Laureate Type'].value_counts()
#get only records of type organization
org = nobel_no_dup.loc[nobel_no_dup['Laureate Type'] == "Organization"]
org
#identify records who are labeled as organization but have a birth date which is not nan
isOrg = nobel_no_dup['Laureate Type'] == "Organization"
isBD = pd.notna(nobel_no_dup['Birth Date'])
mask = isOrg & isBD

#change the laureate type of these records to be individual
#run twice to suppress the warning
#nobel_no_dup.loc[ mask, 'Laureate Type'] = "Individual"

df = nobel_no_dup.loc[mask].copy()
df['Laureate Type'] = "Individual"
nobel1 = nobel_no_dup.drop(nobel_no_dup.loc[ mask].index, axis=0)
nobel1 = pd.concat([nobel1, df], axis=0)
nobel1.shape
#get only records of type organization
org = nobel1.loc[nobel1['Laureate Type'] == "Organization"]
org
isOrg = nobel1['Laureate Type'] == "Organization"
isBD = pd.notna(nobel1['Birth Date'])
mask = isOrg & isBD
nobel1.loc[isOrg & isBD]
#get only records of type individual
ind = nobel1.loc[nobel1['Laureate Type'] == "Individual"]
ind.sample(10)
#get missing values count in individual records
ind_missing_values = ind.isnull().sum()
ind_missing_values
#get records who are individuals and their birth city is NaN
isInd = nobel1['Laureate Type'] == "Individual"
isCityNan = pd.isna(nobel1['Birth City'])

nobel1.loc[isInd & isCityNan]
nobel1.loc[nobel1['Full Name'] == "Sir Vidiadhar Surajprasad Naipaul", 'Birth City'] = "Chaguanas"
nobel1.loc[nobel1['Full Name'] == "Liu Xiaobo", 'Birth City'] = "Changchun"

#get records who are individuals and their birth date is NaN
isInd = nobel1['Laureate Type'] == "Individual"
isBDNan = pd.isna(nobel1['Birth Date'])

nobel1.loc[isInd & isBDNan]
nobel1.loc[nobel1['Full Name'] == "Venkatraman Ramakrishnan", 'Birth Date'] = '1952-1-4'
nobel1.loc[nobel1['Full Name'] == "Saul Perlmutter", 'Birth Date'] = '1959-22-9'

ind = nobel1.loc[nobel1['Laureate Type'] == "Individual"]
#get missing values count in individual records
ind_missing_values = ind.isnull().sum()
ind_missing_values
#check the data type of each column
nobel1.dtypes
#nobel1['Birth Date'] = pd.to_datetime(nobel1['Birth Date'])

#when running the commented line above we get this error:
#ValueError: month must be in 1..12

#get organization records alone
org = nobel1.loc[nobel1['Laureate Type'] == "Organization"]
#get individual records alone
ind = nobel1.loc[nobel1['Laureate Type'] == "Individual"]

#split the date using "-" and store the result in 3 columns in a new dataframe df
df = pd.DataFrame()
df[ ['1','2','3'] ] = ind['Birth Date'].str.split("-", expand=True)
#the year is stored in the first column
df['Birth Year'] = df['1']
#drop 1,2,3 and keep year only
df = df.drop(['1', '2', '3'], axis=1)

#Now we make sure that the year is assigned correctly
#we do this by trying to print all values of Birth Year whose length is not =4, this should be false
df.loc[df['Birth Year'].str.len() != 4]
#concat the year to the ind dataframe 

ind = pd.concat([ ind.iloc[: , 0:11], df, ind.iloc[: , 11:] ], axis=1)
ind['Birth Year'] = ind['Birth Year'].astype(int)

#calculate the age as the difference between year of prize and birth year
ind['Age'] = ind['Year'] - ind['Birth Year']

#we won't add zero values for the age of organization 
#and prefer to keep it as Nan and keep the birth year as float instead

#form the nobel_no_dup as the concatentation of the modified ind and the ord
nobel2 = pd.concat([ind,org], axis=0, sort=False)

#reorder the columns, make age with birth year
cols = nobel2.columns.tolist()
cols_ord = cols[0:12] + cols[19:20] + cols[12:19]
nobel3 = nobel2[cols_ord] 
nobel3
nobel3["Laureate Type"] = nobel3["Laureate Type"].astype('category')
nobel3["Laureate Type"] = nobel3["Laureate Type"].cat.codes
nobel3
#create the dictionary
dict = {'Male' : 0, 'Female' : 1} 

#convert it into categorical
nobel3["Sex"] = nobel3["Sex"].astype('category')  

#Remap the values of the dataframe 
nobel3["Sex"] = nobel3["Sex"].map(dict)

nobel3.sample(10)
nobel3['Category'].value_counts()
#first we label them using numerical encoding

from sklearn.preprocessing import LabelEncoder

labelEncoder = LabelEncoder()

nobel3['Category'] = labelEncoder.fit_transform(nobel3['Category'])

nobel3.sample(10)

#We notice that the encoding is as follows:
# 0=Chemistry, 1=Economics, 2=Literature, 3=Medicine, 4=Peace, 5=Physics

#Now we proceed to transforming the encoding into one hot

from sklearn.preprocessing import OneHotEncoder

oneHotEncoder = OneHotEncoder()
oh = oneHotEncoder.fit_transform(nobel3['Category'].values.reshape(-1,1)).toarray()

dfOneHot = pd.DataFrame(oh, columns = ["Chemistry", "Economics", "Literature", "Medicine", "Peace", "Physics"])
df = pd.get_dummies(dfOneHot)

nobel3.reset_index(drop=True, inplace=True)
df.reset_index(drop=True, inplace=True)

nobel4 = pd.concat([nobel3, df], axis=1)

nobel4.sample(5)

nobel4['Prize Share'].isnull().sum()
#split the date using "/" and store the result in 2 columns in a new dataframe df
df = pd.DataFrame()
df[ ['1','Prize Share'] ] =  nobel4['Prize Share'].str.split("/", expand=True)

#display the part before "/"
df['1'] = df['1'].astype(int)
df['1'].sum()

#drop 1 and keep Prize Share only
df = df.drop(['1'], axis=1)

#drop the old prize share column from the dataframe
nobel4 = nobel4.drop(['Prize Share'], axis=1)

#concat the new prize share column to the dataframe
nobel5 = pd.concat([nobel4, df], axis=1)

#convert it into int
nobel5['Prize Share'] = nobel5['Prize Share'].astype(int)

#now 1 means 1/1, 2 means 1/2, 3 means 1/3 and 4 means 1/4
nobel5.head(5)
#Now we proceed to transforming the encoding into one hot

from sklearn.preprocessing import OneHotEncoder

oneHotEncoder = OneHotEncoder()
oh = oneHotEncoder.fit_transform(nobel5['Prize Share'].values.reshape(-1,1)).toarray()

dfOneHot = pd.DataFrame(oh, columns = ["Share 1/1", "Share 1/2", "Share 1/3", "Share 1/4"])
df = pd.get_dummies(dfOneHot)

nobel5.reset_index(drop=True, inplace=True)
df.reset_index(drop=True, inplace=True)

nobel6 = pd.concat([nobel5, df], axis=1)
nobel6.sample(10)
#drop the columns we don't need
nobel_cleaned = nobel6.drop(['Prize Share', 'Category', 'Prize', 'Birth Date', 'Birth City' ,'Death Date', 'Death Country', 'Death City','Organization City', 'Organization Country'], axis=1)
nobel_cleaned.sample(10)
nobel_cleaned.to_csv('nobel_data_cleaned.csv', index= None) 
ind = nobel_cleaned.loc[nobel_cleaned['Laureate Type'] == 0]

phys = ind.loc[nobel_cleaned['Physics'] == 1]
chem = ind.loc[nobel_cleaned['Chemistry'] == 1]
econ = ind.loc[nobel_cleaned['Economics'] == 1]
liter = ind.loc[nobel_cleaned['Literature'] == 1]
med = ind.loc[nobel_cleaned['Medicine'] == 1]
peace = ind.loc[nobel_cleaned['Peace'] == 1]


#we get the number of won prizes in each field 
numOfRows = phys.shape[0]
print('Number of Rows in pyhsics : ' , numOfRows)
numOfRows = chem.shape[0]
print('Number of Rows in chem : ' , numOfRows)
numOfRows = econ.shape[0]
print('Number of Rows in econ : ' , numOfRows)
numOfRows = liter.shape[0]
print('Number of Rows in liter : ' , numOfRows)
numOfRows = med.shape[0]
print('Number of Rows in med : ' , numOfRows)
numOfRows = peace.shape[0]
print('Number of Rows in peace : ' , numOfRows)

phys_avg_age = (phys['Age'].sum() / phys.shape[0]).astype(int)
chem_avg_age = (chem['Age'].sum() / chem.shape[0]).astype(int)
econ_avg_age = (econ['Age'].sum() / econ.shape[0]).astype(int)
liter_avg_age = (liter['Age'].sum() / liter.shape[0]).astype(int)
med_avg_age = (med['Age'].sum() / med.shape[0]).astype(int)
peace_avg_age = (peace['Age'].sum() / peace.shape[0]).astype(int)
avg_age = [('Physics',phys_avg_age), ('Chemistry', chem_avg_age), ('Economics', econ_avg_age), 
           ('Literature', liter_avg_age), ('Medicine', med_avg_age), ('Peace', peace_avg_age)] 

avg_age_dist = pd.DataFrame(avg_age,columns=['Field', 'Average Age'])

avg_age_dist = avg_age_dist.set_index('Field')
avg_age_dist.plot.bar(rot=45)
# Create data
df = pd.DataFrame(columns=['Age', 'Freq'])
df = ind['Age'].value_counts()

x = np.arange(1,101).astype(int)
y = np.zeros(shape=(100)).astype(int)
for item in df.iteritems(): 
    index = int(item[0])
    y[index] = item[1]

#Plot
plt.scatter(x, y)
plt.title('Number of winners (y-axis) vs age (x-axis)')
plt.xlabel('Age')
plt.ylabel('# of Winner')
plt.show()
#first we get the individuals only.
ind = nobel_cleaned.loc[nobel_cleaned['Laureate Type'] == 0]

#then we get females only.
females = ind[ind['Sex'] == 1]

#then we get lowest year
min_year = females['Year'].min()
first_female = females.loc[females['Year'] == min_year]
first_female
#first we get the individuals.
ind = nobel_cleaned.loc[nobel_cleaned['Laureate Type'] == 0]

#we then get the count of the males and the females
male= ind[ind['Sex'] == 0]
male_count = male['Sex'].count()
female = ind[ind['Sex'] == 1]
female_count = female['Sex'].count()

#plot the bar chart
df = pd.DataFrame({'winners':['males', 'females'], 'count':[male_count, female_count]})
df.plot.bar(x='winners', y='count', rot=0)
#first we get the individuals only.
ind = nobel_cleaned.loc[nobel_cleaned['Laureate Type'] == 0]

min_age = ind['Age'].min()
youngest_winner = ind.loc[ind['Age'] == min_age]

youngest_winner
#first we get the individuals only.
ind = nobel_cleaned.loc[nobel_cleaned['Laureate Type'] == 0]

max_age = ind['Age'].max()
oldest_winner = ind.loc[ind['Age'] == max_age]

oldest_winner
#first we get the individuals only.
ind = nobel_cleaned.loc[nobel_cleaned['Laureate Type'] == 0]

#get individuals with no organsiation.
ind_solo = ind[ind['Organization Name'].isnull()]
#get individuals with organsiation.
ind_org = ind[ind['Organization Name'].notnull()]

ind_solo_count = ind_solo.shape[0]
ind_org_count = ind_org.shape[0]

counts = [('No Organization',ind_solo_count), ('Belonging to Organisation', ind_org_count)] 

counts_df = pd.DataFrame(counts,columns=['Individual Label', 'Count'])
counts_df = counts_df.set_index('Individual Label')
counts_df.plot.bar(rot=0)

#In the first line, we get the physics award winners,
#then we get the counts depdening on the birth country
#finally the graph is plotted.
#This process is repeated for each field

phys = nobel_cleaned.loc[nobel_cleaned['Physics'] == 1]
phys_country = phys['Birth Country'].value_counts()
phys_country = phys_country[phys_country > 1]
# phys_country.plot.barh(rot=0, figsize=(10,10), colors=(31/255,119/255,180/255))
chem = nobel_cleaned.loc[nobel_cleaned['Chemistry'] == 1]
chem_country = chem['Birth Country'].value_counts()
chem_country = chem_country[chem_country > 1]
# chem_country.plot.barh(rot=0, figsize=(10,10), colors=(31/255,119/255,180/255))
econ = nobel_cleaned.loc[nobel_cleaned['Economics'] == 1]
econ_country = econ['Birth Country'].value_counts()
econ_country = econ_country[econ_country > 1]
# econ_country.plot.barh(rot=0, figsize=(5,5), colors=(31/255,119/255,180/255))
liter = nobel_cleaned.loc[nobel_cleaned['Literature'] == 1]
liter_country = liter['Birth Country'].value_counts()
liter_country = liter_country[liter_country > 1]
liter_country.plot.barh(rot=0, figsize=(10,10), colors=(31/255,119/255,180/255))
med = nobel_cleaned.loc[nobel_cleaned['Medicine'] == 1]
med_country = med['Birth Country'].value_counts()
med_country = med_country[med_country > 1]
med_country.plot.barh(rot=0, figsize=(10,10))
peace = nobel_cleaned.loc[nobel_cleaned['Peace'] == 1]
peace_country = peace['Birth Country'].value_counts()
peace_country = peace_country[peace_country > 1]
peace_country.plot.barh(rot=0, figsize=(10,10))
#first we get the individuals and organizations count.
ind = nobel_cleaned.loc[nobel_cleaned['Laureate Type'] == 0].count()
org = nobel_cleaned.loc[nobel_cleaned['Laureate Type'] == 1].count()

ind_count = ind[ind['Laureate Type'] == 0]
org_count = org[org['Laureate Type'] == 1]

#plot the bar chart
df = pd.DataFrame({'winners':['individual', 'organisation'], 'count':[ind_count, org_count]})
df.plot.bar(x='winners', y='count', rot=0)
from collections import Counter

#We seperate the Motivation column in a seperate Dataframe
Motivation_column = nobel_cleaned["Motivation"]

#We drop all the rows which holds null values
Motivation_column  = Motivation_column.dropna(how='any',axis=0) 

# Then w apply Method counter that results in an array, at which each instance of that array includes
# a word and counter for each repetition for it.
common_words = Counter('"'.join(Motivation_column.str.lower()).split()).most_common(100)
common_words
#in order to include stopwords in our code we had to use nltk.download() which will open a window for us,
#from which we can use stopwords to download 

# nltk.download()
import nltk
from nltk.corpus import stopwords

#By trial and error, we found that the process of eliminating the stop words from our data
#needs to be done 3 times at least to get rid of all stop words
for i in range(0,3):
    for word in common_words:
        if word[0] in (stopwords.words('english')):
            common_words.remove(word)
common_words
#We created a dataframe that holds the word and its frequency in 2 separate columns

Most_frequent = pd.DataFrame(common_words,columns=['Word', 'Frequency'])
Most_frequent = Most_frequent.head(10)
Most_frequent
#We removed the index column and set the Word column to be the index, in order to make the visualization more clear.

Most_frequent = Most_frequent[Most_frequent.Word != '"'].reset_index()
del Most_frequent['index']
Most_frequent = Most_frequent.set_index('Word')
print(Most_frequent)
Most_frequent.plot.bar(rot=0, figsize=(17,8), width=0.8)
