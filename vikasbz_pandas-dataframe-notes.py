#Import the Pandas package
import pandas as pd

#1. Manually entering the data
df2 = pd.DataFrame(
    {
        "SLNO" : [1,2,3,4,5],
        "Months" : ['Jan', 'Feb', 'Mar', 'Apr', 'May'],
        "Col3" : [0.11, 0.23, 0.56, 0.12, 0.87],
        "Bool_Col" : [True, True, False, True, False]
    }
)

df2
#2. From CSV files (.csv)
df = pd.read_csv("../input/train.csv")

#Similar fucntion available for spreadsheet as well (XLS)

print(type(df))
#Dimension
df.ndim #2-dimensional
#Shape
df.shape #891 rows x 12 columns
#Total number of cells
df.size # 891*12 = 10692
#Columns
df.columns
#Change column name
df2.columns
#1. 
df2.columns = ['index', 'mon', 'nums', 'bools']

df2.columns
#2. 
df2 = df2.rename(columns={'mon':'Month'})

df2.columns
#inplace parameter will change the dataframe without assignment
df2.rename(columns={"Month" : "months",
                   "nums" : "Numbers"}, 
           inplace=True)

df2.columns
#First few rows
df.head(6)

#Indexing starts with 0
#Last few rows
df.tail(6)
#Datatype of data in each column
df2.dtypes
#Summarizing
df.describe()

#Describe funtion gives statistical summary column-wise of numerical columns including quartile values (25%, 50%, 75%)
df.info()

#'Age' has only 714 values, rest are NAs: null values/missing values
#1. using dot notation
df.Fare.head(4)
#2. using square braces
df['Fare'].head(4)
#3. using iloc
df.iloc[:,9].head(4) #9th column is 'Fare'
#iloc[:,9] is same as iloc[0:890, 9]
df.iloc[0:890,9].head(4)
df[['Fare', 'Age', 'Sex']].head(4)
#Multiple columns selection using iloc
df.iloc[:4, [9, 5, 4]]

#Can select rows as well using iloc
df.iloc[2:8, 9] #From row 2 to row 7; 8 excluded
#Row selection can also be used with square braces
df['Fare'][2:8]
#Or with dot notation
df.Fare[2:8]
#1. summing
df.Fare.sum()
#2. averaging
df.Fare.mean()
#3. counting
df.Fare.count()
#4. median
df.Fare.median()
#5. minimum value
df.Fare.min()
#6. Maximum value
df.Fare.max()
#7. unique values
df.Sex.unique()
#8. Number of unique values
df.Sex.nunique()
#9. Missing values imputation using fillna function
df.Age.count()
#Total 891 rows, only 714 present. Rest are NAs (missing values)
df.Age = df.Age.fillna(20)

df.Age.count() #All missing values are filled with '20', new count is 891.
#1. 
df.iloc[2:6, :] #Same as df.iloc[2:6,]
#2.
df.iloc[[2,3,4,10,20], :]
#3. Logical selection
df[ df['Sex'] == 'male' ].head(4)
# Deleting columns
    
# Delete a column from the dataframe
df = df.drop("Pclass", axis=1)
    
# alternatively, delete columns using the columns parameter of drop
df = df.drop(columns="SibSp")
df.columns
# Delete the Area column from the dataframe in place
# Note that the original 'data' object is changed when inplace=True
df.drop("Parch", axis=1, inplace=True)

# Delete multiple columns from the dataframe
df = df.drop(["Ticket", "Cabin", "Name"], axis=1)
df.columns
df.head()
# Delete the rows with labels 0,1,5
df = df.drop([0,1,2], axis=0)
df.head(5)
# Delete the rows with label "male"
# For label-based deletion, set the index first on the dataframe:
df = df.set_index("Sex")
df = df.drop("male", axis=0) # Delete all rows with label "male"

df.head(8)
# Delete the first five rows using iloc selector
df = df.iloc[5:,]

df.head(8)
df.head()
df.to_csv("new_train.csv", index=False)