# Loading the package
import pandas as pd

import numpy as np;
a = [1,2,3,4];

#List to DataFrame
a_dataframe = pd.DataFrame(a);
print(a_dataframe);
print(a_dataframe.index);
a_dataframe.index = ['I','II','III','IV'];
print(a_dataframe.index);
print(a_dataframe);
# DataFrame from Dictionaries

b = {'A':[1,2,3,4,5], 'B':[6,7,8,9,10]};
print (b);
b_d = pd.DataFrame(b, index = ['I','II','III','IV','V']);
print(b_d);


import numpy as np;
# Concatenating 2 dataframes
b1 = {'A':[1,2,3,4,5], 'B':[6,7,8,9,10]};
d1 = pd.DataFrame(b1);
b2 = {'A':[11,22,33,44,55], 'B':[66,77,88,99,110], 'C':[0,8,6,np.nan,np.nan]};
d2 = pd.DataFrame(b2);
print (d1);
print (d2);
d = pd.concat([d1,d2]);
print(d);
# CHecking NAN values
# In Series
s= pd.Series([1,3,np.nan,7,9,np.nan,8,9]);
print(s);
print('----');
print(s.isna().sum());
#In Dataframe
print(d);
print(d.isna().sum());

#Checking in a particular column

print(d['C'].isna().sum());
# Python merge/join dataframes

p1 = pd.DataFrame({
  'A':[1,2,3],
  'B':[4,5,6],
  'K':['a','b','c']

});
 
p2 = pd.DataFrame({
  'A1':[11,22,33],
  'B1':[45,55,66],
  'K':['a','b','d']    
    
});
print('-----------------------------------');
p=pd.merge(p1,p2,how="inner",on="K");
print(p);
print('--------------------------------------');
p=pd.merge(p1,p2,how="left",on="K");
print(p);
print('----------------------------------------');
p=pd.merge(p1,p2,how="right",on="K");
print(p);
print('----------------------------------------');
p=pd.merge(p1,p2,how="outer",left_on="K",right_on="K");
print(p);
# Dropping columns

print(d);

d.drop(["C"],axis=1,inplace=True);
print(d);
#Dropping rows by index

d.drop([0],inplace=True,axis=0);
print(d);
b = {'A':list(range(1,16)), 'B':list(range(10,160,10)), 'C':list(map(chr,range(97,112)))};

d = pd.DataFrame(b);

print(d);
#EDA
#No of rows and columns
print(d.shape);
print('---------------------------------');
# No of rows
print(d.shape[0]);
print('---------------------------------');
# No of columns
print(d.shape[1]);
print('---------------------------------');
# No of NA values
print(d.isna().sum());
print('---------------------------------');
# Summary
print(d.describe());
print('---------------------------------');
print(d.isnull().values.sum());
print(d.describe(include="all"));
print(d.C.describe());
print(d.head(3));

print('------------------');
print(d.tail(3));

print('------------------');
# 2nd and 3rd row, 2nd column.
print(d.iloc[[1,2],1]);
# By Column name
print(d.loc[[1,2],"B"]);
d = {
    'Name':['Alisa','Bobby','jodha','jack','raghu','Cathrine',
            'Alisa','Bobby','kumar','Alisa','Alex','Cathrine'],
    'Age':[26,24,23,22,23,24,26,24,22,23,24,24]
}
 
df = pd.DataFrame(d,columns=['Name','Age'])
print(df);
print(df.shape[0]);
#unique rows
print(df.drop_duplicates().shape[0]);
print('--------------');
#Unique Names
print(df.Name.unique());
print(df);
print(df.groupby('Name').agg({'Age':['count', 'min']}))
print(df.info());
# Total no of entries
print(len(df.Name));
#Unique Entries
print(len(set(df.Name)));
print(df.Age.count());
print(df.Age.median());
print(df.Age.mean());
e=[1,np.nan,3,np.nan,5,6,np.nan];
e1=[11,22,33,np.nan,6,7,np.nan];
e2 = [1,1,1,1,1,1,np.nan]
df=pd.DataFrame({'A':e,'B':e1,'C':e2});
print(df);
print(df.isna().sum());
print('---------------------');
print(df.isnull().values.sum());
df.A.fillna(df.A.mean(),inplace=True);
print(df);
df.fillna(0,inplace = True);
print(df);
# Pandas Series

s=pd.Series([1,2,3,np.nan,5]);
print(s);
a=np.array([1,2,3]);

s= pd.Series(a,index=['I','II','III']);
print(s);
a={'A':1,'B':2,'C':3};

s= pd.Series(a);
print(s);
a={'A':1,'B':2,'C':3};

s= pd.Series(a,index=['C','B','A','D']);
print(s);
s= pd.Series(4,index=['I','II','III']);
print(s);



a={'A':1,'B':2,'C':3,'D':4,'E':5,'F':6};

s= pd.Series(a,index=['C','B','A','F','E','D']);
print(s);
print('____________________________');
print(s[0]);
print(s[1:4]);
print(s[-3:]);
print(s['A']);
print(s[['A','B']]);
a=[[1,2,3],[4,5,6,7]];
print(a);
a1=pd.DataFrame(a,index=[1,2]);
print(a1);
a2=pd.DataFrame([[1,2,3,4]]);
print(a2);
a3=pd.DataFrame([np.arange(1,40,step=10)]);
print(a3);
a1 = a1.append(a2);
a1 = a1.append(a3);


a1.columns=['A','B','C','D'];
print(a1);
print(a1.T);
a1['E']= a1['C']+a1['D'];
print(a1);
#Dropiing Rows having NaN
a1.dropna(inplace=True);
print(a1);
a1.iloc[[0],[1]]=np.nan;
print(a1);
#Dropping columns with atleast 1 NaN
a1.dropna(inplace=True, axis=1);
print(a1);
a1=pd.DataFrame({'A':[1,np.nan,2,3,4,np.nan],'B':[1,2,3,4,5,6],'C':[np.nan,22,33,44,55,np.nan],'D':[1,np.nan,2,np.nan,np.nan,np.nan]});
print(a1);
#Drop rows if NaN is present in either 'A' or 'D'

a1.dropna(inplace=True,subset=['A','D']);
print(a1);
#Deleting a column
del a1['B'];
print(a1);
#Deleting a column
a1.pop('C');
print(a1);
#Deleting row with index =0
a1.drop(0,inplace=True);
print(a1);
a1=pd.DataFrame({'A':[np.nan,np.nan],'B':[np.nan,np.nan],'C':[np.nan,22],'D':[np.nan,3]});
print(a1);
#deleting rows where the whole row has NaN
print(a1.dropna(how="all"));
#deleting columns where the whole column has NaN
print(a1.dropna(how="all",axis="columns"));
