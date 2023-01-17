import pandas as pd



def curr_to_float(val):  

    if val[0] == '€':

        val = val[1:]

    if val[-1] == 'M':

        res = float(val[:-1])*1000000

    elif val[-1] == 'K':    

        res = float(val[:-1])*1000

    else:

        res = val   

    return res
#generate list of columns that I want to load, @todo use skipcols parameter

use_cols = list(range(1,89))



#full data set

csv_set_full = pd.read_csv('../input/data.csv',usecols=use_cols)



#create data frame

df = pd.DataFrame(csv_set_full)



#replacing nulls with 0

df['Release Clause'] = df['Release Clause'].fillna('0')



#covert currency to float, @todo use convertion

for i, j in df.iterrows():

    df.at[i,'Value'] = curr_to_float(df.at[i,'Value'])

    df.at[i,'Release Clause'] = curr_to_float(df.at[i,'Release Clause'])



#convert whole column to float    

#df['Overall'] = df['Overall'].astype(float)   
df[['Overall','Value','Release Clause']]
df_corr1 = df[['Overall','Value']]

#convert all to numeric

df_corr1 = df_corr1.apply(lambda x: pd.to_numeric(x, errors='ignore'))

df_corr1.corr()
df_corr1[['Overall','Value']].plot.scatter(x='Overall',y='Value')
df_corr1a = df_corr1[df_corr1['Overall'] > 80]

df_corr1a.corr()
df_corr1a[['Overall','Value']].plot.scatter(x='Overall',y='Value')
df_corr1b = df_corr1[(70 < df_corr1['Overall']) & (df_corr1['Overall'] < 80)]

df_corr1b.corr()
df_corr1b[['Value','Overall']].plot.scatter(x='Overall',y='Value')
df_corr1c = df_corr1[df_corr1['Overall'] < 70]

df_corr1c.corr()
df_corr1c[['Value','Overall']].plot.scatter(x='Overall',y='Value')
df_corr2 = df[['Overall','Release Clause']]

df_corr2 = df_corr2.apply(lambda x: pd.to_numeric(x, errors='ignore'))

df_corr2.corr()
df_corr3 = df[['Value','Release Clause']]

df_corr3 = df_corr3.apply(lambda x: pd.to_numeric(x, errors='ignore'))

df_corr3.corr()
df_corr3.plot.scatter(x='Release Clause',y='Value')