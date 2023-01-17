#importing libraries
import pandas as pd
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
# read salaries datasets
unemployment = pd.read_csv('../input/output.csv',low_memory=False)
#check the head of data
unemployment.head()
#check the back of the data
unemployment.tail()
#info about data
unemployment.info()
#now, we are checking start with a pairplot, and check for missing values
sns.heatmap(unemployment.isnull(),cbar=False)
#Which state has the most data
color = sns.color_palette()
cnt_srs = unemployment.State.value_counts()

plt.figure(figsize=(12,8))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[1])
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('States', fontsize=12)
plt.title('Count the states', fontsize=15)
plt.xticks(rotation='vertical')
plt.show()
# take the mean of rate state by state
grouped_df = unemployment.groupby(["State"])["Rate"].aggregate("mean").reset_index()

plt.figure(figsize=(12,8))
sns.pointplot(grouped_df['State'].values, grouped_df['Rate'].values, alpha=0.8, color=color[2])
plt.ylabel('Mean rate', fontsize=12)
plt.xlabel('States', fontsize=12)
plt.title("Average of mean", fontsize=15)
plt.xticks(rotation='vertical')
plt.show()
#https://www.kaggle.com/sudalairajkumar/simple-exploration-notebook-instacart
#see the number of unique states
unemployment.State.nunique()
#See exact numbers
make_total = unemployment.pivot_table("Rate",index=['State'],aggfunc='mean')
topstate=make_total.sort_values(by='Rate',ascending=False)[:47]
print(topstate)
#Calculate  which models has highest yearly fluncations
maketotal_1 = unemployment.pivot_table(values='Rate',index=['Month','State','County'],aggfunc=np.std)
df1 = maketotal_1.reset_index().dropna(subset=['Rate'])
df2 = df1.loc[df1.groupby('State')['Rate'].idxmax()]
for index,row in df2.iterrows():
    print(row['State'],"State which",row['County'],"has the highest yearly fluncation.")
