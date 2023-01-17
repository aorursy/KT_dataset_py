import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
df = pd.read_json('../input/searches.json', orient='records',lines=True)
df.head()
stats.normaltest(df['login_count']).pvalue
sns.distplot(df['login_count'], fit=stats.norm)
#log transfromation
sns.distplot(np.log(df['login_count']), fit=stats.norm)
#Users with an odd-numbered uid were shown a new-and-improved search box. 
odd = df[df['uid'] % 2 != 0]
even = df[df['uid'] % 2 ==0]
odd_one = odd[odd['search_count']>0].count()
odd_zero = odd[odd['search_count']==0].count()
even_one = even[even['search_count']>0].count()
even_zero = even[even['search_count']==0].count()
contingency = [[odd_one['uid'], odd_zero['uid']], [even_one['uid'], even_zero['uid']]]
Q1 = stats.chi2_contingency(contingency)
Q1[1]
#nonparametric tests
Q2 = stats.mannwhitneyu(even['search_count'], odd['search_count']).pvalue
Q2
new_odd = odd[odd['is_instructor'] == True]
new_even = even[even['is_instructor'] == True]
odd_one = new_odd[new_odd['search_count']>0].count()
odd_zero = new_odd[new_odd['search_count']==0].count()
even_one = new_even[new_even['search_count']>0].count()
even_zero = new_even[new_even['search_count']==0].count()
contingency = [[odd_one['uid'], odd_zero['uid']], [even_one['uid'], even_zero['uid']]]
Q3 = stats.chi2_contingency(contingency)
Q3[1] 
Q4 = stats.mannwhitneyu(new_even['search_count'], new_odd['search_count']).pvalue
Q4 