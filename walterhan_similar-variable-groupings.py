import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')

df = pd.concat( (df_train, df_test), axis=0, ignore_index=True)

# One of the column names seems like it should have an underscore based off of other names
df.rename(columns={'ind_var10cte_ult1': 'ind_var10_cte_ult1'}, inplace=True)

# Set ID to index
df.set_index('ID', inplace=True)
df.drop('TARGET', axis=1, inplace=True)
"""
Create a dataframe of column names:
  * original - original column name
  * var - the parsed var name (var1, var3, etc)
  * base - the original text of the var name without the varX (i.e. num_var1 -> num_ )
"""
cols = pd.DataFrame(list(df.columns))
cols.columns = ['original']

split = lambda x: [c for c in x.split('_') if c.startswith('var')][0]

cols['var'] = cols['original'].apply(split)
cols['base'] = cols['original'].apply(lambda x: '_'.join([a for a in x.split('_') if 'var' not in a]))
cols.head()
# Create a dataframe where the columns are different base texts, and rows are different vars
columns = cols[['var', 'base']].groupby('var')['base'].value_counts().unstack().fillna(0)
columns.head()
# Probably a better way to do this
# Transpose and Calculate correlation matrix,
#   and find combinations where correlation = 1 and col name/row index don't match
same_types = columns.T.corr()

results = {}
for i in same_types.columns:
    results[i] = [i]
    for j in same_types.columns:
        if i != j and same_types[i].loc[j] == 1:
            results[i].append(j)

# Dedupe results and produce groupings
var_groups = []
for i in results.values():
    if set(i) not in var_groups:
        var_groups.append(set(i) )
var_groups
# Double check, let's look at a sample group
# var22 and var45
print([x for x in df.columns if 'var22' in x])
print([x for x in df.columns if 'var45' in x])
