import pandas as pd

df = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

columns = ['Neighborhood', 'HouseStyle', 'OverallQual', 'OverallCond', 'SalePrice']

df = df[columns]

print(df.shape)

print(df.head())
df_expensive = df[df.SalePrice >= 150000]

print(df_expensive.shape)

print(df_expensive.head())
# loc uses NAMES to index the DataFrame, iloc uses the integer location regardless of the name

# if the names are numeric, numeric types can be used. Conditional arrays are also valid.

# syntax is df.loc[row_names, col_names] and df.iloc[row_nums, col_nums]

print(df_expensive.loc[1,'Neighborhood'])

df_expensive.rename(index={1:'one'}, inplace=True)

try:

    df_expensive.loc[1,'Neighborhood']

except KeyError:

    print('Error was thrown, numbers in loc don\'t always behave!')

    print('Using iloc')

    print(df_expensive.iloc[1,0])

df_expensive.reset_index(drop=True, inplace=True)

df_expensive.rename(columns={'Neighborhood':'Hood'}, inplace=True)

print(df_expensive.head())
# parenthesis allow the carriage returns

df_expensive_ = (

    df[df.SalePrice >= 150000]

    .reset_index(drop=True)

    .rename(columns={'Neighborhood':'Hood'})

)

print(df_expensive_.head())
avg_price = (

    df[['Neighborhood', 'SalePrice']]

    .groupby(['Neighborhood'], as_index=False).mean()

)

print(avg_price.head())
hood_count = (

    df[['Neighborhood', 'HouseStyle', 'SalePrice']]

    .groupby(['Neighborhood', 'HouseStyle'])

    .count()

)

print(hood_count.head(10))
df_sorted = df.sort_values(by=['SalePrice'])

print(df_sorted.head())

# sort descending

df_sorted = df.sort_values(by=['SalePrice'], ascending=False)

print(df_sorted.head())
num_hood = df[['Neighborhood', 'HouseStyle']].groupby(['Neighborhood'], as_index=False).count().rename(columns={'HouseStyle':'NumHood'})

num_style = df[['Neighborhood', 'HouseStyle']].groupby(['HouseStyle'], as_index=False).count().rename(columns={'Neighborhood':'NumStyle'})

df_merged = df.merge(num_hood, on=['Neighborhood'], how='left').merge(num_style, on=['HouseStyle'], how='left')

print(df_merged.head())