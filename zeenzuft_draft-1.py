import pandas as pd
df = pd.DataFrame()

test = [1,2,3,4,5]

df['number'] = test

df.to_csv("coucou.csv")