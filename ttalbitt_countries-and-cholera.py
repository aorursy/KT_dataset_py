import pandas

file = open('../input/cholera-dataset/data.csv', 'r')
df = pandas.read_csv(file)

df['Number of reported cases of cholera'].iloc[1059] = '35'
df['Number of reported cases of cholera'] = df['Number of reported cases of cholera'].fillna('-1')
df['Number of reported cases of cholera'] = df['Number of reported cases of cholera'].apply(int)

truth = df['Number of reported cases of cholera'] != 0
count = -1
indices = []
for t in truth:
    count += 1
    if t is True:
        indices.append(count)
df = df.drop(indices)

print('none in the past 10 years:')
print(df['Year'].value_counts())
print(df['Country'].value_counts())




























































































