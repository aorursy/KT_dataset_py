import pandas as pd

df = pd.read_csv('../input/world_cup_goals.csv')

print(df.shape)
print(df.head(5))
print(df['Country'].unique())
import matplotlib.pyplot as plt
%matplotlib inline
df['WC'] = df['WC'].apply(lambda x: x[-4:])

x_small = df[df['Goals'] < 16]['WC']
y_small = df[df['Goals'] < 16]['Country']

x_big = df[df['Goals'] > 15]['WC']
y_big = df[df['Goals'] > 15]['Country']

fig, axes = plt.subplots(figsize=(10,30))


axes.scatter(x_small, y_small, color="red", marker='x')
axes.scatter(x_big, y_big, color="green")

plt.xticks(rotation=70)
plt.gca().invert_yaxis()
axes.legend(['15 or less', 'More than 15'])
df[df['Goals'] > 15].shape[0]
df['Goals'].count()
df[df['Goals'] > 15].shape[0] / df['Goals'].count()