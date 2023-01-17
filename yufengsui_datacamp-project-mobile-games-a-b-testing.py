import pandas as pd



df = pd.read_csv('../input/cookie_cats.csv')

df.head()
df.info()
df.groupby('version').count()
df['sum_gamerounds'].plot.box(figsize=(5,10))
df['sum_gamerounds'].describe()
# Counting the number of players for each number of gamerounds 

plot_df = df.groupby('sum_gamerounds')['userid'].count()

plot_df
# Plot the distribution of players that played 0 to 100 game rounds during their first week playing the game.

%matplotlib inline



ax = plot_df[:100].plot(figsize=(10,6))

ax.set_title("The number of players that played 0-100 game rounds during the first week")

ax.set_ylabel("Number of Players")

ax.set_xlabel('# Game rounds')
df['retention_1'].sum() / df['retention_1'].count() # When using .sum(), T/F will first be converted to 1/0.



# Equivalent to df['retention_1'].mean()

# Mean is calculated by summing the values and dividing by the total number of values.
df.groupby('version')['retention_1'].mean()
# Creating an list with bootstrapped means for each AB-group

boot_1d = []

for i in range(1000):

    boot_mean = df.sample(frac = 1,replace = True).groupby('version')['retention_1'].mean()

    boot_1d.append(boot_mean)

    

# Transforming the list to a DataFrame

boot_1d = pd.DataFrame(boot_1d)

    

# A Kernel Density Estimate plot of the bootstrap distributions

boot_1d.plot(kind='density')
# Adding a column with the % difference between the two AB-groups

boot_1d['diff'] = (boot_1d.gate_30 - boot_1d.gate_40)/boot_1d.gate_40*100



# Ploting the bootstrap % difference

ax = boot_1d['diff'].plot(kind='density')

ax.set_title('% difference in 1-day retention between the two AB-groups')



# Calculating the probability that 1-day retention is greater when the gate is at level 30

print('Probability that 1-day retention is greater when the gate is at level 30:',(boot_1d['diff'] > 0).mean())
df.groupby('version')['retention_7'].sum() / df.groupby('version')['retention_7'].count()
# Creating a list with bootstrapped means for each AB-group

boot_7d = []

for i in range(500):

    boot_mean = df.sample(frac=1,replace=True).groupby('version')['retention_7'].mean()

    boot_7d.append(boot_mean)

    

# Transforming the list to a DataFrame

boot_7d = pd.DataFrame(boot_7d)



# Adding a column with the % difference between the two AB-groups

boot_7d['diff'] = (boot_7d.gate_30 - boot_7d.gate_40)/boot_7d.gate_40*100



# Ploting the bootstrap % difference

ax = boot_7d['diff'].plot(kind='density')

ax.set_title('% difference in 7-day retention between the two AB-groups')



# Calculating the probability that 7-day retention is greater when the gate is at level 30

print('Probability that 7-day retention is greater when the gate is at level 30:',(boot_7d['diff'] > 0).mean())