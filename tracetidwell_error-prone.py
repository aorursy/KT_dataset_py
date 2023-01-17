# First we need to import our relevant libraries



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# We will be working with the fielding.csv file, so let's read it to a dataframe using Pandas

fielding = pd.read_csv('../input/fielding.csv')
# Check the head of the dataframe

fielding.head()
# Get information regarding the makeup of the observations

fielding.info()
# In order to cleanup the data, we can drop all observations with the position DH, 

# since the DH only bats.

fielding = fielding[fielding['pos']!='DH']
# We can check the size of the updated dataframe.

# We still have 18 columns, but we only have 165,477 rows

fielding.shape
# The first thing we can do is compare the total number of errors for each position

errors = fielding.groupby(by='pos')['e'].sum()
# Let's view the results

sns.barplot(x=errors.sort_values().index, y=errors.sort_values())

plt.title('Number of Errors by Position')

plt.xlabel('Position')

plt.ylabel('Total Errors')
# We can see from the plot that the SS position has the most errors overall. But does

# that mean that SS is more error prone? Short-stop is a very important position, so

# maybe they are involved in more plays. We also see that CF, RF, and LF have far fewer

# errors than the other positions. That's because prior to 1954, they were all recorded

# as OF. If we add all the outfield positions together, let's see what we get.

errors['OF'] = errors['CF'] + errors['RF'] + errors['LF'] + errors['OF']
#Now let's remove the individual outfield positions.

errors.drop(['CF', 'RF', 'LF'], axis=0, inplace=True)
# Let's view the updated results.

sns.barplot(x=errors.sort_values().index, y=errors.sort_values())

plt.title('Number of Errors by Position')

plt.xlabel('Position')

plt.ylabel('Total Errors')
# So now it looks like outfielders may have made more errors than shortstops. But perhaps there

# is more to it than total errors. Certain positions may handle the ball more, so we'd expect more

# errors. To account for this, we'll now compare errors per putout for each position.

# First we'll group the dataframe by position, then select the error and putouts columns,

# then we'll sum them.

errors_po = fielding.groupby(by='pos')[['e', 'po']].sum()
# Let's check the head to make sure we got what we want.

errors_po.head()
# Since we already know about the outfield problem, let's go ahead and sum the results

# of the individual outfield positions and add them to the OF line.

errors_po.loc['OF']['e'] = errors_po.loc['OF']['e'] + errors_po.loc['LF']['e'] + errors_po.loc['CF']['e'] + errors_po.loc['RF']['e']

errors_po.loc['OF']['po'] = errors_po.loc['OF']['po'] + errors_po.loc['LF']['po'] + errors_po.loc['CF']['po'] + errors_po.loc['RF']['po']
#Now let's remove the individual outfield positions.

errors_po.drop(['CF', 'RF', 'LF'], axis=0, inplace=True)
# We have errors and putouts, but we need to create the column error/putout.

# We can define a function that takes in a dataframe and returns the errors column

# divided by the putouts column.

def e_per_po(df):

    return df['e'] / df['po']
# Now we apply our new function to the dataframe.

errors_po['e/po'] = errors_po.apply(e_per_po, axis=1)
# Check the head to see if calculated correctly

errors_po.head()
# We can once again plot our results, but this time with our new e/po column.

sns.barplot(x=errors_po['e/po'].sort_values().index, y=errors_po['e/po'].sort_values())

plt.title('Number of Errors per Putout by Position')

plt.xlabel('Position')

plt.ylabel('Average Errors per Putout')
# Here we see a different picture. Relative to the number of putouts, outfielders

# commit amongst the fewest errors, with third basemen and pitchers leading the way.

# However, we might once again be overlooking something. Pitchers are also responsible

# for strikeouts. Perhaps we should add strikeouts and putouts to determine the total

# number of outs produced. We'll need the pitching table

pitching = pd.read_csv('../input/pitching.csv')
pitching.head()
pitching.info()
so = pitching['so'].sum()
errors_po.loc['P']
errors_to = errors_po.copy()
errors_to['so'] = 0
errors_to
errors_to.reset_index(inplace=True)
errors_to
def set_so(row):

    if row == 'P':

        return so

    else:

        return 0
errors_to['so'] = errors_to['pos'].apply(set_so)
errors_to
def set_to(df):

    return df['so'] + df['po']
errors_to['to'] = errors_to.apply(set_to, axis=1)
errors_to
# We have errors and putouts, but we need to create the column error/putout.

# We can define a function that takes in a dataframe and returns the errors column

# divided by the putouts column.

def e_per_to(df):

    return df['e'] / df['to']
# Now we apply our new function to the dataframe.

errors_to['e/to'] = errors_to.apply(e_per_to, axis=1)
# We can once again plot our results, but this time with our new e/po column.

errors_to.sort_values(by='e/to', inplace=True)

sns.barplot(x=errors_to['pos'], y=errors_to['e/to'])

plt.title('Number of Errors per Out by Position')

plt.xlabel('Position')

plt.ylabel('Average Errors per Out')
errors_to['pos'].sort_values(by='e/to')