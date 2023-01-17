import pandas as pd               # for data manipulation
import matplotlib.pyplot as plt   # for plotting 
import seaborn as sns             # an extension of matplotlib for statistical graphics
orders = pd.read_csv('../input/orders.csv' ) #orders: type (orders) --> DataFrame
orders.shape #3421083 rows and 7 columns
orders.info() #presents all columns, variable types, and how many of each there are.
#the argument in .head() represents how many first rows we want to get.
orders.head(12) 
#prior: order of past date, test/train: projected (future) orders
#1. Import departments.csv from directory: ../input/departments.csv'
departments = pd.read_csv('../input/departments.csv')
departments.shape
departments.info ()
departments.head ()
orders.days_since_prior_order.max()
#orders['days_since_prior_order'].max ()
orders.days_since_prior_order.mean()
#orders['days_since_prior_order'].mean ()
orders.days_since_prior_order.median()
#orders['days_since_prior_order'].median ()
#7.0: 50% of the inputs are < 7.0 -- 50% of the inputs are > 7.0
# alternative syntax: orders.days_since_prior_order.plot(kind='box')
orders.boxplot('days_since_prior_order')
#orders['days_since_prior_order'].plot (kind = 'box')
#I can either call .boxplot or call .plot and specify the kind of plot I want (scatter, box, etc.)
orders.head(20)


order_hours = orders.order_hour_of_day.value_counts()
#Βρίσκει διακριτές τιμές και μου επιστρέφει τη συχνότητα της κάθε μίας. Ξεκινάει από την τιμή που εμφανίζεται συχνότερα.
order_hours
#alternative syntax : order_hours.plot(kind='bar')
order_hours.plot.bar() 
#Remember that the alias that we have defined for seaborn is the sns.
sns.countplot(x="order_hour_of_day", data=orders, color='red')
#A plot with sorted variables. we can now see first 0, 1, 2, 3, ... 23, hours in order
#Εδώ κάνω απευθείας Plot χωρίς να κάνω νέα μεταβλητή. Στο x ορίζω τα δεδομένα μου (order_hour_of_day --> ώρες της ημέρας) και στο data ορίζω από ποιό DataFrame λαμβάνω αυτά τα στοιχεία.
# Step one - define the dimensions of the plot (15 for x axis, 5 for y axis)
plt.figure(figsize=(15,5))

# Step two - define the plot that we want to produce with seaborn
# Here we also define the color of the bar chart as 'red'
sns.countplot(x="order_hour_of_day", data=orders, color='red')

# Step three - we define the name of the axes and we add a title to our plot
# fontsize indicates the size of the titles
plt.ylabel('Total Orders', fontsize=10)
plt.xlabel('Hour of day', fontsize=10)
plt.title("Frequency of order by hour of day", fontsize=15)

# Step four - we produce our plot
plt.show()
sns.countplot(x='order_dow', data=orders )
plt.figure(figsize=(10,10))
sns.countplot(x="order_dow", data=orders, color='red')
plt.ylabel('Orders', fontsize=10)
plt.xlabel('Day of the Week', fontsize=10)
plt.title("Orders per Day", fontsize=15)
plt.show()
orders_first = orders.loc [orders['order_number'] == 1]
orders_first.head(20)
orders_second = orders.loc[orders['order_number'] == 2]
orders_second.head(20)

#create a subplot which contains two plots; one down the other
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,8))

#assign each plot to the appropiate axes
sns.countplot(ax= axes[0], x='order_dow', data=orders_first, color='red')
sns.countplot(ax= axes[1], x='order_dow', data=orders_second, color='red')

# produce the final plot
plt.show()
orders.head(15)
order_count = orders.order_number.value_counts()
order_count
# Set size 15x5 and bar color red
plt.figure(figsize=(15,5))
sns.countplot(x='order_number', data=orders, color='red')
plt.ylabel('Total Customers', fontsize=10)
plt.xlabel('Total Orders', fontsize=10)
plt.show()
#import the required function
import matplotlib.ticker as ticker

plt.figure(figsize=(15,5))

#assign plot in a variable
ax = sns.countplot(x='order_number', data=orders, color='red')
plt.ylabel('Total Customers', fontsize=10)
plt.xlabel('Total Orders', fontsize=10)

#select the step for xticks (2)
ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

plt.show()