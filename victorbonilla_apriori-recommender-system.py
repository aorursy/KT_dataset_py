import pandas as pd
import numpy as np

import datetime 
import time

%matplotlib inline
import matplotlib.pyplot as plt 
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
events_df = pd.read_csv('../input/events.csv')
category_tree_df = pd.read_csv('../input/category_tree.csv')
item_properties_1_df = pd.read_csv('../input/item_properties_part1.csv')
item_properties_2_df = pd.read_csv('../input/item_properties_part2.csv')
events_df.head()
#Which event has a value in its transaction id
events_df[events_df.transactionid.notnull()].event.unique()
#Which event/s has a null value
events_df[events_df.transactionid.isnull()].event.unique()
item_properties_1_df.head()
category_tree_df.head()
item_properties_1_df.loc[(item_properties_1_df.property == 'categoryid') & (item_properties_1_df.value == '1016')].sort_values('timestamp').head()
#Let's get all the customers who bought something
customer_purchased = events_df[events_df.transactionid.notnull()].visitorid.unique()
customer_purchased.size
#Let's get all unique visitor ids as well
all_customers = events_df.visitorid.unique()
all_customers.size
customer_browsed = [x for x in all_customers if x not in customer_purchased]
len(customer_browsed)
#Another way to do it using Numpy
temp_array = np.isin(customer_browsed, customer_purchased)
temp_array[temp_array == False].size
#A sample list of the customers who bought something
customer_purchased[:10]
events_df[events_df.visitorid == 102019].sort_values('timestamp')
tz = int('1433221332')
new_time = datetime.datetime.fromtimestamp(tz)
new_time.strftime('%Y-%m-%d %H:%M:%S')
tz = int('1438400163')
new_time = datetime.datetime.fromtimestamp(tz)
new_time.strftime('%Y-%m-%d %H:%M:%S')
# Firstly let's create an array that lists visitors who made a purchase
customer_purchased = events_df[events_df.transactionid.notnull()].visitorid.unique()
    
purchased_items = []
    
# Create another list that contains all their purchases 
for customer in customer_purchased:

    #Generate a Pandas series type object containing all the visitor's purchases and put them in the list
    purchased_items.append(list(events_df.loc[(events_df.visitorid == customer) & (events_df.transactionid.notnull())].itemid.values))                                  
purchased_items[:5]
# Number of unique items in transactions
events_df.loc[events_df.transactionid.notnull(), 'itemid'].unique().size
max_len=0
for tran in purchased_items:
    if len(tran) > max_len:
        max_len = len(tran)
        
print(f'Biggest purchase size: {max_len} items')
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
te = TransactionEncoder()
te_ary = te.fit(purchased_items).transform(purchased_items)
df_pi = pd.DataFrame(te_ary, columns=te.columns_)
df_pi.shape
df_pi.head()
frq_items = apriori(df_pi, min_support=0.001)
frq_items.head()
rules = association_rules(frq_items, metric ="confidence", min_threshold = 0.4)
rules
# TODO: factorization matrix or asso rules with item category???
# Write a function that would show items that were bought together (same of different dates) by the same customer
def recommender_bought_bought(item_id, purchased_items):
    
    # Perhaps implement a binary search for that item id in the list of arrays
    # Then put the arrays containing that item id in a new list
    # Then merge all items in that list and get rid of duplicates
    recommender_list = []
    for x in purchased_items:
        if item_id in x:
            recommender_list += x
    
    #Then merge recommender list and remove the item id
    recommender_list = list(set(recommender_list) - set([item_id]))
    
    return recommender_list
recommender_bought_bought(302422, purchased_items)
#Put all the visitor id in an array and sort it ascendingly
all_visitors = events_df.visitorid.sort_values().unique()
all_visitors.size
buying_visitors = events_df[events_df.event == 'transaction'].visitorid.sort_values().unique()
buying_visitors.size
viewing_visitors_list = list(set(all_visitors) - set(buying_visitors))

def create_dataframe(visitor_list):
    
    array_for_df = []
    for index in visitor_list:

        #Create that visitor's dataframe once
        v_df = events_df[events_df.visitorid == index]

        temp = []
        #Add the visitor id
        temp.append(index)

        #Add the total number of unique products viewed
        temp.append(v_df[v_df.event == 'view'].itemid.unique().size)

        #Add the total number of views regardless of product type
        temp.append(v_df[v_df.event == 'view'].event.count())

        #Add the total number of purchases
        number_of_items_bought = v_df[v_df.event == 'transaction'].event.count()
        temp.append(number_of_items_bought)

        #Then put either a zero or one if they made a purchase
        if(number_of_items_bought == 0):
            temp.append(0)
        else:
            temp.append(1)

        array_for_df.append(temp)
    
    return pd.DataFrame(array_for_df, columns=['visitorid', 'num_items_viewed', 'view_count', 'bought_count', 'purchased'])
buying_visitors_df = create_dataframe(buying_visitors)
buying_visitors_df.shape
#Let's shuffle the viewing visitors list for randomness
import random
random.shuffle(viewing_visitors_list)
viewing_visitors_df = create_dataframe(viewing_visitors_list[0:27820])
viewing_visitors_df.shape
main_df = pd.concat([buying_visitors_df, viewing_visitors_df], ignore_index=True)
#Let's shuffle main_df first
main_df = main_df.sample(frac=1)
#Plot the data
sns.pairplot(main_df, x_vars = ['num_items_viewed', 'view_count', 'bought_count'],
             y_vars = ['num_items_viewed', 'view_count', 'bought_count'],  hue = 'purchased')
X = main_df.drop(['purchased', 'visitorid', 'bought_count'], axis = 'columns')
y = main_df.purchased
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, train_size = 0.7)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
# Let's now use the model to predict the test features
y_pred_class = logreg.predict(X_test)
print('accuracy = {:7.4f}'.format(metrics.accuracy_score(y_test, y_pred_class)))
# Generate the prediction values for each of the test observations using predict_proba() function rather than just predict
preds = logreg.predict_proba(X_test)[:,1]

# Store the false positive rate(fpr), true positive rate (tpr) in vectors for use in the graph
fpr, tpr, _ = metrics.roc_curve(y_test, preds)

# Store the Area Under the Curve (AUC) so we can annotate our graph with theis metric
roc_auc = metrics.auc(fpr, tpr)

# Plot the ROC Curve
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw = lw, label = 'ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color = 'navy', lw = lw, linestyle = '--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc = "lower right")
plt.show()