import pandas as pd

import numpy as np

import seaborn as sns

from statsmodels.stats.weightstats import zconfint

from scipy.stats import mannwhitneyu

from statsmodels.sandbox.stats.multicomp import multipletests 

%matplotlib inline
posts = pd.read_csv('../input/post.csv', parse_dates=['timeStamp'])
comments = pd.read_csv('../input/comment.csv')
com_count = comments.groupby('pid').count()['cid']

data = posts.join(com_count,on='pid', rsuffix='c')[['msg', 'likes', 'shares', 'cid', 'gid']]

data.columns = ['msg', 'likes', 'shares', 'comments', 'gid']

data['msg_len'] = data.msg.apply(len)
#117291968282998 Elkins Park Happenings

#25160801076 Unofficial Cheltenham Township

#1443890352589739 Free Speech Zone

data.gid = data.gid.map({117291968282998: 1, 25160801076: 2, 1443890352589739: 3})
data.fillna(0,inplace=True)

data.head()
sns.pairplot(data, hue='gid')
park = data[data.gid == 1]

town = data[data.gid == 2]

free = data[data.gid == 3]



def conf_interval(field):

    """"

    Calculate confidence interval for given field

    """

    # I've rounded numbers to integers because estimated values (likes, shares, ...) are integers themselves.

    print("95% confidence interval for the EPH posts mean number of {:s}: ({z[0]:.0f}, {z[1]:.0f})".format(field, z=zconfint(park[field])))

    print("95% confidence interval for the UCT posts mean number of {:s}: ({z[0]:.0f}, {z[1]:.0f})".format(field, z=zconfint(town[field])))

    print("95% confidence interval for the FSZ posts mean number of {:s}: ({z[0]:.0f}, {z[1]:.0f})".format(field, z=zconfint(free[field])))
conf_interval('likes')
def compare_means(field):

    """

    Mann–Whitney test to compare mean values level

    """

    mapping = {1: 'EPH', 2: 'UCT', 3: 'FSZ'}

        

    comparison = pd.DataFrame(columns=['group1', 'group2', 'p_value'])

    # compare number of <field> in each group 

    for i in range(1,4):

        for j in range(1,4):

            if i >= j:

                continue

            # obtaining p-value after Mann–Whitney U test

            p = mannwhitneyu(data[data.gid == i][field], data[data.gid == j][field])[1]

            comparison = comparison.append({'group1': mapping[i], 'group2': mapping[j], 'p_value': p},ignore_index=True)

    # holm correction

    rejected, p_corrected, a1, a2 = multipletests(comparison.p_value, 

                                            alpha = 0.05, 

                                            method = 'holm') 

    comparison['p_value_corrected'] = p_corrected

    comparison['rejected'] = rejected

    return comparison    
conf_interval('likes')

print(compare_means('likes'))

# compare number of likes in group1 with number of likes in group2, 

# and if rejected field is True make a conclusion that 

# mean number of likes in the first group is different from mean number of likes in the second one.  
conf_interval('shares')

print(compare_means('shares'))
conf_interval('comments')

print(compare_means('comments'))
conf_interval('msg_len')

print(compare_means('msg_len'))
shared = data[data.shares > data.shares.quantile(0.98)][data.shares > data.likes*10][['msg','shares']]



top = 10

print("top %d out of %d" % (top, shared.shape[0]))

sorted_data = shared.sort_values(by='shares', ascending=False)[:top]

for i in sorted_data.index.values:

    print('shares:',sorted_data.shares[i], '\n','message:', sorted_data.msg[i][:200], '\n')
likes = data[data.likes > data.likes.quantile(0.98)][data.likes > data.shares*100][['msg', 'likes']]

print("top %d out of %d" % (top, likes.shape[0]))

sorted_data = likes.sort_values(by='likes', ascending=False)[:top]

for i in sorted_data.index.values:

    print('likes:',sorted_data.likes[i], '\n','message:', sorted_data.msg[i][:300], '\n')
discussed = data[data.comments > data.comments.quantile(0.98)][['msg', 'comments']]



print("top %d out of %d\n" % (top, discussed.shape[0]))

sorted_data = discussed.sort_values(by='comments', ascending=False)[:top]

for i in sorted_data.index.values:

    print('comments:',sorted_data.comments[i], '\n','message:', sorted_data.msg[i][:300], '\n')