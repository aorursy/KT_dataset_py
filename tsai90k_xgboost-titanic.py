# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt # for plotting graphs
import seaborn # statistical data visualization library
from functools import cmp_to_key

import tensorflow as tf
from tensorflow.python.data import Dataset
tf.logging.set_verbosity(tf.logging.ERROR)

import xgboost as xgb
from sklearn import metrics
import math
import copy
import sys

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/"))
# Any results you write to the current directory are saved as output.
class BucketMapper(object):
    def __init__(self, field_info, assume_no_na = False):
        self.field_info = copy.copy(field_info)
        self.digit_keys = list(self.field_info.keys())
        self.digit_keys.sort()
        self.digits = []
        self.digits_rev = []
        self.multipiliers = []
        self.total_comb = 1
        self.assume_no_na = assume_no_na
        self.add_for_na = 1
        if self.assume_no_na == True:
            self.add_for_na = 0
        # for example:
        # 1. (consider N/A) If field_info = {'A':['a','b'],'B':['x']}
        #    then combinations will be 3 * 2:
        #    0 -> 0 0 -> (na, na)   -> [1, 0, 0, 0, 0, 0]
        #    1 -> 0 1 -> (na, 'x')  -> [0, 1, 0, 0, 0, 0]
        #    2 -> 1 0 -> ('a', na)  -> [0, 0, 1, 0, 0, 0]
        #    3 -> 1 1 -> ('a', 'x') -> [0, 0, 0, 1, 0, 0]
        #    4 -> 2 0 -> ('b', na)  -> [0, 0, 0, 0, 1, 0]
        #    5 -> 2 1 -> ('b', 'x') -> [0, 0, 0, 0, 0, 1]
        #
        #    then self.digit_keys = ['A','B']
        #    then self.digits will become
        #    [
        #        {'a':1, 'b':2}, # implies that N/A = 0
        #        {'x':1},        # implies that N/A = 0
        #    ]
        #    and self.multipiliers will become
        #    [
        #        2,              # multipiliers of digits of 'A' field
        #        1,              # multipiliers of digits of 'B' field
        #    ]
        #
        #    so ('a', 'x') will be (1*2) + (1*1) = 3
        #    and ('b', 'x') will be (2*2) + (1*1) = 5
        #
        # 2. (assume no N/A) If the field info = {'A':['a','b','c'],'B':['x','y']}
        #    NOTE: if there is no N/A, then a field MUST have at least 2 categories.
        #    then combination will be 3 * 2:
        #    0 -> 0 0 -> ('a', 'x') -> [1, 0, 0, 0, 0, 0]
        #    1 -> 0 1 -> ('a', 'y') -> [0, 1, 0, 0, 0, 0]
        #    2 -> 1 0 -> ('b', 'x') -> [0, 0, 1, 0, 0, 0]
        #    3 -> 1 1 -> ('b', 'y') -> [0, 0, 0, 1, 0, 0]
        #    4 -> 2 0 -> ('c', 'x') -> [0, 0, 0, 0, 1, 0]
        #    5 -> 2 1 -> ('c', 'y') -> [0, 0, 0, 0, 0, 1]
        #
        #    then self.digit_keys = ['A','B']
        #    then self.digits will become
        #    [
        #        {'a':0, 'b':1, 'c':2},
        #        {'x':0, 'y':1},
        #    ]
        #    and self.multipiliers will become
        #    [
        #        2,              # multipiliers of digits of 'A' field
        #        1,              # multipiliers of digits of 'B' field
        #    ]
        #
        #    so ('a', 'x') will be (0*2) + (0*1) = 0
        #    and ('b', 'y') will be (1*2) + (1*1) = 3
        
        # check each field to ensure every felds have at least 2 categories
        # include or not include N/A
        for k in self.digit_keys:
            if (len(self.field_info[k]) + self.add_for_na) < 2:
                if self.assume_no_na == True:
                    raise ValueError('the field %s must have at least 2 categories')
                else:
                    raise ValueError('the field %s must have at least 1 category')
        for i in range(0, len(self.digit_keys)):
            k = self.digit_keys[i]
            next_i = i + 1
            # a higher digit's base (mul) is decided by variation of its lower digit
            mul = 1
            if next_i < len(self.digit_keys):
                next_k = self.digit_keys[next_i]
                mul = len(self.field_info[next_k]) + self.add_for_na
            dig = copy.copy(self.field_info[k])
            self.total_comb *= (len(dig) + self.add_for_na)
            dig_d = {}
            # with N/A, i starts from 1 (0 for N/A)
            # without N/A, i starts from 0
            i = self.add_for_na
            for v in dig:
                dig_d[v] = i
                i += 1
            self.multipiliers.append(1)
            self.multipiliers = [_m * mul for _m in self.multipiliers]
            self.digits.append(dig_d)
            self.digits_rev.append(dict([(_v,_k) for (_k,_v) in dig_d.items()]))

    def fields_to_bucket_id(self, fields):
        n = 0
        for i in range(0, len(fields)):
            mul = self.multipiliers[i]
            dig_d = self.digits[i]
            k = fields[i]
            try:
                n += (dig_d[k] * mul)
            except KeyError as e:
                if self.assume_no_na == True:
                    raise ValueError('unexpected value: %s is not allowed because assume_no_na is True' % (str(e)))
                else:
                    n += 0
        return n
   
    def bucket_id_to_fields(self, bucket_id):
        fields = []
        for i in range(0, len(self.digit_keys)):
            mul = self.multipiliers[i]
            dig_rev_d = self.digits_rev[i]
            digit = bucket_id // mul
            bucket_id = bucket_id % mul
            try:
                fields.append(str(dig_rev_d[digit]))
            except KeyError as e:
                if self.assume_no_na == True:
                    raise ValueError('unexpected value: %s is not allowed because assume_no_na is True' % (str(e)))
                else:
                    fields.append('na')
        return str.join('_', fields)
            
    def to_bucketlized(self, df, name):
        df1 = df[self.digit_keys]
        df2 = pd.DataFrame()
        s = []
        for i in range(0, len(df1)):
            r = [n for n in df1.iloc[i]]
            bucket_id = self.fields_to_bucket_id(r)
            s.append(bucket_id)
        df2[name] = s
        return df2
    
    def to_one_hot(self, df, name):
        df1 = df[self.digit_keys]
        df2 = pd.DataFrame()
        s = []
        for i in range(0, self.total_comb):
            s.append([])
        for i in range(0, len(df1)):
            r = [n for n in df1.iloc[i]]
            bucket_id = self.fields_to_bucket_id(r)
            for j in range(0, self.total_comb):
                s[j].append(0)
            s[bucket_id][-1] = 1
        for i in range(0, self.total_comb):
            field_name = self.bucket_id_to_fields(i)
            df2['%s_%s' % (name, field_name)] = s[i]
        return df2
df_train = pd.read_csv('../input/train.csv')
df_train.describe()
print('record number of df_train = %d' % (len(df_train)))
df_train.head(50)
df_test = pd.read_csv('../input/test.csv')
df_test.describe()
print('record number of df_test = %d' % (len(df_test)))
# combine training data and test data
df_all = df_test.copy()
df_all = df_all.append(df_train, ignore_index = True)
# list all fields
print('total fields = %d' % (len(df_train.keys())))
print('fields = %s' % (df_train.keys()))
# check N/A fields
df_all.drop('Survived', axis = 1).loc[:,df_all.isnull().any()].isnull().sum()
def log_no_error(n):
    if n <= 0:
        return 0.0
    else:
        return np.log1p(n)

def min2(l, default = 0.0):
    if len(l) == 0:
        return default
    else:
        return min(l)

def max2(l, default = 0.0):
    if len(l) == 0:
        return default
    else:
        return max(l)

def avg2(l, default = 0.0):
    if len(l) == 0:
        return default
    else:
        return float(sum(l)) / float(len(l))

def std2(l, default = 0.0):
    if len(l) == 0:
        return default
    else:
        return np.std(l)

def histogram_for_non_numerical_series(s):
    d = {}
    for v in s:
        d[v] = d.get(v, 0) + 1
    bin_s_label = list(d.keys())
    bin_s_label.sort()
    bin_s = list(range(0, len(bin_s_label)))
    hist_s = [d[v] for v in bin_s_label]
    bin_s.append(len(bin_s))
    bin_s_label.insert(0, '_')
    return (hist_s, bin_s, bin_s_label)
    
def plot_hist_with_target3(plt, df, feature, target, histogram_bins = 10):
    # reference:
    #    https://stackoverflow.com/questions/33328774/box-plot-with-min-max-average-and-standard-deviation
    #    https://matplotlib.org/gallery/api/two_scales.html 
    #    https://matplotlib.org/1.2.1/examples/pylab_examples/errorbar_demo.html
    #    https://matplotlib.org/2.0.0/examples/color/named_colors.html
    #    https://matplotlib.org/api/_as_gen/matplotlib.pyplot.xticks.html
    title = feature
    plt.title(title)
    s = df[feature]
    t = df[target]
    t_max = max(t)
    # get histogram of the feature
    bin_s_label = None
    # fillna with 0.0 or '_N/A_'
    na_cnt = sum(s.isna())
    if na_cnt > 0:
        if True in [type(_) == str for _ in s]:
            print('found %d na in string field %s' % (na_cnt, feature))
            s = s.fillna('_N/A_')
        else:
            print('found %d na in numerical field %s' % (na_cnt, feature))
            s = s.fillna(0.0)
    try:
        hist_s, bin_s = np.histogram(s, bins = histogram_bins)
    except Exception as e:
        # print('ERROR: failed to draw histogram for %s: %s: %s' % (name, type(e).__name__, str(e)))
        hist_s, bin_s, bin_s_label = histogram_for_non_numerical_series(s)
        # return
    # histogram of target by distribution of feature
    hist_t_by_s_cnt = [0] * (len(bin_s) - 1)
    hist_t_by_s = [] 
    for i in range(0, (len(bin_s) - 1)):
        hist_t_by_s.append([])
    # get target histogram for numerical feature
    if bin_s_label is None:
        for (sv, tv) in zip(s, t):
            pos = 0
            for i in range(0, len(bin_s) - 1):
                if sv >= bin_s[i]:
                    pos = i
            hist_t_by_s_cnt[pos] += 1
            hist_t_by_s[pos].append(tv)
        bin_s_new = []
        hist_t_by_s_new = []
        bin_s_new.append(bin_s[0])
        for (bv, hv) in zip(bin_s[1:], hist_t_by_s):
            if len(hv) != 0:
                bin_s_new.append(bv)
                hist_t_by_s_new.append(hv)
        bin_s_err = bin_s_new
        hist_t_by_s = hist_t_by_s_new
        # print('target_hist:\n%s(%d)\n%s(%d)' % (bin_s, len(bin_s), hist_t_by_s, len(hist_t_by_s)))
    else:
        for (sv, tv) in zip(s, t):
            pos = bin_s_label.index(sv) - 1
            hist_t_by_s_cnt[pos] += 1
            hist_t_by_s[pos].append(tv)
        # count avg, to re-sort bin_s and bin_s_label by avg
        hist_t_by_s_avg = [float(avg2(n)) for n in hist_t_by_s]
        # hist_t_by_s_std = [float(std2(n)) for n in hist_t_by_s]
        # hist_t_by_s_adj = list(np.array(hist_t_by_s_avg) + np.array(hist_t_by_s_std))
        hist_t_by_s_adj = hist_t_by_s_avg
        # print('before sort:\n%s\n%s\n%s' % (bin_s, bin_s_label, hist_t_by_s_adj))
        bin_hist_label = list(zip(bin_s[1:], hist_t_by_s_adj, bin_s_label[1:]))
        bin_hist_label.sort(key = cmp_to_key(lambda x, y: x[1] - y[1]))
        (bin_s, hist_t_by_s_adj, bin_s_label) = zip(*bin_hist_label)
        bin_s = list(bin_s)
        hist_t_by_s_adj = list(hist_t_by_s_adj)
        bin_s_label = list(bin_s_label)
        bin_s.insert(0, 0)
        bin_s_label.insert(0, '_')
        # re-arrange hist_s and hist_t_by_s
        hist_s_new = []
        hist_t_by_s_new = []
        for i in bin_s[1:]:
            hist_s_new.append(hist_s[i - 1])
            hist_t_by_s_new.append(hist_t_by_s[i - 1])
        hist_s = hist_s_new
        hist_t_by_s = hist_t_by_s_new
        # print('after sort:\n%s\n%s\n%s' % (bin_s, bin_s_label, hist_t_by_s_adj))
        # reset bin_s's ordering
        bin_s.sort()
        bin_s_err = bin_s
    hist_s = list(hist_s)
    if len(hist_s) < len(bin_s):
        hist_s.insert(0, 0.0)
    hist_s_max = max(hist_s)
    plt.fill_between(bin_s, hist_s, step = 'mid', alpha = 0.5, label = feature)
    if bin_s_label is not None:
        plt.xticks(bin_s, bin_s_label)
    plt.xticks(rotation = 90)
    # just to show legend for ax2
    # plt.errorbar([], [], yerr = [], fmt = 'ok', lw = 3, ecolor = 'sienna', mfc = 'sienna', label = target)
    plt.legend(loc = 'upper right')
    hist_t_by_s = list(hist_t_by_s)
    if len(hist_t_by_s) < len(bin_s):
        hist_t_by_s.insert(0, [0.0])
    hist_t_by_s_min = [float(min2(n)) for n in hist_t_by_s]
    hist_t_by_s_max = [float(max2(n)) for n in hist_t_by_s]
    hist_t_by_s_avg = [float(avg2(n)) for n in hist_t_by_s]
    # hist_t_by_s_std = [float(std2(n)) for n in hist_t_by_s][1:]
    hist_t_by_s_25_percentile = [float(np.percentile(n, 25)) for n in hist_t_by_s]
    hist_t_by_s_75_percentile = [float(np.percentile(n, 75)) for n in hist_t_by_s]
    hist_t_by_s_std = [[max(0, _) for _ in (np.array(hist_t_by_s_avg) - np.array(hist_t_by_s_25_percentile))[1:]],
                       [max(0, _) for _ in (np.array(hist_t_by_s_75_percentile) - np.array(hist_t_by_s_avg))[1:]]]
    hist_t_by_s_err = [(np.array(hist_t_by_s_avg) - np.array(hist_t_by_s_min))[1:], (np.array(hist_t_by_s_max) - np.array(hist_t_by_s_avg))[1:]]
    plt.xlabel(feature)
    plt.ylabel('Count')
    ax2 = plt.twinx()
    ax2.grid(False)
    ax2.errorbar(bin_s_err[1:], hist_t_by_s_avg[1:], yerr = hist_t_by_s_err, fmt='.k', lw = 1, ecolor = 'sienna')
    ax2.errorbar(bin_s_err[1:], hist_t_by_s_avg[1:], yerr = hist_t_by_s_std, fmt='ok', lw = 3, ecolor = 'sienna', mfc = 'sienna', label = target)
    ax2.set_ylabel(target)
    plt.legend(loc = 'upper left')
    plt.tight_layout()
    
def plot_df(df, y, fields = None):
    if fields is None:
        fields = df.keys()
    figs = len(fields)
    cols = 4
    rows = int(figs / cols)
    if (rows * cols) < figs:
        rows += 1
    plt.figure(figsize = (5 * cols, 5 * rows))
    i = 1
    for name in fields:
        if name == y:
            continue
        plt.subplot(rows, cols, i)
        plot_hist_with_target3(plt, df, name, y, histogram_bins = 'rice')
        i += 1
    plt.tight_layout()
# draw relationship between target and features
plot_df(df_train.drop('Name', axis = 1), y = 'Survived')
plt.figure(figsize = (10, 10))
seaborn.heatmap(df_train.corr())
_df = df_train.copy()
plot_df(_df[['Pclass','Survived']], y = 'Survived')
plot_df(_df[_df['Sex'] == 'female'][['Pclass','Survived']], y = 'Survived')
plot_df(_df[_df['Sex'] == 'male'][['Pclass','Survived']], y = 'Survived')
_df['Cabin'] = df_train['Cabin'].apply(lambda x: x[0] if type(x) == str else 'nan')

plot_df(_df[['Cabin','Survived']], y = 'Survived')
plot_df(_df[_df['Sex'] == 'female'][['Cabin','Survived']], y = 'Survived')
plot_df(_df[_df['Sex'] == 'male'][['Cabin','Survived']], y = 'Survived')
_df['Ticket'] = df_train['Ticket'].apply(lambda x: x[0] if type(x) == str else 'num')
plot_df(_df[['Ticket','Survived']], y = 'Survived')
plot_df(_df[_df['Sex'] == 'female'][['Ticket','Survived']], y = 'Survived')
plot_df(_df[_df['Sex'] == 'male'][['Ticket', 'Survived']], y = 'Survived')
# convert N/A age to 100 to see its distribution
_df2 = _df.copy()
_df2['Age'] = _df2['Age'].fillna(100)
plot_df(_df2[['Age', 'Survived']], y = 'Survived')
plot_df(_df2[_df2['Sex'] == 'female'][['Age', 'Survived']], y = 'Survived')
_df[_df['Sex'] == 'female']['Age'].median()
plot_df(_df2[_df2['Sex'] == 'male'][['Age', 'Survived']], y = 'Survived')
_df[(_df['Age'] > 62) & (_df['Sex'] == 'male')]
_df[_df['Sex'] == 'male']['Age'].median()
_df['Fare'].median()
plot_df(_df[['Fare', 'Survived']], y = 'Survived')
plot_df(_df[_df['Sex'] == 'female'][['Fare', 'Survived']], y = 'Survived')
plot_df(_df[_df['Sex'] == 'male'][['Fare', 'Survived']], y = 'Survived')
_df[(_df['Sex'] == 'male') & (_df['Fare'] > 200) & (_df['Fare'] < 500)].sort_values(by = ['Fare'])
_df[(_df['Sex'] == 'male') & (_df['Fare'] > 500)].sort_values(by = ['Fare'])
plot_df(_df[['SibSp', 'Survived']], y = 'Survived')
plot_df(_df[_df['Sex'] == 'female'][['SibSp', 'Survived']], y = 'Survived')
plot_df(_df[_df['Sex'] == 'male'][['SibSp', 'Survived']], y = 'Survived')
plot_df(_df[['Parch', 'Survived']], y = 'Survived')
plot_df(_df[_df['Sex'] == 'female'][['Parch', 'Survived']], y = 'Survived')
plot_df(_df[_df['Sex'] == 'male'][['SibSp', 'Survived']], y = 'Survived')
plot_df(_df[['Embarked', 'Survived']], y = 'Survived')
plot_df(_df[_df['Sex'] == 'female'][['Embarked', 'Survived']], y = 'Survived')
plot_df(_df[_df['Sex'] == 'male'][['Embarked', 'Survived']], y = 'Survived')
_df['SibSp_Parch'] = _df['SibSp'] + _df['Parch']
plot_df(_df[['SibSp_Parch', 'SibSp', 'Parch', 'Survived']], y = 'Survived')
plot_df(_df[_df['Sex'] == 'female'][['SibSp_Parch', 'SibSp', 'Parch', 'Survived']], y = 'Survived')
plot_df(_df[_df['Sex'] == 'male'][['SibSp_Parch', 'SibSp', 'Parch', 'Survived']], y = 'Survived')
fix_funcs = []

'''
Following fields sohuld be categorical fields
  - Pclass
'''
def fix_df_pclass_to_cat(df):
    df1 = df.copy()
    for k in ['Pclass']:
        s = df1[k]
        df1[k] = s.astype(str)
    return df1

fix_funcs.append(fix_df_pclass_to_cat)

'''
1. The first latter of Cabin will be the category name, replace N/A with 'nan' category
2. The first latter of Ticket will be the category name, replace N/A with 'nan' category
'''
def fix_df_cabin_and_ticket(df):
    df1 = df.copy()
    for k in ['Cabin', 'Ticket']:
        df1[k] = df1[k].apply(lambda x: x[0] if type(x) == str else 'nan')
    return df1

fix_funcs.append(fix_df_cabin_and_ticket)

'''
1. Add SibSp and Parch + 1 as faimly number
2. Convert Fare to 4 level
3. Convert Age to 5 level
'''
def wrap_fix_df_copied_from_kaggle():
    d = {}
    def _fix_df_copied_from_kaggle(df):
        df1 = df.copy()
        # FamilyNum
        df1['FamilyNum'] = df1['SibSp'] + df1['Parch'] + 1
        # IsAlone, single passenger?
        df1['IsAlone'] = df1['FamilyNum'].apply(lambda x: 1 if x == 1 else 0)
        # bucketlize Fare to 4 quantilies
        if not ('fare_med' in d):
            d['fare_med'] = df1['Fare'].median()
        fare_med = d['fare_med']
        df1['FareLevel'] = df1['Fare'].fillna(fare_med)
        if not ('fare_bin' in d):
            d['fare_bin'] = list(pd.qcut(df1['FareLevel'], 4).unique())
            d['fare_bin'].sort()
        fare_bin = d['fare_bin']
        df1['FareLevel'] = df1['FareLevel'].apply(lambda x: sum([x >= b.left for b in fare_bin]))
        # bucketlize Age to 5 buckets
        if not ('age_med' in d):
            d['age_med'] = df1['Age'].median()
        age_med = d['age_med']
        df1['AgeLevel'] = df1['Age'].fillna(age_med)
        if not ('age_bin' in d):
            d['age_bin'] = list(pd.cut(df1['AgeLevel'], 5).unique())
            d['age_bin'].sort()
        age_bin = d['age_bin']
        df1['AgeLevel'] = df1['AgeLevel'].apply(lambda x: sum([x >= b.left for b in age_bin]))
        return df1
    return _fix_df_copied_from_kaggle

#fix_funcs.append(wrap_fix_df_copied_from_kaggle())

'''
Converts Age to categorical feature, 10 years a category
Age has N/A value, use median value.
'''
def wrap_fix_age():
    med_l = []
    def _fix_age(df):
        df1 = df.copy()
        if len(med_l) == 0:
            med_l.append(df1['Age'].median())
        med = med_l[0]
        df1['Age'] = df1['Age'].fillna(med)
        df1['Age'] = df1['Age'].apply(lambda x: str(int(x / 10) * 10))
        return df1
    return _fix_age

fix_funcs.append(wrap_fix_age())

'''
Converts Fare to categorical feature, 100 a category
Fare has N/A value, use median value.
'''
def wrap_fix_fare():
    med_l = []
    def _fix_fare(df):
        df1 = df.copy()
        if len(med_l) == 0:
            med_l.append(df1['Fare'].median())
        med = med_l[0]
        df1['Fare'] = df1['Fare'].fillna(med)
        df1['Fare'] = df1['Fare'].apply(lambda x: str(int(x / 100) * 100))
        return df1
    return _fix_fare

fix_funcs.append(wrap_fix_fare())

'''
Converts SibSp to no(0), less(1~2), Many(3+)
'''
def fix_sibsp(df):
    df1 = df.copy()
    df1['SibSp'] = df1['SibSp'].apply(lambda x: 'no' if x == 0 else 'less' if x <= 2 else 'many')
    return df1

fix_funcs.append(fix_sibsp)


'''
Converts Parch to no(0), less(0~3), Many(4+)
'''
def fix_parch(df):
    df1 = df.copy()
    df1['Parch'] = df1['Parch'].apply(lambda x: 'no' if x == 0 else 'less' if x <= 3 else 'many')
    return df1

fix_funcs.append(fix_parch)


'''
Converts N/A in Embarked to a new category most repeated value
'''
def fix_embarked(df):
    df1 = df.copy()
    most_repeated_value = df1['Embarked'].mode()[0]
    df1['Embarked'] = df1['Embarked'].fillna(most_repeated_value)
    return df1

fix_funcs.append(fix_embarked)

'''
Convert categorical fields to binned fields and do cross features
'''
def wrap_fix_bin_and_cross():
    bm_d = {}
    def _fix_bin_and_cross(df):
        df1 = df.copy()
        for kl in [
                   ['SibSp','Parch'],
                   ['Sex','Age'],
                   #['Sex', 'Fare'],
                   ['Sex'], 
                   ['Cabin'],
                   ['Pclass'],
                   #['SibSp'],
                   #['Parch'],
                   ['Fare'],
                   ['Ticket'], 
                   ['Embarked'], 
                   ['Age']
                  ]:
            kname = str.join('_', kl)
            if not (kname in bm_d):
                field_info = {}
                for k in kl:
                    field_info[k] = df1[k].unique()
                bm_d[kname] = BucketMapper(field_info, assume_no_na = True)
            bm = bm_d[kname]
            df_kl = df1[kl]
            df_bin = bm.to_one_hot(df_kl, kname)
            df1 = df1.join(df_bin)
        return df1
    return _fix_bin_and_cross

fix_funcs.append(wrap_fix_bin_and_cross())


'''
Drop fields
'''
def fix_df_drop_fields(df):
    df1 = df.copy()
    for k in [
              'Name',
              'PassengerId',
              'Sex',
              'Pclass',
              'Cabin',
              'Parch',
              'Fare',
              'SibSp', 
              'Ticket',
              'Embarked',
              'Age'
             ]:
        df1.drop(k, axis = 1, inplace = True)
    return df1

fix_funcs.append(fix_df_drop_fields)

def fix_df(df, funcs):
    df1 = df.copy()
    for func in funcs:
       df1 = func(df1)
    return df1

df_train[df_train['Sex'] == 'male']

print('fix df_train2')
df_train2 = fix_df(df_train, fix_funcs)

print('fix df_test2')
df_test2 = fix_df(df_test, fix_funcs)

print('features in df_train2 (%d) = %s' % (len(df_train2.keys()), list(df_train2.keys())))
print('features in df_test2 (%d) = %s' % (len(df_test2.keys()), list(df_test2.keys())))

features = list(df_train2.keys())
features.remove('Survived')


# draw relationship between target and features
# plot_df(df_train2, y = 'Survived')
def calc_acc_at_threshold(y_prob, y_true, threshold):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = np.array([1 if p >= th else 0 for p in y_prob])
    acc = sum(y_pred == y_true).astype(float) / len(y_true)
    return acc

def train_xgb_classifier(param,
                         df,
                         features,
                         target,
                         rounds):
    param = param.copy()
    param['silent'] = 1
    # DMatrix of training set
    mat = xgb.DMatrix(df[features], 
                      df[target])
    # train model with best tree number
    xgb_classifier = xgb.train(param,
                               mat,
                               num_boost_round = rounds)
    return xgb_classifier

def plot_cv_results(cv_results, metrics):
    items = []
    for m in metrics:
        items.append(('train-%s-mean', 'test-%s-mean'))
    plt.figure(figsize = (5 * len(items), 5))
    n = 0
    for item in items:
        n += 1
        plt.subplot(1, len(items), n)
        for lb in item:
            plt.plot(cv_results[lb], label = lb)
            plt.legend()
    plt.tight_layout()
    

def cv(param,
       df,
       features,
       target,
       nfold,
       rounds,
       metrics):
    '''
    @return: a dict contains at least:
        {
            'test-error-mean': a_list,
            'test-error-mean': a_list,
        }
    '''
    train_mat = xgb.DMatrix(df[features],
                            df[target])
    cv_results = xgb.cv(param,
                        train_mat,
                        num_boost_round = rounds,
                        nfold = nfold,
                        metrics = metrics)
    return cv_results 

def train_and_validate(param,
                       df_train,
                       df_validate,
                       features,
                       target,
                       rounds,
                       threshold):
    param = param.copy()
    param['silent'] = 1
    cv_stats = pd.DataFrame()
    # DMatrix of training set
    train_mat = xgb.DMatrix(df_train[features], 
                            df_train[target])
    # DMatrix of validation set
    validate_mat = xgb.DMatrix(df_validate[features], 
                               df_validate[target])
    # train the model
    classifier = train_xgb_classifier(param = param,
                                      df = df_train,
                                      features = features,
                                      target = target,
                                      rounds = rounds)
    # predict the training data
    train_prop = classifier.predict(train_mat)
    # find best threshold for training set
    train_acc = calc_acc_at_threshold(train_prop, df_train[target], threshold)
    # predict the valication data
    validate_prop = classifier.predict(validate_mat)
    validate_acc = calc_acc_threshold(validate_prop, df_validate[target], threshold)
    # stats
    stats = {
        'train-acc@%.2f' % (threshold): train_acc,
        'train-props': train_prop,
        'train-labels': np.array(df_train[target]),
        'validate-acc@%.2f' % (threshold): validate_acc,
        'validate-props': validate_prop,
        'validate-labels': np.array(df_validate[target]),
    }
    return stats

def iter_combinations(comb_list):
    (name, values) = comb_list[0]
    for v in values:
        if len(comb_list) > 1:
            for p in iter_combinations(comb_list[1:]):
                yield [(name, v)] + p
        else:
            yield [(name, v)]
        
def grid_search(grid_param,
                eval_metrics,
                param,
                df,
                features,
                target,
                nfold,
                rounds):
    grid_stats = pd.DataFrame()
    df = df.reindex(np.random.permutation(df.index))
    df = df.reset_index(drop = True)
    list_metric_names = []
    for em in eval_metrics:
        list_metric_names.append('test-%s-mean' % (em))
        list_metric_names.append('train-%s-mean' % (em))
    for p in iter_combinations(grid_param):
        pdict = dict(p)
        cv_param = param.copy()
        cv_param.update(pdict)
        cv_param['silent'] = 1
        cv_results = cv(param = cv_param,
                        df = df,
                        features = features,
                        target = target,
                        nfold = nfold,
                        rounds = rounds,
                        metrics = eval_metrics)
        # find the round with minimun error
        cv_stats = pd.DataFrame(cv_results)
        for em in eval_metrics:
            metric_name = 'test-%s-mean' % (em)
            metric_min = cv_stats[metric_name].min()
            idx = cv_stats[cv_stats[metric_name] == metric_min].index.values.astype(int)[0]
            if len(grid_stats) == 0 or grid_stats[metric_name].min() > metric_min:
                stats = cv_stats.iloc[idx:idx + 1][list_metric_names]
                prefix = str.join(', ', ['%s=%s' % (_k, _v) for (_k, _v) in p])
                print('%s, metric=%s, best_round=%s: %s' % (prefix, em, idx + 1, dict(stats.iloc[0])))
                for (_k, _v) in p:
                    stats[_k] = [_v]
                stats['best_round'] = [idx + 1]
                stats['metric'] = [em]
                grid_stats = grid_stats.append(stats, ignore_index = True)
    return grid_stats

eval_metrics = ['error', 'logloss']
grid_param = [
    ('learning_rate',[0.5,0.1,0.05]),
    ('max_depth',[2,4,6,8]),
    #('gamma',[0.0,1.0,2.0]),
    #('subsample',[1.0,0.5]),
    #('min_child_weight',[0,1,2]),
    #('lambda',[0,1,2]),
    #('alpha',[0,1,2]),
]
train_param = {'objective':'binary:logistic'}
grid_stats = grid_search(grid_param = grid_param,
                         eval_metrics = eval_metrics,
                         param = train_param,
                         df = df_train2,
                         features = features,
                         target = 'Survived',
                         nfold = 3,
                         rounds = 300)
# get the best hyper parameters
eval_metrics = ['error', 'logloss']

best_grid_stats = pd.DataFrame()
for em in eval_metrics:
    metric_name = 'test-%s-mean' % (em)
    best_grid_stats = best_grid_stats.append(grid_stats[grid_stats[metric_name] == grid_stats[metric_name].min()])
                       
best_grid_stats
# train with full data
df_train3 = df_train2.reindex(np.random.permutation(df_train2.index))
df_train3 = df_train2.reset_index(drop = True)
train_param = {'objective':'binary:logistic', 'learning_rate':0.05, 'max_depth':4}
xgb_classifier = train_xgb_classifier(param = train_param,
                                      df = df_train3,
                                      features = features,
                                      target = 'Survived',
                                      rounds = 90)
print(xgb_classifier.get_fscore())
fig = plt.figure(figsize = (20, 40))
ax = plt.subplot(1,1,1)
xgb.plot_importance(xgb_classifier, ax = ax)
# predict the test data
threshold = 0.5
test_mat = xgb.DMatrix(df_test2[features])
test_prop = xgb_classifier.predict(test_mat)
test_pred = np.array([1 if _ >= threshold else 0 for _ in test_prop])

df_submit = pd.DataFrame()
df_submit['PassengerId'] = df_test['PassengerId']
df_submit['Survived'] = test_pred
df_submit.to_csv('./test_prediction_xgboost.csv', index = False)