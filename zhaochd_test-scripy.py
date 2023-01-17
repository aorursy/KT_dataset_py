# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost.sklearn import XGBClassifier


def get_year(date):
    date_list = str(date).split('-'); 
    return int(date_list[0])

def get_month(date):
    date_list = str(date).split('-'); 
    return int(date_list[1])

def get_day(date):
    date_list = str(date).split('-'); 
    return int(date_list[2])


def get_first_year(date):
    if date == date:
        return int(str(date)[:4])
    return date

def get_first_month(date):
    if date == date:
        return int(str(date)[4:6])
    return date

def get_first_day(date):
    if date == date:
        return int(str(date)[6:8])
    return date

def getWeekDay(date):
    return date.weekday();

def calculate_pct(device_type_time, total_time):
      return device_type_time/total_time if total_time > 0 else None

#sessions
sessions = pd.read_csv('../input/sessions.csv');
sessions.head()
