import numpy as np 
import pandas as pd 
import os
from datetime import date, timedelta
import pandas as pd
from datetime import datetime
import ipywidgets as widgets

def all_wednesdays(year):
    d = date(year, 1, 1)                    # January 1st
    d += timedelta(days = 2 - d.weekday())  # First Wednesday
    while d.year == year:
        yield d
        d += timedelta(days = 7)

def add_active(activity, name, started, ended):
    activity[name] = (started <= activity['date']) & (activity['date'] <= ended)
    
def show_between(start, end, membership):
    wednesdays = [x for x in all_wednesdays(start.year) if x >= start and x<= end]
    activity = pd.DataFrame()
    activity['date'] = wednesdays
    activity.set_index('date')
    membership.apply(lambda row: add_active(activity, row['name'], row['start'], row['end']), axis=1)
    return activity

def display_activity(activity):
    to_pay = pd.DataFrame()
    to_pay['date'] = activity['date']
    for name in membership['name']:
        to_pay[name] = activity[name] * activity['price_pp']
        to_pay.at['total', name] = to_pay[name].sum()
    from IPython.display import display, HTML
    table = HTML(to_pay.to_html())
    display(table)
    
def format_activity(activity, meetup_fee=30):
    activity['headcount'] = activity.sum(axis=1)
    activity['price_pp'] = meetup_fee / activity['headcount']
    activity['total'] = activity['price_pp']


# membership dates
membership = pd.DataFrame(data={	
    "name": 	["Tomasz", "Uli", "Louis", "Steffen", "Bruce", "Dylan", "Nick", "Tom", "Aerrow"], 
	"start": 	["20180101","20180101", "20180101", "20180101", "20180601", "20180501", "20181001", "20181201", "20181201"],
	"end": 		["20190111",None,None,None,None,None,None,None, None]})
membership[['start', 'end']] = membership[['start', 'end']] \
            .applymap(lambda x: datetime.strptime(x, "%Y%m%d").date() if x != None else date(2099, 1, 1))
print(membership)

# fall 2018
attendance_fall_2018 = show_between(date(2018, 9, 1), date(2018, 12, 20), membership)
format_activity(attendance_fall_2018)
display_activity(attendance_fall_2018)

# 2019
attendance_winter_2019 = show_between(date(2019, 1, 8), date(2019, 3, 30), membership)
format_activity(attendance_winter_2019)
display_activity(attendance_winter_2019)