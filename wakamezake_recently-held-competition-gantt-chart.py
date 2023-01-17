import pandas as pd

import plotly.plotly as py

import plotly.figure_factory as ff

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)
competitions = pd.read_csv("../input/Competitions.csv")

competitions["EnabledDate"] = pd.to_datetime(competitions["EnabledDate"])

competitions["DeadlineDate"] = pd.to_datetime(competitions["DeadlineDate"])

target_categories = ["Featured"]

comp_featured = competitions[competitions["HostSegmentTitle"].isin(target_categories)]

# convert from pandas datetimeformat to plotly timestamp format

def timestamp_format(ts):

    str_format = "{}-{}-{}".format(ts.year, ts.month, ts.day)

    return str_format

# EnabledDate is 

# from 2018-01-01 to 2018-12-31

query = (comp_featured["EnabledDate"] >= "2018-01-01")

recently_comp = comp_featured[query]

df = [dict(Task=sl, Start=timestamp_format(begin), Finish=timestamp_format(end))

      for sl, begin, end in zip(recently_comp["Title"], recently_comp["EnabledDate"], recently_comp["DeadlineDate"])]

fig = ff.create_gantt(df)

iplot(fig, filename='gantt-simple-gantt-chart')