# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
%matplotlib inline

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

sql_conn = sqlite3.connect('../input/database.sqlite')

list_of_tables = sql_conn.execute("SELECT * FROM sqlite_master where type='table'")
print(list_of_tables.fetchall())
pd.read_sql("SELECT * from Sentiment", sql_conn)
query = """SELECT candidate,
        SUM(CASE sentiment WHEN 'Positive' THEN 1 ELSE 0 END) AS positive,
        SUM(CASE sentiment WHEN 'Negative' THEN 1 ELSE 0 END) as negative,
        SUM(CASE sentiment WHEN 'Neutral' THEN 1 ELSE 0 END) AS neutral
        FROM Sentiment 
        GROUP BY candidate 
        ORDER BY 3 DESC,4 DESC"""
sentiment_by_candidate = pd.read_sql(query, sql_conn)
sentiment_by_candidate