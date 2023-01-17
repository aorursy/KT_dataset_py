# The data comes as the raw data files, a transformed CSV file, and a SQLite database

import pandas as pd
import sqlite3

# You can read in the SQLite datbase like this
import sqlite3
con = sqlite3.connect('../input/database.sqlite')
raw = pd.read_sql('select cast(UNITID as int) UNITID, INSTNM, ZIP, cast(LO_INC_DEBT_MDN as int) LO_INC_DEBT_MDN, \
cast(PAR_ED_PCT_1STGEN as float) PAR_ED_PCT_1STGEN from Scorecard', con)
# drop duplicate schools (UNITID)
raw = raw.sort_values(by='UNITID')
print(raw.head(50))