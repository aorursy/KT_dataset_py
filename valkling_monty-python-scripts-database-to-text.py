import numpy as np
import pandas as pd
import sqlite3

import os
print(os.listdir("../input"))
conn = sqlite3.connect('../input/database.sqlite')
df = pd.read_sql(con=conn, sql='select * from scripts')

df.character = df.character.astype(str)
df.actor = df.actor.astype(str)
df[:10]
%%time
All_MP_Scripts = ''
last_type = ''

for index, line in df.iterrows():
    type_of_line = line[4]
    actor = line[5]
    character = line[6]
    detail = line[7]
    Script = ''
    if type_of_line == 'Direction':
        if last_type == 'Direction':
            Script += ' '+detail
        else:
            Script += '<Direction: '+ detail+''
    else:
        if last_type == 'Direction':
            Script += "> \n\n"
        Script += character+'('+actor+'): '+ detail+' \n\n'
    last_type = type_of_line
    All_MP_Scripts += Script
print(All_MP_Scripts[:1000])
text_file = open("All_MP_Scripts.txt", "w")
text_file.write(All_MP_Scripts)
text_file.close()
