import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
# Number of employee =n
n=11
np.random.seed(n)  

# Assuming the office will start at 9 am and will close at 6pm. 
# So the final time to enter into the office is 2 pm (Because we were assuming minimum shift of 4 hours)
# Assuming military time. So 1pm will be identified as 13.
# So column can be generated as Mon9 Mon10 ... Mon14 Tue9 Tue10 ... Tue14....
# However, we can edit list day and time as our wish
day=['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
time= ['9', '10', '11', '12', '13', '14']
col_len=len(day) * len(time)
col = [[] for i in range(col_len)]

k=0
for i in range(len(day)):
    for j in range (len(time)):
        col[k]=day[i]+time[j]
        k=k+1
#df = pd.DataFrame(np.random.randn(n, col_len), columns=[col])
df = pd.DataFrame(np.random.randint(20,120,size=(n, col_len)), columns=[col])

df
df.to_csv('random_time.csv')
df2 = pd.DataFrame(np.random.randint(4,9,size=(n, 5)), columns=['Shift_duration_Mon','Shift_duration_Tue','Shift_duration_Wed','Shift_duration_Thu','Shift_duration_Fri'])
df2
df2.to_csv('random_shift_len.csv')
