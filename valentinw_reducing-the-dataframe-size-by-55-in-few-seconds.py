import pandas as pd
import numpy as np
import gc
# Rows importation
df = pd.read_csv('../input/train.csv', skiprows = 9308568, nrows = 59633310)

# Header importation
header = pd.read_csv('../input/train.csv', nrows = 0) 
df.columns = header.columns
df

# Cleaning
del header
gc.collect()

# And check his size        
print("The created dataframe contains", df.shape[0], "rows.")   
total_before_opti = sum(df.memory_usage())

# Type's conversions
def conversion (var):
    if df[var].dtype != object:
        maxi = df[var].max()
        if maxi < 255:
            df[var] = df[var].astype(np.uint8)
            print(var,"converted to uint8")
        elif maxi < 65535:
            df[var] = df[var].astype(np.uint16)
            print(var,"converted to uint16")
        elif maxi < 4294967295:
            df[var] = df[var].astype(np.uint32)
            print(var,"converted to uint32")
        else:
            df[var] = df[var].astype(np.uint64)
            print(var,"converted to uint64")
for v in ['ip', 'app', 'device','os', 'channel', 'is_attributed'] :
    conversion(v)
print("Memory usage before optimization :", str(round(total_before_opti/1000000000,2))+'GB')
print("Memory usage after optimization :", str(round(sum(df.memory_usage())/1000000000,2))+'GB')
print("We reduced the dataframe size by",str(round(((total_before_opti - sum(df.memory_usage())) /total_before_opti)*100,2))+'%')