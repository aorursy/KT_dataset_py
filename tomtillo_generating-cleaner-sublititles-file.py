import re 

import os 

import pandas as pd 
DIR = '../input/chai-time-data-science/Raw Subtitles/'

L_fname_txt = os.listdir(DIR)

L_fname_csv = [k.replace('.txt','.csv') for k in L_fname_txt]   # create a list of same files with .csv ext
#Just a function to write one line to a file - not the efficient way, but yeah works 

def Fn_write_line(sLine,fname):

    fname = '../working/'+fname

    target = open(fname, 'a')

    target.write(sLine)

    target.close()
for fname_txt in L_fname_txt:

    fname = open(DIR+fname_txt, 'r') 

    Lines = fname.readlines() 

    pattern1 = "  [0-9]+:[0-9]+  "   # time pattern eg: 10:43

    pattern2 = "  [0-9]+:[0-9]+:[0-9]+  " # time pattern eg: 1:00:14  



    for line in Lines: 

        if (re.search(pattern1,line)) or (re.search(pattern2,line) ):  # line [8]

            line = line.replace("  ","_-_").replace("\n","")  # line [9] - See comments below

        if len(line) >1 :    # Line [10]

            Fn_write_line(line,fname_txt)

            df = pd.read_csv( '../working/'+fname_txt,sep = "_-_" , header = None)

            df.columns = ['speaker','time','text']

            df = df[1:-1:]   # line [14]

            df.to_csv('../working/'+fname_txt.replace('.txt','.csv'),index=False)    
L_fname_csv
L_fname_csv[18]
df = pd.read_csv( '../working/'+ L_fname_csv[18] )

df.head()