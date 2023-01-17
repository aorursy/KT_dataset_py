#https://www.kaggle.com/tribedoor/support-files

#Read Out 1: Compare genome sequences if same then list the sequence, if different expand

#Read Out 2: Aggregate up the list states

#            If diff and equiv base pair then > "divergent"

#            If diff and non match base pair then ) "divergent"

#            If equall then < "convergent"

#            Line num= line in which seqence aggregation starts

#            This may illimate some sort of distribution on sequencing and resonance







import time

import re

import traceback

import os

import random

import sys

import glob

import csv

import math

import matplotlib.pyplot as plt





    

if __name__=="__main__":





    for dirname, _, filenames in os.walk('/kaggle/input'):

        for filename in filenames:

            print(os.path.join(dirname, filename))



    print('\n')

    

    LARGEMERS='/kaggle/input/support-files/DQ648857_.txt'

    DQ648857=[]

    with open(LARGEMERS) as lgmrs:

        for line in lgmrs:

            for ch in line:

                if ch!='\n':

                    DQ648857.append(str(ch))







    #>DQ412043 |Bat SARS coronavirus Rm1| complete genome

    LARGEMERS='/kaggle/input/support-files/DQ412043_.txt'

    DQ412043=[]

    with open(LARGEMERS) as lgmrs:

        for line in lgmrs:

            for ch in line:

                if ch!='\n':

                    DQ412043.append(str(ch))



     



   

                    

    

    

    

    #Difference Build  1

    diffarr=[]

    for dq,dqval in enumerate(DQ648857):

        try:

            if dqval!=DQ412043[dq]:

                diffarr.append([dqval,DQ412043[dq]])

            else:

                diffarr.append([dqval])

        except IndexError:

            arbit=1





    

    #difference analyais    

    diffcnt=0

    smecnt=0

    

    mutcnt=0

    mutsw=0

    cnstcnt=0

    cnstsw=0

    mcgrph=[]

    

    #frespth='C:/Users/The Architect/Desktop/Numerical Testing/Covid19/Result1.txt'

    #f=open(frespth,'w')



    print('\n')

    print('DQ648857 vs DQ412043----------------------------------------------------------------------Start')

    print('SEQUENCE COMPARE--------------------------------------------')

    for dif,diffval in enumerate(diffarr):

        

        #kaggle output consideration

        if dif <=100:

            print(str(diffval))

            

        #f.write(str(diffval)+'\n')

        if len(diffval)!=1:

            mutsw=1

            mutcnt+=1

            cnstsw=0

            cnstcnt=0

        else:

            cnstsw=1

            cnstcnt+=1

            mutsw=0

            mutcnt=0



        if mutcnt==1 or cnstcnt==1:

            mcgrph.append(f'{dif+1:08}'+':')



        if mutsw==1:

            mcchr='>'



            try:

                t=diffval.index('T')

            except ValueError:

                t=-1

                

            try:

                a=diffval.index('A')

            except ValueError:

                a=-1

                

            try:

                c=diffval.index('C')

            except ValueError:

                c=-1

                

            try:

                g=diffval.index('G')

            except ValueError:

                g=-1



            if (t!=-1 or a!=-1) and (c!=-1 or g!=-1):

                mcchr=')'

            

            mcgrph[len(mcgrph)-1]+=mcchr

            

        else:

            mcgrph[len(mcgrph)-1]+='<'



    #f.close()







    print('\n')

    print('VERT GRAPH--------------------------------------------------')

    #write the graph

    #frespth='C:/Users/The Architect/Desktop/Numerical Testing/Covid19/Result2.txt'

    #f=open(frespth,'w')

    for gr,grph in enumerate(mcgrph):

        if gr<=100:

            print(grph)

        #f.write(grph+'\n')

        

    #f.close()





    #difference analyais 2 with seq   

    diffcnt=0

    smecnt=0

    

    mutcnt=0

    mutsw=0

    cnstcnt=0

    cnstsw=0

    mcgrph=[]

    

    #frespth='C:/Users/The Architect/Desktop/Numerical Testing/Covid19/Result1.txt'

    #f=open(frespth,'w')



    for dif,diffval in enumerate(diffarr):

        



            

        #f.write(str(diffval)+'\n')

        if len(diffval)!=1:

            mutsw=1

            mutcnt+=1

            cnstsw=0

            cnstcnt=0

        else:

            cnstsw=1

            cnstcnt+=1

            mutsw=0

            mutcnt=0



        if mutcnt==1 or cnstcnt==1:

            mcgrph.append(f'{dif+1:08}'+':')



        if mutsw==1:

            mcchr=''.join(diffval)+'>'



            try:

                t=diffval.index('T')

            except ValueError:

                t=-1

                

            try:

                a=diffval.index('A')

            except ValueError:

                a=-1

                

            try:

                c=diffval.index('C')

            except ValueError:

                c=-1

                

            try:

                g=diffval.index('G')

            except ValueError:

                g=-1



            if (t!=-1 or a!=-1) and (c!=-1 or g!=-1):

                mcchr=''.join(diffval)+')'

            

            mcgrph[len(mcgrph)-1]+=mcchr

            

        else:

            mcgrph[len(mcgrph)-1]+='<'



    #f.close()





    print('\n')

    print('VERT GRAPH W DIV BP SEQ-------------------------------------')

    #write the graph

    #frespth='C:/Users/The Architect/Desktop/Numerical Testing/Covid19/Result2.txt'

    #f=open(frespth,'w')

    grph2vert=[]

    for gr,grph in enumerate(mcgrph):

        grph2vert.append(grph)

        if gr<=100:

            print(grph)

           

        #f.write(grph+'\n')

    print('\n')    

    print('DQ648857 vs DQ412043----------------------------------------------------------------------Finish')

 







   

  

 #>MN988669.1 Severe acute respiratory syndrome coronavirus 2 isolate 2019-nCoV WHU02, complete genome

    LARGEMERS='/kaggle/input/support-files/MN988669_.txt'

    MN988669=[]

    with open(LARGEMERS) as lgmrs:

        for line in lgmrs:

            for ch in line:

                if ch!='\n':

                    MN988669.append(str(ch))





    #>AY274119.3 Severe acute respiratory syndrome-related coronavirus isolate Tor2, complete genome

    LARGEMERS='/kaggle/input/support-files/AY274119_.txt'

    AY274119=[]

    with open(LARGEMERS) as lgmrs:

        for line in lgmrs:

            for ch in line:

                if ch!='\n':

                    AY274119.append(str(ch))  





    #Difference Build  2

    diffarr=[]

    for dq,dqval in enumerate(MN988669):

        try:

            if dqval!=AY274119[dq]:

                diffarr.append([dqval,AY274119[dq]])

            else:

                diffarr.append([dqval])

        except IndexError:

            arbit=1





    

    #difference analyais    

    diffcnt=0

    smecnt=0

    

    mutcnt=0

    mutsw=0

    cnstcnt=0

    cnstsw=0

    mcgrph=[]

    

    #frespth='C:/Users/The Architect/Desktop/Numerical Testing/Covid19/Result1.txt'

    #f=open(frespth,'w')



    print('\n')

    print('MN988669 vs AY274119----------------------------------------------------------------------Start')

    print('SEQUENCE COMPARE--------------------------------------------')

    for dif,diffval in enumerate(diffarr):

        

        #kaggle output consideration

        if dif <=100:

            print(str(diffval))

            

        #f.write(str(diffval)+'\n')

        if len(diffval)!=1:

            mutsw=1

            mutcnt+=1

            cnstsw=0

            cnstcnt=0

        else:

            cnstsw=1

            cnstcnt+=1

            mutsw=0

            mutcnt=0



        if mutcnt==1 or cnstcnt==1:

            mcgrph.append(f'{dif+1:08}'+':')



        if mutsw==1:

            mcchr='>'



            try:

                t=diffval.index('T')

            except ValueError:

                t=-1

                

            try:

                a=diffval.index('A')

            except ValueError:

                a=-1

                

            try:

                c=diffval.index('C')

            except ValueError:

                c=-1

                

            try:

                g=diffval.index('G')

            except ValueError:

                g=-1



            if (t!=-1 or a!=-1) and (c!=-1 or g!=-1):

                mcchr=')'

            

            mcgrph[len(mcgrph)-1]+=mcchr

            

        else:

            mcgrph[len(mcgrph)-1]+='<'



    #f.close()







    print('\n')

    print('VERT GRAPH--------------------------------------------------')

    #write the graph

    #frespth='C:/Users/The Architect/Desktop/Numerical Testing/Covid19/Result2.txt'

    #f=open(frespth,'w')

    for gr,grph in enumerate(mcgrph):

        if gr<=100:

            print(grph)

        #f.write(grph+'\n')

        

    #f.close()





    #difference analyais 2 with seq   

    diffcnt=0

    smecnt=0

    

    mutcnt=0

    mutsw=0

    cnstcnt=0

    cnstsw=0

    mcgrph=[]

    

    #frespth='C:/Users/The Architect/Desktop/Numerical Testing/Covid19/Result1.txt'

    #f=open(frespth,'w')



    for dif,diffval in enumerate(diffarr):

        



            

        #f.write(str(diffval)+'\n')

        if len(diffval)!=1:

            mutsw=1

            mutcnt+=1

            cnstsw=0

            cnstcnt=0

        else:

            cnstsw=1

            cnstcnt+=1

            mutsw=0

            mutcnt=0



        if mutcnt==1 or cnstcnt==1:

            mcgrph.append(f'{dif+1:08}'+':')



        if mutsw==1:

            mcchr=''.join(diffval)+'>'



            try:

                t=diffval.index('T')

            except ValueError:

                t=-1

                

            try:

                a=diffval.index('A')

            except ValueError:

                a=-1

                

            try:

                c=diffval.index('C')

            except ValueError:

                c=-1

                

            try:

                g=diffval.index('G')

            except ValueError:

                g=-1



            if (t!=-1 or a!=-1) and (c!=-1 or g!=-1):

                mcchr=''.join(diffval)+')'

            

            mcgrph[len(mcgrph)-1]+=mcchr

            

        else:

            mcgrph[len(mcgrph)-1]+='<'



    #f.close()





    print('\n')

    print('VERT GRAPH W DIV BP SEQ-------------------------------------')

    #write the graph

    #frespth='C:/Users/The Architect/Desktop/Numerical Testing/Covid19/Result2.txt'

    #f=open(frespth,'w')

    grph2vert=[]

    for gr,grph in enumerate(mcgrph):

        grph2vert.append(grph)

        if gr<=100:

            print(grph)

           

        #f.write(grph+'\n')

    print('\n')    

    print('MN988669 vs AY274119----------------------------------------------------------------------Finish')

            