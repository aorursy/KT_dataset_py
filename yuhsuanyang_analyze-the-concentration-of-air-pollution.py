import pandas as pd
import numpy as np
from datetime import datetime
from collections import Counter
import math
import matplotlib.pyplot as pl

def convergeMarkov(MarkovMat,iter_n):
    iterHistory={}
    iterHistory[1]=np.dot(MarkovMat,MarkovMat)
    for i in range(2,iter_n):
        iterHistory[i] = np.dot(MarkovMat,iterHistory[i-1])
        for y in range(3):
            for z in range(3):
                iterHistory[i][y][z] = np.round(iterHistory[i][y][z],4)       
    test_n = 1
    
    while not((iterHistory[test_n][0][0]==iterHistory[test_n+1][0][0]) and (iterHistory[test_n][0][1]==iterHistory[test_n+1][0][1])and(iterHistory[test_n][0][2]==iterHistory[test_n+1][0][2])and(iterHistory[test_n][1][0]==iterHistory[test_n+1][1][0])and(iterHistory[test_n][1][1]==iterHistory[test_n+1][1][1])and(iterHistory[test_n][1][2]==iterHistory[test_n+1][1][2])and(iterHistory[test_n][2][0]==iterHistory[test_n+1][2][0])and(iterHistory[test_n][2][1]==iterHistory[test_n+1][2][1])and(iterHistory[test_n][2][2]==iterHistory[test_n+1][2][2])):
    #while not((iterHistory[test_n]==iterHistory[test_n+1])):  
        test_n = test_n+1
    print('   The matrix converge at the',test_n,'th time!!!')
    return iterHistory,test_n



def mark10(x,matrixname):
    month=[90,91,92,92]
    p=month[x]
    pm10season=np.zeros((25,p))
    for i in range(25):
        for j in range(p):
            pm10season[i][j]=pm10dailyColor[i][j+sum(month[0:x])]
    pm10seasonmark=np.zeros((3,3))
    
    for j in range(p-1):
      for i in range(25):
        if pm10season[i][j] == 1:
            n = 1 
        elif pm10season[i][j] == 2:
            n = 2
        elif pm10season[i][j] == 3:
            n = 3
        else:
            n = 0
            
        if pm10season[i][j+1] == 1:
            m = 1 
        elif pm10season[i][j+1] == 2:
            m = 2
        elif pm10season[i][j+1] == 3:
            m = 3
        else:
            m = 0            
        
        if(n != 0 and m != 0):
            pm10seasonmark[n-1][m-1] = pm10seasonmark[n-1][m-1]+1

    for i in range(3):
        if sum(pm10seasonmark[i])==0:
            matrixname[i]=0
        else:
            matrixname[i]=pm10seasonmark[i]/sum(pm10seasonmark[i])
    return matrixname

def mark25(x,matrixname):
    month=[90,91,92,92]
    p=month[x]
    pm25season=np.zeros((25,p))
    for i in range(25):
        for j in range(p):
            pm25season[i][j]=pm25dailyColor[i][j+sum(month[0:x])]
    pm25seasonmark=np.zeros((4,4))
    
    for j in range(p-1):
      for i in range(25):
        if pm25season[i][j] == 1:
            n = 1 
        elif pm25season[i][j] == 2:
            n = 2
        elif pm25season[i][j] == 3:
            n = 3
        elif pm25season[i][j] == 4:
            n = 4 
        else:
            n = 0
            
        if pm25season[i][j+1] == 1:
            m = 1 
        elif pm25season[i][j+1] == 2:
            m = 2
        elif pm25season[i][j+1] == 3:
            m = 3
        elif pm25season[i][j+1] == 4:
            m = 4
        else:
            m = 0            
        
        if(n != 0 and m != 0):
            pm25seasonmark[n-1][m-1] = pm25seasonmark[n-1][m-1]+1

    for i in range(4):
        if sum(pm25seasonmark[i])==0:
            matrixname[i]=0
        else:
            matrixname[i]=pm25seasonmark[i]/sum(pm25seasonmark[i])
    return matrixname