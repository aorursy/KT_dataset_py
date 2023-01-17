# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from numpy import genfromtxt

import csv

import os

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



from os import listdir

from os.path import isfile, join

import time

import datetime

import sys





class bitmapCounter:

    def __init__(self, alphaError, isMonitored):

        self.alphaError = 0

        self.isMonitored = False





class fssElement:

    def __init__(self, value, frequency, error, counter, index):

        self.value = value

        self.frequency = frequency

        self.error = error

        self.counter = counter

        self.counter_index = index

        self.isEmpty = True;





class fss:

    def __init__(self, bitmap_counter_size, topN):

        # h - hash function for no of cells

        self.bitmapCounterSize_h = bitmap_counter_size



        # m - no of monitored elements in the list

        self.monitoredListSize_m = topN



        # Monitored list which holds elements that are monitored

        self.monitoredList = []

        for _i in range(self.monitoredListSize_m):

            self.monitoredList.append(fssElement('', 0, 0, 0, 0))



        # bitmap list which holds the counter

        self.bitmapList = []

        for _k in range(self.bitmapCounterSize_h):

            self.bitmapList.append(bitmapCounter(0, False))



        # is monitored list full

        self.isMonitoredListFull = False



    def getEmptyIndexFromMonitoredList(self):

        for _i in range(self.monitoredListSize_m):

            if self.monitoredList[_i].isEmpty is True:

                return _i

        return -1



    def appendToList(self, data):

        # calculate hash.

        # Note hash can return -ve value but we are using % which gives +ve

        # value. Hence we don't care.

        should_sort_list = False

        bitmap_index = hash(data) % self.bitmapCounterSize_h

        if self.bitmapList[bitmap_index].isMonitored is True:

            for element in self.monitoredList:

                if element.value == data:

                    element.frequency += 1

                    should_sort_list = True

            # Sort the list if needed

            if should_sort_list is True:

                self.monitoredList.sort(key=lambda x: x.frequency, reverse=True)

        else:

            if self.isMonitoredListFull is False:

                insert_index = self.getEmptyIndexFromMonitoredList();

                if insert_index is not -1:  # enter new entry into monitored list

                    self.monitoredList[insert_index].value = data

                    self.monitoredList[insert_index].frequency = self.bitmapList[bitmap_index].alphaError + 1

                    self.monitoredList[insert_index].error = self.bitmapList[bitmap_index].alphaError

                    self.monitoredList[insert_index].counter = self.bitmapList[bitmap_index]

                    self.monitoredList[insert_index].counter_index = bitmap_index

                    self.monitoredList[insert_index].isEmpty = False

                    self.bitmapList[bitmap_index].isMonitored = True;

                    should_sort_list = True

                    # Sort monitored list after this

                if should_sort_list is False:

                    self.isMonitoredListFull = True

            else:  # Replace last entry from the list

                if self.bitmapList[bitmap_index].alphaError + 1 > self.monitoredList[

                            self.monitoredListSize_m - 1].frequency:

                    monitored_counter_index = self.monitoredList[self.monitoredListSize_m - 1].counter_index

                    self.bitmapList[monitored_counter_index].isMonitored = False

                    self.bitmapList[monitored_counter_index].alpha_error = self.monitoredList[

                                                                               self.monitoredListSize_m - 1].frequency + 1



                    self.monitoredList[self.monitoredListSize_m - 1].value = data

                    self.monitoredList[self.monitoredListSize_m - 1].frequency = self.bitmapList[

                                                                                     bitmap_index].alphaError + 1

                    self.monitoredList[self.monitoredListSize_m - 1].error = self.bitmapList[bitmap_index].alphaError

                    self.monitoredList[self.monitoredListSize_m - 1].counter = self.bitmapList[bitmap_index]

                    self.monitoredList[self.monitoredListSize_m - 1].counter_index = bitmap_index

                    self.bitmapList[bitmap_index].isMonitored = True;

                    should_sort_list = True

                else:

                    self.bitmapList[bitmap_index].alphaError += 1

            if should_sort_list is True:

                self.monitoredList.sort(key=lambda x: x.frequency, reverse=True)



    def getTopN(self):

        # prints and returns top-n values

        topn = []

        for element in self.monitoredList:

            print ("value=", element.value, "frequency=", element.frequency, "error=", element.error)

            topn.append(element.value)

        return topn

    

    def getTopNWithFrequency(self):

        # prints and returns top-n values

        topn = {}

        for element in self.monitoredList:

            print ("value=", element.value, "frequency=", element.frequency, "error=", element.error)

            topn[element.value] = element.frequency

        return topn

    

    

if __name__ == '__main__':

    _fss = fss(43, 10);

    ts = time.time()

    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S:%f')

    print ("Started counting top values..", st)

    data = pd.read_csv("../input/complete.csv",  usecols=[2])

    # print (data)

    data = data.fillna({'state': 'missing'})

    #print (data)

    for index, row in data.iterrows():

        # print (row['state'])

        if row['state'] is not '' and row['state'] is not 'missing':

            _fss.appendToList(row['state'])        

    top = _fss.getTopNWithFrequency()

    ts = time.time()

    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S:%f')

    print ("Done with counting top values..", st, "\treturned top =", top)

    x = [1,2,3,4,5,6,7,8,9,10]

    plt.bar(x, top.values(), align='center')

    plt.xticks(x, top.keys())

    fig = plt.gcf()
