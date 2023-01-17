import csv

import pprint

import time

import sys

import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import gc

from datetime import datetime

import random as rand

import math

from sklearn.metrics import roc_auc_score



dtypes = {

		'MachineIdentifier':									'category',

		'ProductName':											'category',

		'EngineVersion':										'category',

		'AppVersion':											'category',

		'AvSigVersion':											'category',

		'IsBeta':												'int8',

		'RtpStateBitfield':										'float16',

		'IsSxsPassiveMode':										'int8',

		'DefaultBrowsersIdentifier':							'float16',

		'AVProductStatesIdentifier':							'float32',

		'AVProductsInstalled':									'float16',

		'AVProductsEnabled':									'float16',

		'HasTpm':												'int8',

		'CountryIdentifier':									'int16',

		'CityIdentifier':										'float32',

		'OrganizationIdentifier':								'float16',

		'GeoNameIdentifier':									'float16',

		'LocaleEnglishNameIdentifier':							'int8',

		'Platform':												'category',

		'Processor':											'category',

		'OsVer':												'category',

		'OsBuild':												'int16',

		'OsSuite':												'int16',

		'OsPlatformSubRelease':									'category',

		'OsBuildLab':											'category',

		'SkuEdition':											'category',

		'IsProtected':											'float16',

		'AutoSampleOptIn':										'int8',

		'PuaMode':												'category',

		'SMode':												'float16',

		'IeVerIdentifier':										'float16',

		'SmartScreen':											'category',

		'Firewall':												'float16',

		'UacLuaenable':											'float32',

		'Census_MDC2FormFactor':								'category',

		'Census_DeviceFamily':									'category',

		'Census_OEMNameIdentifier':								'float16',

		'Census_OEMModelIdentifier':							'float32',

		'Census_ProcessorCoreCount':							'float16',

		'Census_ProcessorManufacturerIdentifier':				'float16',

		'Census_ProcessorModelIdentifier':						'float16',

		'Census_ProcessorClass':								'category',

		'Census_PrimaryDiskTotalCapacity':						'float32',

		'Census_PrimaryDiskTypeName':							'category',

		'Census_SystemVolumeTotalCapacity':						'float32',

		'Census_HasOpticalDiskDrive':							'int8',

		'Census_TotalPhysicalRAM':								'float32',

		'Census_ChassisTypeName':								'category',

		'Census_InternalPrimaryDiagonalDisplaySizeInInches':	'float16',

		'Census_InternalPrimaryDisplayResolutionHorizontal':	'float16',

		'Census_InternalPrimaryDisplayResolutionVertical':		'float16',

		'Census_PowerPlatformRoleName':							'category',

		'Census_InternalBatteryType':							'category',

		'Census_InternalBatteryNumberOfCharges':				'float32',

		'Census_OSVersion':										'category',

		'Census_OSArchitecture':								'category',

		'Census_OSBranch':										'category',

		'Census_OSBuildNumber':									'int16',

		'Census_OSBuildRevision':								'int32',

		'Census_OSEdition':										'category',

		'Census_OSSkuName':										'category',

		'Census_OSInstallTypeName':								'category',

		'Census_OSInstallLanguageIdentifier':					'float16',

		'Census_OSUILocaleIdentifier':							'int16',

		'Census_OSWUAutoUpdateOptionsName':						'category',

		'Census_IsPortableOperatingSystem':						'int8',

		'Census_GenuineStateName':								'category',

		'Census_ActivationChannel':								'category',

		'Census_IsFlightingInternal':							'float16',

		'Census_IsFlightsDisabled':								'float16',

		'Census_FlightRing':									'category',

		'Census_ThresholdOptIn':								'float16',

		'Census_FirmwareManufacturerIdentifier':				'float16',

		'Census_FirmwareVersionIdentifier':						'float32',

		'Census_IsSecureBootEnabled':							'int8',

		'Census_IsWIMBootEnabled':								'float16',

		'Census_IsVirtualDevice':								'float16',

		'Census_IsTouchEnabled':								'int8',

		'Census_IsPenCapable':									'int8',

		'Census_IsAlwaysOnAlwaysConnectedCapable':				'float16',

		'Wdft_IsGamer':											'float16',

		'Wdft_RegionIdentifier':								'float16',

		'HasDetections':										'int8'

		}

# 16383 (14bit) > 82 x 199 

column_max=83

column=column_max-1

table_row_max=column

row_max=199

bit=14



halfset=column/2



decision_serial=[51,9,1,6]

decision_serial_len=len(decision_serial)

#51+9x^1+1x^2+6x^3

best_serial=[51,9,1,6]

best_serial_len=len(best_serial)

best_rate=0.731

best_count=3000



def random_serial(serial, serial_len):

	serial_len=rand.randint(0,20)

	serial=[]

	i = 0

	while i < serial_len:

		serial.append(rand.randint(0,100))

		i=i+1

	serial_len=len(serial)

	return serial, serial_len



def print_serial(s1,serial, serial_len):

	s=s1+"="+str(serial[0])

	i = 1

	while i < serial_len:

		s=s+"+"+str(serial[i])+"x^"+str(i)

		i=i+1

	print(s)



def decision_method(number):

	i = 1

	y = decision_serial[0]

	while i < decision_serial_len:

		y = y + decision_serial[i]*math.pow(number,i)

		i=i+1

	return y



def decision_method_inc(decision_serial, decision_serial_len):

	carray=1

	i = 0

	while i < decision_serial_len:

		decision_serial[i] = decision_serial[i]+carray

		if (decision_serial[i] >= table_row_max):

			decision_serial[i] = 0

			carray = 1

			if (i+1) == decision_serial_len:

				decision_serial.append(1)

				decision_serial_len = len(decision_serial)

				break

		else:

			break

		i=i+1

	return decision_serial, decision_serial_len

		

def attribute_to_number(data):

	k=0

	sumup=0

	sumdown=0

	sumtotal=0



	if (type(data) == int):

		sumtotal=sumtotal+data

		if (j>=halfset):

			sumdown=sumdown+data

		if (j<halfset):

			sumup=sumup+data

	else:

		pattern = str(data)

		while k < len(pattern):

			col=pattern[k]

			sumtotal=int(sumtotal+ord(col))

			if (j>halfset):

				sumdown=int(sumdown+ord(col))

			if (j<halfset):

				sumup=int(sumup+ord(col))

			k=k+1

	mutation=int(sumtotal)%8

	mutation=decision_method(mutation)

	sumup=sumup % table_row_max

	sumdown=(sumdown+mutation) % row_max

	sumtotal=(sumtotal+mutation) % bit

	

	if (sumtotal >= bit/2):

		sumup = (int(sumup) ^ (0x01 << int(sumtotal-bit/2))) %table_row_max

	if (sumtotal < bit/2):

		sumdown = (int(sumdown) ^ (0x01 << int(sumtotal))) %row_max

	return sumup,sumdown



cal_value=np.zeros((column,2))



def print_correlation_rate(cal_value, best_rate, best_serial, best_serial_len, count, best_count):

	i=0

	sum=0

	vlist = list(dtypes.keys())

	a=[7,15,23,31,39,47]

	max=0

	while i < len(cal_value):

		vlen=len(vlist[i])

		shift=int(vlen/8)

		if vlen in a:

			shift = shift + 1

		shift_str=""

		for s in range(shift,7):

			shift_str=shift_str+"\t"

		print(vlist[i],shift_str+" correlation:",cal_value[i][1])

		sum = sum + cal_value[i][1]

		if max < cal_value[i][1]:

			max=cal_value[i][1]

		i=i+1

	reset=(max>0.69)

	if (best_rate*best_count < max*count and reset):

		best_rate=max

		best_count=count

		best_serial=decision_serial.copy()

		best_serial_len=len(best_serial)

	print_serial("best",best_serial,best_serial_len)

	

	print("average:", sum/i, " max:", max, " best:", best_rate, "best count: ", best_count, " seed:", seed)	

	return best_rate, best_serial, best_serial_len, reset, best_count

	

path="../input/"



skipstep=1

steprow=100000

submission=0

verify=1



default_auc=0

default_auc_seed=1549097396



bauc=0

bauc_seed=1



force=True



while True:

	if (bauc_seed == 0):

		bauc = default_auc

		bauc_seed = default_auc_seed

		seed=bauc_seed

	else:

		day=datetime.today()

		seed=math.floor(day.timestamp())

	

	print("seed:",seed)

	rand.seed(seed)

	decision_serial, decision_serial_len = random_serial(decision_serial,decision_serial_len)

	

	tables=np.zeros((table_row_max,row_max))

	skip=0

	beginrow=0

	stoprow=9000000

	reset=True

	print("starting train from ",beginrow," step ",steprow)



	start = time.time()

	while True:

		gc.collect()

		sys.stdout.write('\r'+"training "+str(skip+1)+" x "+str(steprow/1000)+"K")

		sys.stdout.flush()

		if ((beginrow+skip*steprow) > stoprow):

			print("\nstop training")

			break

		try:

			if (reset == True):

				rows=pd.read_csv(path+'train.csv', dtype=dtypes, nrows=steprow, skiprows=beginrow+skip*steprow, low_memory=False)

		except Exception as e:

			print("\nstop training ",str(e))

			time.sleep(10)

			break

		print("\n")

		

		#print(tables)

		best_rate, best_serial, best_serial_len, reset, best_count = print_correlation_rate(cal_value, best_rate, best_serial, best_serial_len, beginrow+skip*steprow, best_count)

		if force==True:

			reset=True

		#print(beginrow+skip*steprow)

		if  reset == False:

			decision_serial, decision_serial_len = decision_method_inc(decision_serial, decision_serial_len)

			beginrow=0

			skip=0

			cal_value=np.zeros((column,2))

			tables=np.zeros((table_row_max,row_max))

		print_serial("decision",decision_serial,decision_serial_len)

		

		for i,row in enumerate(rows.values, start=0):

			result=0

			j=0

			length = len(row)

			while j < length:

				if (j == column):

					break

				if row[column] == 1:

					result=1

				else:

					result=-1

				sumup, sumdown = attribute_to_number(row[j])

				sumup=int(sumup)

				sumdown=int(sumdown)

				if (tables[sumup][sumdown] >= 0):

					guess = 1

				else:

					guess = -1

				sample=cal_value[j][0]

				if (tables[sumup][sumdown] == 0):

					rate=cal_value[j][1]*sample + 1

					tables[sumup][sumdown] = int(tables[sumup][sumdown] + result)

				elif (result == guess): 

					rate=cal_value[j][1]*sample + 1

					tables[sumup][sumdown] = int(tables[sumup][sumdown] + guess)

				else:

					rate=cal_value[j][1]*sample

					tables[sumup][sumdown] = int(tables[sumup][sumdown] - guess)

				cal_value[j][0] = sample + 1

				cal_value[j][1] = rate/cal_value[j][0]



				j=j+1

		skip=skip+skipstep

		#del rows

	end = time.time()

	print("training spend",(end - start),"sec")



	print(tables)



	skip=0

	shift=rand.randint(4500000,8500000)

	beginrow=shift-steprow

	stoprow=shift

	verify1=[]

	verify2=[]

	print("starting verification from ",beginrow," step ",steprow)

	start = time.time()

	while True:

		gc.collect()

		sys.stdout.write('\r'+"verification "+str(skip+1)+" x "+str(steprow/1000)+"K")

		sys.stdout.flush()

		if ((beginrow+skip*steprow) > stoprow):

			print("\nstop verification")

			break

		try:

			rows=pd.read_csv(path+'train.csv', dtype=dtypes, nrows=steprow, skiprows=beginrow+skip*steprow, low_memory=False)

		except Exception as e:

			print("\nstop verification ",str(e))

			break



		for i,row in enumerate(rows.values, start=0):

			result=0

			result2=0

			j=0

			length = len(row)

			while j < length:

				if (j == column):

					verify1.append(row[j])

					break

				sumup,sumdown = attribute_to_number(row[j])

				sumup = int(sumup)

				sumdown = int(sumdown)

				if (tables[sumup][sumdown] >= 0):

					guess = 1

				else:

					guess = 0



				result = result + guess*cal_value[j][1]

				result2 = result2 + cal_value[j][1]

				j=j+1

			result=result/(result2)

			verify2.append(result)

		skip=skip+skipstep

		del rows

	end = time.time()

	auc=roc_auc_score(verify1,verify2)

	if (bauc < auc):

		bauc=auc

		bauc_seed=seed

		

	print("verification spend",(end - start),"sec","auc",auc,"best auc",bauc,"best seed",bauc_seed)

	del tables

	del verify2

	del verify1
