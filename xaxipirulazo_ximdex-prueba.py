# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json 

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
id_length = 7;


ventas = pd.read_csv('/kaggle/input/1.csv',delimiter = '\t')
ventas['PRODUCT'] = ventas['PRODUCT'].map(lambda id: ''.join([ '0' for char in range(id_length-len(str(id)))])+str(id)) 
with open('/kaggle/input/1.json') as json_file:
    categorias = json.load(json_file)['categories']
ventas

categorias

ventas2 = pd.read_csv('/kaggle/input/2.csv',delimiter = '\t')
ventas2['PRODUCT'] = ventas2['PRODUCT'].map(lambda id: ''.join([ '0' for char in range(id_length-len(str(id)))])+str(id)) 
with open('/kaggle/input/2.json') as json_file:
    categorias2 = json.load(json_file)['categories']
ventas2

categorias2
ventas3 = pd.read_csv('/kaggle/input/3.csv',delimiter = '\t')
ventas3['PRODUCT'] = ventas3['PRODUCT'].map(lambda id: ''.join([ '0' for char in range(id_length-len(str(id)))])+str(id)) 
with open('/kaggle/input/3.json') as json_file:
    categorias3 = json.load(json_file)['categories']
ventas3
categorias3
def beneficios(ventas,categorias):
    resultados = dict()
    comisiones = dict()
    
    for categoria in categorias.keys():
        comisiones[categoria] = transforma_comision(categorias[categoria])
        
    for i in range(ventas.shape[0]):
        categoria = ventas.iloc[i]['CATEGORY']
        if  categoria not in resultados.keys():
                resultados[categoria] = 0
        if categoria not in comisiones.keys():
            comision = comisiones['*']
        else: 
            comision = comisiones[categoria]
        
        precio = float(ventas.iloc[i]['COST'][:-1].replace('.','').replace(',','.'))
        cantidad = ventas.iloc[i]['QUANTITY']
        signo_comision_absoluta = comision[0]
        comision_absoluta = float(comision[1]) if comision[1] != '' else 0
        signo_comision_porcentaje = comision[2]
        comision_porcentaje = float(comision[3]) if comision[3] != '' else 0
        resultados[categoria] = float(resultados[categoria]) + comision_absoluta * cantidad if signo_comision_absoluta == '+' else float(resultados[categoria]) - comision_absoluta * cantidad
        if comision_porcentaje != 0:
            resultados[categoria] = float(resultados[categoria]) + (precio * comision_porcentaje /100)* cantidad  if signo_comision_porcentaje == '+' else float(resultados[categoria]) - (precio * comision_porcentaje /100)* cantidad

    return resultados
        

def transforma_comision(comision):
    descomposicion_valor_absoluto = comision.split('€')
    descomposicion_porcentaje = comision.split('%')
    abs_symbol,abs_value,percentage_symbol,percentage_value = ('','','','')
    if len(descomposicion_valor_absoluto)== 2: #hay comision por valor absoluto
        if  '%' not in descomposicion_valor_absoluto[0] :
            abs_symbol = descomposicion_valor_absoluto[0][0]
            abs_value = descomposicion_valor_absoluto[0][1:]
        else:
            index_percentage_next_char = len(descomposicion_valor_absoluto[0]) - descomposicion_valor_absoluto[0][::-1].index('%')
            abs_symbol = descomposicion_valor_absoluto[0][index_percentage_next_char]
            abs_value = descomposicion_valor_absoluto[0][index_percentage_next_char+1:]
            
    if len(descomposicion_porcentaje)== 2:
        if  '€' not in descomposicion_porcentaje[0]:
            percentage_symbol = descomposicion_porcentaje[0][0]
            percentage_value = descomposicion_porcentaje[0][1:]
    return abs_symbol,abs_value,percentage_symbol,percentage_value
            
            
        
    

        
    
    
beneficios(ventas,categorias)

beneficios(ventas2,categorias2)
beneficios(ventas3,categorias3)