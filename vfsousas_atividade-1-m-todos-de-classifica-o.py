import pandas as pd
palavras_positivas = pd.read_csv("/kaggle/input/hoteldata/palavras_positivas.csv", sep=",")
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
palavras_positivas.sort_values(['High'], ascending=False).head(10)
palavras_negativas = pd.read_csv("/kaggle/input/hoteldata/palavras_negativas.csv", sep=",")
palavras_negativas.sort_values(['Low'], ascending=False).head(10)
hotel_data_with_country =  pd.read_csv("/kaggle/input/hoteldata/hotel_data_with_country.csv", sep=";")
hotel_data_with_country.groupby('Pais')['Pais'].agg(['count']).sort_values('count',ascending=False).head(10)
hotel_data_with_country =  pd.read_csv("/kaggle/input/hoteldata/hotel-data-with-avaliation.csv", sep=";")
hotel_data_with_country = hotel_data_with_country.where(hotel_data_with_country['Avaliacao'] == 'Low')

hotel_data_with_country.groupby('Nacionalidade_Revisor')['Nacionalidade_Revisor'].agg(['count']).sort_values('count',ascending=False).head(10)
hotel_data_result = pd.read_csv("/kaggle/input/hoteldata/hotel-data-with-avaliation.csv", sep=";")
hotel_data_result['Data_Revisao'] = pd.to_datetime(hotel_data_result['Data_Revisao'])



hotel_data_result['ano'] =  hotel_data_result.Data_Revisao.dt.year

hotel_data_result['mes'] =  hotel_data_result.Data_Revisao.dt.month



hotel_data_result_anomes = hotel_data_result.groupby(['Nome_Hotel','ano','mes'])['Nota_Revisao'].mean().reset_index()
hotel_data_result_anomes.fillna(0)

hotel_data_result_anomes = hotel_data_result.groupby(['Nome_Hotel','ano','mes'])['Nota_Revisao'].mean().reset_index()
hotel_data_result_best = hotel_data_result_anomes.groupby(['Nome_Hotel'])['Nota_Revisao'].mean().reset_index().head(10)
hotel_data_result_best.sort_values('Nota_Revisao',ascending=False).head(10)
hotel_data_result_best