%%time
#  このセル（jupyter でのプログラムの実行単位です。）は、ライブラリを読み込み、sqlite データと接続して、一時的なテーブルを作成します。
# pandas というデータを扱うライブラリを読み込みます。
# このプログラムでは、データ操作はＳＱＬを通じて行うので、pandas は、グラフを書くために使うのが主です。
import pandas as pd # sql の検索結果を保存する
import numpy as np  # pandas に付随するもの、1次元の操作の時に使う
import pandas.io.sql as psql # sql と pandas を結びつける

# sqliteのライブラリ
import sqlite3

# 線形回帰　使う頻度はあまりないですが、とりあえず、いれておきます。
from sklearn import linear_model
clf = linear_model.LinearRegression()

# HTMLで表示する　エクセルにコピペするときに便利かも
from IPython.display import display, HTML

# markdown 用 qiita に
from tabulate import tabulate

# 日時を扱う
from datetime import datetime as dt
import time

# グラフ
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import ticker
%matplotlib inline

# system 関係のライブラリ
import sys
# os の機能を使うライブラリ
import os

# 正規表現
import re

# json,yaml 形式を扱う
import json
import yaml

# 変数の状態を調べる
import inspect

# 文字コード　日本語の文字コードが sjis のときに役立ちます。
import codecs

# Web からデータを取得する
import requests

# 貿易統計のデータ
# http://www.customs.go.jp/toukei/info/tsdl_e.htm
# コード　輸出は日本語のみ
# https://www.customs.go.jp/toukei/sankou/code/code_e.htm 

# sqlite に show tables がないので補足するもの
show_tables = "select tbl_name from sqlite_master where type = 'table'"
# describe もないで、補完します。
desc = "PRAGMA table_info([{table}])"
# メモリで、sqlite を使います。kaggle のスクリプト上では、オンメモリでないと新規テーブルがつくれません
# プログラムの一行が長いときは　\　で改行します。
conn = \
    sqlite3.connect(':memory:')

# sql を実行するための変数です。
cursor = conn.cursor()

# 1997 年から、2019 年までの年ベースのデータです。テーブルは、year_from_1997 
# year_from_1997
attach = 'attach "../input/japan-trade-statistics/y_1997.db" as y_1997'
cursor.execute(attach)

# 2018 年の月別集計 テーブル名も ym_2018 
attach = 'attach "../input/japan-trade-statistics/ym_2018.db" as ym_2018'
cursor.execute(attach)

# 2019 年の月別集計 テーブル名も ym_2019
attach = 'attach "../input/japan-trade-statistics/ym_2019.db" as ym_2019'
cursor.execute(attach)

# 2020 年の月別集計 テーブル名も ym_2020
attach = 'attach "../input/japan-trade-statistics/ym_2020.db" as ym_2020'
cursor.execute(attach)

# hs code,country,HSコードです。使いやすいように pandas　に変更しておきます。
attach = 'attach "../input/japan-trade-statistics/codes.db" as code'
cursor.execute(attach)
# import hs,country code as pandas
tmpl = "{hs}_{lang}_df =  pd.read_sql('select * from code.{hs}_{lang}',conn)"
for hs in ['hs2','hs4','hs6','hs6','hs9']:
    for lang in ['jpn','eng']:
        exec(tmpl.format(hs=hs,lang=lang))        

# 国コードも pandas で扱えるようにします。
# country table: country_eng,country_jpn
country_eng_df = pd.read_sql('select * from code.country_eng',conn)
country_eng_df['Country']=country_eng_df['Country'].apply(str)
country_jpn_df = pd.read_sql('select * from code.country_jpn',conn)
country_jpn_df['Country']=country_jpn_df['Country'].apply(str)

# custom  table: code.custom 税関別のコードです
custom_df = pd.read_sql('select * from code.custom',conn)
attach = 'attach "../input/japan-trade-statistics/custom_from_2012.db" as custom_from'
cursor.execute(attach)
attach = 'attach "../input/custom-2016/custom_2018.db" as custom_2018'
cursor.execute(attach)
attach = 'attach "../input/custom-2016/custom_2019.db" as custom_2019'
cursor.execute(attach)

attach = 'attach "../input/japan-trade-statistics/custom_2020.db" as custom_2020'
cursor.execute(attach)

# 計算時間を節約するために、年のデータから、2019 年を切り出します。
# 最初のはエラー処理です。y_2019 というテーブルが存在すると、新規に y_2019 を作ろうとするとエラーになります。
# error の場合は、何もせず、次にすすみます。
try:
    cursor.execute('drop table y_2019')
except:
    pass

# これからが、SQl になります。複数行で書くことが多いのでsql という変数に複数行を代入します。
# 最後の [1:-1] は、一行目（改行で空白）と最後の行（これも改行だけで空白）をとりのぞくためです。
# 0 から始まるので、1 だと、２行目から最後の行のひとつ手前までです。
sql = """
create table y_2019 
as select * from year_from_1997
where Year = 2019
"""[1:-1]
# 上記の sql を実行しして、2019 年のデータをつくります。
cursor.execute(sql)
conn.commit()
# sql の説明です。
# create table テーブル名 : テーブルを新規作成　ここでは、y_2019 
# as select * from  テーブル名　: テーブル名(year_from_1997)からつくります。
# where Year = 2019 : 2019 年のデータを指定します。Year は、数値なので、2019 と書きます。

# ym_2018,ym_2019 に ym_2020 を加えて,ym_2018_2020 を作成 
# 年＋月のカラム(ym)をつくります。
# 年は、整数、月は、文字列なので、型変換と文字列結合を行います。
# CAST(Year AS  str )||month as ym がその処理をしている部分です。
# 文字列結合が、|| なので違和感ありますが、しょうがないです。

try:
    cursor.execute('drop table ym_2018_2020')
except:
    pass

sql = """
create table ym_2018_2020
as select CAST(Year AS  str )||month as ym,* from ym_2018
"""[1:-1]
cursor.execute(sql)


sql = """
insert into  ym_2018_2020
 select CAST(Year AS  str )||month as ym,* from ym_2019
"""[1:-1]
cursor.execute(sql)

sql = """
insert into  ym_2018_2020
 select CAST(Year AS  str )||month as ym,* from ym_2020
"""[1:-1]
cursor.execute(sql)
conn.commit()
# 便利なクラス（sql 実行 + グラフ）ut.関数名で使います。
class util():
    def sql(self,sql):
        # sql で抽出されたデータを pandas の形式で戻します。
        return(pd.read_sql(sql,conn))
 
    # 折れ線グラフ 一系列　色は、b ( blue )
    def g1(self,df,x,y,color='b'):
        plt.figure(figsize=(20, 10))
        ax = sns.lineplot(x=x,y=y,data=df,linewidth=7.0,color=color)
        # これは、x軸（時系列）の単位が省略されないようにする設定
        # 何もしないと、2000,2005,2010のように一年分がとばされてしまいます。
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1)) 
        # 以下は、グラフ表示と表示終了
        plt.show()
        plt.close()
        
    # 折れ線グラフ 二系列　主に輸出入　比較につかいます。
    # hue は項目（輸出、輸入）
    # 輸出がb ( blue )、輸入がr ( red )
    # 指定例 ut.g2(df,'ym','Value','exp_imp')
    def g2(self,df,x,y,hue,palette={1: "b", 2: "r"}):
        plt.figure(figsize=(20, 10))
        ax  = sns.lineplot(x=x,y=y,hue=hue,linewidth = 7.0,
             palette=palette,
             data=df)
        # 凡例の位置　２は左上
        ax.legend_._loc = 2
        # 目盛り（年や、年月を省略しない設定）
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        plt.show()
        plt.close()
        
    # 複数系列の折れ線グラフ 
    def gx(self,df,x,y,hue,palette={}):
        plt.figure(figsize=(20, 10))
        if palette == {}:
            ax  = sns.lineplot(x=x,y=y,hue=hue,linewidth = 7.0,data=df)
        else:
            ax  = sns.lineplot(x=x,y=y,hue=hue,linewidth = 7.0,palette=palette,data=df)
        # 凡例の位置　２は左上
        ax.legend_._loc = 2
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))    
        
    def bar(self,df,y,x,prefix='',color='b'):
        # 色見本
        #https://matplotlib.org/examples/color/named_colors.html
        # 色の意味　合計: gold     輸出: b (blue) 輸入: r ( red )　をつかっています。
        if len(prefix) > 0:
            df[y] = df[y].map(lambda x: 'hs' + str(x))
        ax = sns.barplot(y=y, x=x, data=df,color=color)
        plt.show()
        plt.close()


    # 輸出入コードのurl を表示する
    # 2種類だします。
    def hs_url(self,hs_code,exp_imp=2,yyyy_mm='2020_4'):
        # 最新貿易統計
        # 輸出 https://www.customs.go.jp/yusyutu/index.htm
        # 輸入 https://www.customs.go.jp/tariff/index.htm
        hs = hs_code[0:2]
      
        if exp_imp == 1:
            ex = 'yusyutu'
        else:
            ex = 'tariff'
        tmpl = 'https://www.customs.go.jp/{ex}/{yyyy_mm}/data/print_j_{hs}.htm'
        print(tmpl.format(ex=ex,yyyy_mm=yyyy_mm,hs=hs))
        
        # HS 4桁早見表（最新でないことがあります。）
        # 
        tmpl = 'https://www.toishi.info/hscode/{ths}/{ths4}.html'

        ths = str(int(hs))
        ths4 = hs_code[0:4]
        print(tmpl.format(ths=ths,ths4=ths4))
                

    # db との接続 conn は、global 変数として使う 
    def hs_table_create(self,hs_code,tables=['y_2019','year_from_1997','ym_2018_2020']):
        
        if len(hs_code) not in (2,4,6,9):
            print(hs_code + ': 桁数がおかしいです。')
            return
        
        hs = 'hs' + str(len(hs_code))
        
        sql = """
        create table hs{hs_code}_{table}
        as select * from {table}
        where {hs} = '{hs_code}'
        """[1:-1]
        
        for table in tables:
            tg = 'drop table hs{hs_code}_{table}'.format(hs_code=hs_code,table=table)
            print(tg)
            try:
                cursor.execute(tg)
            except:
                pass
            cursor.execute(sql.format(hs=hs,hs_code=hs_code,table=table))

        conn.commit()
        
    def hs_name_get(self,hs_code):
        hs = len(hs_code)
        if hs not in (2,4,6,9):
            print('HS コードの長さがまちがっています。 ' + str(hs))
        hs = str(hs)
        print(hs_code)
        text = 'hs' + hs + '_eng_df.query(' +"'"+ 'hs' + hs + '=="' + hs_code + '"' + "')"
        df = eval(text)
        print(df['hs' + hs + '_name'].values[0])
        text = 'hs' + hs + '_jpn_df.query(' +"'"+ 'hs' + hs + '=="' + hs_code + '"' + "')"
        df = eval(text)
        print(df['hs' + hs + '_name'].values[0])

            
        

    #  国コード(複数) のデータを抽出　国コードは、文字列のはずだが、ときどきなる整数になるので注意
    def countries_table_create(self,countries=['105','304','103','106','601'],tables=['y_2019','year_from_1997','ym_2018_2020']):
        clist = "('" + "','".join(countries) + "')" 
        sql = """
        create table countries_{table}
        as select * from {table}
        where Country in {clist}
        """[1:-1]
        
        for table in tables:
            tg = 'drop table countries_{table}'.format(table=table)
            print(tg)
            try:
                cursor.execute(tg)
            except:
                pass
            cursor.execute(sql.format(clist=clist,table=table))

        conn.commit()
        
    # 国別折れ線グラフのときに、国与える色です。
    def national_colors(self):
        return ({'105': ['中国', 'gold'],
        '304': ['アメリカ', 'red'],
         '103': ['韓国', 'blue'],
         '106:': ['台湾', 'cyan'],
         '601:': ['オーストラリア', 'green'],
         '111:': ['タイ', 'violet'],
         '213:': ['ドイツ', 'lightgrey'],
         '110:': ['ベトナム', 'crimson'],
         '108:': ['香港', 'orangered'],
         '112:': ['シンガポール', 'aqua'],
         '147:': ['アラブ首長国連邦', 'black'],
         '137:': ['サウジ', 'darkgreen'],
         '118:': ['インドネシア', 'darkorange'],
         '113:': ['マレーシア', 'yellow'],
         '205:': ['イギリス', 'darkblue'],
         '224:': ['ロシア', 'pink'],
         '117:': ['フィリピン', 'olive'],
         '302:': ['カナダ', 'salmon'],
         '210:': ['フランス', 'indigo'],
         '305:': ['メキシコ', 'greenyellow']})
    
    # 虹の7色を割り当てる
    def rank_color(self,xlist):
        clist = ['red','ornage','yellow','green','blue','indigo','violet']
        palette = {xlist[i]:clist[i] for i in range(len(xlist))}
        return(palette)


ut = util()
# sql テンプレート hs2 コード指定　輸出のみ  
# 一系統のグラフサンプル

sql_hs2 = """
select Year,sum(Value) as Value 
from year_from_1997
where 
hs2 = '{hs2}' and exp_imp=1
group by Year
"""

hs2 = '87'  #自動車　リーマンショックから立ち直っていませんね。
ut.g1(ut.sql(sql_hs2.format(hs2=hs2)),'Year','Value')
# sql テンプレート 国 country コード指定　輸出入用
sql_country = """
select Year ,exp_imp,sum(Value) as Value 
from year_from_1997
where 
Country = '{country}'
group by Year,exp_imp
"""
country = '105' # 中国の輸出入の年別のグラフ　赤が輸入、青が輸出　リーマンショックで、輸出入金額がほぼ同じになっています。
ut.g2(ut.sql(sql_country.format(country = country )),'Year','Value','exp_imp')
country = '304' # アメリカ　赤が輸入、青が輸出　日本からの輸出超過が続いています。
ut.g2(ut.sql(sql_country.format(country = country )),'Year','Value','exp_imp')
# 複数の系統のグラフの例
# 中古車の輸出
ulist= '("870321915","870321925","870322910","870323915","870323925","870324910","870331100","870332915","870332925","870333910","870390100")'

sql = """
select Year,sum(Value) as Value from year_from_1997
where exp_imp = 1 and
hs9 in {ulist}
group by Year
"""[1:-1]
sql = sql.format(ulist=ulist)
ut.g1(ut.sql(sql),'Year','Value')

sql = """
select ym,sum(Value) as Value from ym_2018_2020
where exp_imp = 1 and
hs9 in {ulist}
group by ym
"""[1:-1]
sql = sql.format(ulist=ulist)
ut.g1(ut.sql(sql),'ym','Value')
sql = """
select y.Country,Country_name,sum(Value) as Value 
from y_2019 y , country_eng c
where exp_imp = 1 and
hs9 in {ulist} and
y.Country = c.Country
group by y.Country
order by Value desc
"""[1:-1]
sql = sql.format(ulist=ulist)
df = ut.sql(sql)


"""
0	113	56599857
1	541	41427206
2	224	38296958
3	606	28359053
4	147	27005315
"""

df.head()
palette={"Malaysia": "black", "Kenya": "brown",
         "Russia":"gold","New_Zealand":"yellow","United_Arab_Emirates":"blue"}
clist = "('113', '541', '224', '606', '147')"
sql = """
select Year,y.Country,Country_name,sum(Value) as Value 
from year_from_1997 y ,country_eng c
where exp_imp = 1 and
hs9 in {ulist} and 
y.Country in {clist} and
y.Country = c.Country
group by Year,y.Country
order by Year
"""[1:-1]
sql = sql.format(ulist=ulist,clist=clist)
df = ut.sql(sql)
#ut.gx(df,'Year','Value')
# Seaborn のhueに数値データに変換可能なものをいれるとAttributeErrorになるので、Country_name にする　バグとしいいようがない
ut.gx(df,'Year','Value','Country_name',palette)
clist = "('113', '541', '224', '606', '147')"
sql = """
select ym,y.Country,Country_name,sum(Value) as Value 
from ym_2018_2020 y ,country_eng c
where exp_imp = 1 and
hs9 in {ulist} and 
y.Country in {clist} and
y.Country = c.Country
group by ym,y.Country
order by ym
"""[1:-1]
sql = sql.format(ulist=ulist,clist=clist)
df = ut.sql(sql)
ut.gx(df,'ym','Value','Country_name',palette)