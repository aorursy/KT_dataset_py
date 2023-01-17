# Testador de HTML: https://www.w3schools.com/html/tryit.asp?filename=tryhtml_intro

'''



<!DOCTYPE html>

<html lang="pt">

    <head>

        <title> Minha Pagina </title>

    </head>

    <body>

        <h1> Esta é a minha página HTML</h1>

        <p> Eu escrevo o que quiser aqui... </p>

    </body>

</html>



'''
# Importa as bibliotecas

import urllib
# Abre uma conexao com o site e lê o conteúdo

site = urllib.request.urlopen('https://www.comprasgovernamentais.gov.br/').read()

site
conexao2 = urllib.request.urlopen('https://desciclopedia.org/wiki/P%C3%A1gina_principal').read()

conexao2

# 403 Forbidden
site='https://desciclopedia.org/wiki/Categoria:Melhores_artigos'

headers={'User-Agent':'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7',} 



conexao=urllib.request.Request(site,None,headers)

desciclopedia = urllib.request.urlopen(conexao).read()

desciclopedia
# Tratamento de erros com urllib.error.URLError

try:

    urllib.request.urlopen('https://desciclopedia.org/wiki/Categoria:Melhores_artigos').read()

except urllib.error.URLError as e:

    print("Erro... Ih, não conectou")
# Importe a biblioteca urllib.

import urllib



# Conecte ao site https://pt.wikipedia.org/wiki/Caneta, leia o conteúdo e 

conexao = urllib.request.Request('https://pt.wikipedia.org/wiki/Caneta')



# Se o site não exigir dados de conexão, pode executar o read() direto

# grave o conteúdo lido em uma variável chamada wikicaneta. 

wikicaneta = urllib.request.urlopen(conexao).read()



# Imprima o conteúdo de wikicaneta

wikicaneta
import requests

requests.get('https://www.comprasgovernamentais.gov.br')
# Traz o conteúdo

conteudo = requests.get('https://www.comprasgovernamentais.gov.br').text

conteudo
# Tratamento de erros

conexao = requests.get('https://desciclopedia.org/wiki/Categoria:Melhores_artigos')

if conexao:

    print('Sucesso!')

    conteudo = conexao.text

else:

    print('Erro.')
# Importa BeautifulSoup e Urllib

import bs4 as bs

import urllib



# Informa qual página, abre uma conexão e traz o conteúdo HTML

pagina = urllib.request.urlopen('https://pt.wikipedia.org/wiki/Caneta').read()



# Faz o parse

soup = bs.BeautifulSoup(pagina, 'html.parser')
# Imprime a tag <title>

soup.title



# Imprime o valor da tag <title> 

soup.title.string
# Busca por todos os paragrafos: representados pela tag <p> 

soup.findAll('p')

# Quantos parágrafos são?

len(soup.findAll('p'))

# Retorna apenas o primeiro elemento de paragrafo 

soup.findAll('p')[0]
# Busca apenas o conteúdo (sem marcacoes HTML) do primeiro paragrafo <p>

soup.findAll('p')[0].text
# Busca apenas o conteúdo (sem marcacoes HTML) de todos os paragrafos <p>

for paragrafo in soup.findAll('p'):

    print(paragrafo.text)
# Traz todos os textos de todas as tags

soup.get_text()
# Traz todas as URLs citadas na pagina

for url in soup.findAll('a'): print(url.get('href'))
# Traz apenas as URLs <a> da class="mw-jump-link"

for url in soup.findAll('a', class_='mw-jump-link'): print(url.get('href'))
# Traz apenas o texto da tag <a> da classe class="mw-jump-link"

for url in soup.findAll('a', class_='mw-jump-link'): print(url.text)
# Traz os link do google , noticiass, livro, academico

# Como trazer tags que estao dentro de tags?

#   <html>

#      |-- <td>

#          |-- <a> 



# Traz o primeiro nível: tag <td>

td = soup.findAll('td')

td[1].findAll('a')



# Desce mais um nível: tag <a>

a = td[1].findAll('a', class_="external text")



# Traz um elemento

td[1].findAll('a', class_="external text")[3].get('href') 



 # Traz todos os elementos

for x in a: print(x.get('href')) 

    

# Cria uma lista com os elementos

dfelementos = []

for x in a: dfelementos.append(x.get('href')) 

    

# Transforma para dataframe e altera o nome da coluna

import pandas as pd

dfelementos = pd.DataFrame(dfelementos)

dfelementos.columns=['Links de Interesse']



dfelementos
# Importe as bibliotecas necessárias. 



# Faça raspagem do site https://desciclopedia.org/wiki/Piscina_de_1000_litros 



# Obter o texto apenas das legendas das imagens



# Crie um Dataframe com o resultado e Renomeia a Coluna para "Galeria de Fotos"



# Imprima o resultado
# Import

import pandas as pd

import requests

import urllib
# Conecta no Site

#site='https://www.staples.pt/portal-informativo/guia-de-compras/guia-do-comprador-canetas.html'

site='http://www.linhadecodigo.com.br/artigo/3439/introducao-ao-html-usando-tabelas-em-html.aspx'

headers={'User-Agent':'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7',} 

request=urllib.request.Request(site,None,headers)

conexao = urllib.request.urlopen(request).read() 

conexao
# Raspar a tabela com o PANDAS

dfs = pd.read_html(conexao, header=0)

dfs[0]
# Raspa as tabelas da página

dfs = pd.read_html('https://en.wikipedia.org/wiki/Timeline_of_programming_languages', header=0)

dfs[4]
dfs[5].head()
# Concatena as tabelas

df = pd.concat(dfs[4:12])

df.head()
# Salva o dataframe

df.to_csv('raspagemdetabelas.csv')
# Faça a raspagem de todas as tabelas de UF e grave no arquivo estados.csv

tabestados = pd.read_html('https://atendimento.tecnospeed.com.br/hc/pt-br/articles/360021494734-Tabela-de-C%C3%B3digo-de-UF-do-IBGE', header=0)

tabestados
estados = pd.concat(tabestados[:])

estados
estados.shape
estados.to_csv('estados.csv')
# Raspagem de um CEP pela API

df = pd.read_json('https://api.postmon.com.br/v1/cep/20020-010')

df
# Faça a raspagem da página 1 de órgãos SIAPE da API do Portal de Transparencia

dforgaos = pd.read_json('http://www.transparencia.gov.br/api-de-dados/orgaos-siape?pagina=1')

dforgaos
# import requests



## Se identificando apropriadamente...

headers = {

    'User-Agent': 'Seu nome, exemplo.com',

    'From': 'email@exemplo.com'}



#page = requests.get('https://site.com', headers = headers)
# Cria uma lista vazia

orgaosiape = []



# Traz os dados de Orgao SIAPE da página 1 até a página 6

for page_num in range(1, 7):

    url = "http://www.transparencia.gov.br/api-de-dados/orgaos-siape?pagina=" + str(page_num)

    print("Downloading", url)

    response = requests.get(url)

    data = response.json()

    orgaosiape = orgaosiape + data



# Imprime o resultado

#orgaosiape
# Procura por páginas sequenciais, começando da primeira página

response = requests.get("https://swapi.co/api/people/?search=a")

data = response.json()



# Enquanto não está vazio, traz a próxima URL

while data['next'] is not None:

    print("Próxima página encontrada, downloading", data['next'])

    response = requests.get(data['next'])

    data = response.json()
# Extrai as opiniões sobre produtos da Amazon

import requests

from bs4 import BeautifulSoup



def get_reviews(s,url):

    s.headers['User-Agent'] = 'Mozilla/5.0'

    response = s.get(url)

    soup = BeautifulSoup(response.text,"lxml")

    return soup.find_all("div",{"data-hook":"review-collapsed"})



if __name__ == '__main__':

    link = 'https://www.amazon.com.br/Bicicleta-Aro-Rossettis-Marchas-Shimano/dp/B07N466HH5/ref=sr_1_1?qid=1559789367&s=garden&sr=1-1'    

    with requests.Session() as s:

        for review in get_reviews(s,link):

            print(f'{review.text}\n')
# RESPOSTA 3

# Importe a biblioteca urllib. 

import urllib 



# Conecte ao site https://pt.wikipedia.org/wiki/Caneta, 

# leia o conteúdo e grave o conteúdo lido em uma variável chamada wikicaneta.

wikicaneta = urllib.request.urlopen('https://pt.wikipedia.org/wiki/Caneta').read()



#print(wikicaneta)
# RESPOSTA 5

# Importe as bibliotecas necessárias. 

import bs4 as bs

import urllib



# Faça raspagem do site https://desciclopedia.org/wiki/Piscina_de_1000_litros 

site='https://desciclopedia.org/wiki/Piscina_de_1000_litros'

headers={'User-Agent':'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7',} 

conexao=urllib.request.Request(site,None,headers)

conteudohtml = urllib.request.urlopen(conexao).read()

piscina = bs.BeautifulSoup(conteudohtml, 'html.parser')



# Obter o texto apenas das legendas das imagens

# ul = piscina.findAll('ul', class_="gallery mw-gallery-traditional")

# li = ul[0].findAll('li')

# div = li[0].findAll('p')

# for p in div: print(p.text)



# Obter o texto apenas das legendas das imagens

piscinadf = []

for x in piscina.findAll('ul', class_="gallery mw-gallery-traditional"):

    li = x.findAll('li')

    for y in li:

        div = y.findAll('p')

        for p in div: 

            piscinadf.append(p.text)



# Crie um Dataframe com o resultado e Renomeia a Coluna

df = pd.DataFrame(piscinadf)

df.columns=['Galeria de Fotos']



# Imprima o resultado

df
# RESPOSTA 6

ufs = pd.read_html('https://atendimento.tecnospeed.com.br/hc/pt-br/articles/360021494734-Tabela-de-C%C3%B3digo-de-UF-do-IBGE', header=0)

estados = pd.concat(ufs[0:5])

estados.to_csv('estados.csv')
# RESPOSTA 7

# http://www.transparencia.gov.br/api-de-dados/orgaos-siape?pagina=1

orgaosiape1 = pd.read_json('http://www.transparencia.gov.br/api-de-dados/orgaos-siape?pagina=1')