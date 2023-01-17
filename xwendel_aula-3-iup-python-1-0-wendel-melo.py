nome = "jessica"      #string é um container de caracteres
#Outro exemplo de container é a tupla, que é uma espécie de array (vetor) imutável

dados = (5, 6, 1.6810, False, "camila")        #note que a tupla pode misturar objetos de diferentes tipos
#quando o container possui a ideia de ordem, podemos acessar seus elementos através de um índice, 

#que representa a ordem em que os elementos apaarecem no container.

#começamos a numerear os índices dos elementos a partir do zero. Assim, o índice 0 se refere ao primeiro elemento do container



nome[0]     #acessa o primeiro elemento do container
#como a contagem dos índices começa dos 0, para acessar o quinto elemento de um container ordenado, usamos o índice 4

dados[4]
#python também trabalha com a idéia do índice negativo para acessar os elementos de um container.

#Todavia, os índices negativos começam a contar a partir do final:



#ex:



t = "universo"



#cada elemento do container pode ser acessado através de dois índices:

#    0  1  2  3  4  5  6  7  

#    u  n  i  v  e  r  s  o

#   -8 -7 -6 -5 -4 -3 -2 -1



#assim acessar o índice -1 é o mesmo que acessar o último elemento de um container ordenado:

t[-1]
#por sua vez, acessar o índice -2 é o mesmo que acessar o penúltimo elemento:

t[-2]
#E, assim, sucessivamente...

dados[-3]
#acesso a fatia do conteiner apontado por t que vai do índice 0 até o índice 2

t[0:3]        

#precisamos indicar que o fim da fatia é 4 porque, por definição, o índice apontado como fim não é incluido na fatia resultante
#pega a fatia do índice 3 até o índice 7

t[3:8]    #note que precisamos indicar o fim como 8 porque o índice final não é incluído no resultado
#quando omitimos o fim, python asusme que ele deve pegar até o final do container

t[5: ]
#quando o omitimos o início, python assume que a fatia deve começar do índice 0

t[:6]
#podemos colocar mais um argumento para a fatia, que é o passo. Por exemplo, para pegar os caracteres da string de dois em dois:

t[0:8:2]
#podemos obter fatias percorrendo ao reverso ao reverso usando passo negativo

t[ 6: 3:-1]  
#o operador len pode ser aplicado para determinar o tamanho do container, isto é, a quantidade de elementos

#presentes no mesmo. Assim, o tamanho de uma string é o número de caracteres

len(nome)      #a string "jessica" tem 7 caracteres
len(dados)
#é válido mencionar que ponto, vírgula pontuações em geral, enter, tabulação e até espaço em branco 

#também são contados como caracteres:

texto = "oi, Ana!"

len(texto)
#podemos comparar strings usando o operador ==

nome == "carol"
nome == "jessica"
#caracters minúsculos são considerados diferentes de caracteres maiúsculos

"jessica" == "JEssica"
#pode-se comparar tuplas com ==

(1, 2) == (3, 4, 5)
#o operador in testa participação de um objeto como membro de container

"j" in nome 
"w" in nome
7 in dados
#o operador not in retorna o resultado oposto ao do operador in, isto é, testa a não participação como membro de container

7 not in dados
#Exemplo: imprimindo os primeiros 10 números naturais não nulos



n = 1

while n <= 10:

    print( n )

    n = n + 1
#exemplo de laço infinito



#n = 1

#while n <= 10:

    #print( n )



#o laço acima imprimirá 1 indefinidamente, pois a condição de repetição nunca resultará em falso.
#exemplo: programa que lê um número n do teclado e imprime os primeiros n números naturais não nulos na tela



n = input("Entre com a quantidade de números a serem impressos: ")

n = int(n)



contador = 1



while contador <= n:

    print(contador)

    contador = contador + 1



print("Tenha um bom dia!")
primos = (2, 3, 5, 7, 11)

for n in primos: #a variável n percorrerá os elementos da sequência apontada por primos, um a um, na sua respectiva ordem.

    print(n)
nome = "jessica"

for letra in nome: #a variável letra percorrerá todos os elementos da string apontada por nome, um a um, na sua respectiva ordem

    print(letra)
#gera uma sequência que vai de 1 até 9 com passo 2



seq = range(1, 9, 2)

tuple(seq)
#Usando a função range, podemos fazer o laço que imprime os 10 primeiros números naturais não nulos com for



for k in range(1, 11):  #quando [passo] é omitido, o valor 1 é asusmido como passo

    print(k)
#com a função range, podemos percorrer os índices de uma sequência. 

#Para isso, usamos o operador len, que retorna o tamanho de um objeto sequencial qualquer



nome = "jessica"

for i in range(0, len(nome)): # i andará nos índices da string nome. Repare a diferença com o outro exemplo onde percorremos diretamente os caracteres de nome

    print( nome[i] );
#Exemplo: programa que calcula a media de n números lidos do usuário



n = int( input("Entre com a quantidade de numeros da soma: ") )



soma = 0.0



for k in range(1, n+1):    #precisamos colocar o fim como n+1 porque o fim não entra na sequência

    

    numero = float( input("Entre com o numero %s: "%(k) ) )

    

    soma += numero          #equivalente a soma = soma + numero



media = soma/n



print("Media dos numeros: ", media)



#Exemplo: lendo um texto do usuário e contando o número de vogais 



texto = input("Entre com um texto: ")



nvogais = 0            #contador de vogais



for letra in texto:

    if letra in ('a', 'e', 'i', 'o', 'u'):

        nvogais += 1    #note que essa linha está dentro de if, que está dentro de um for. Por isso, precisamos indentar com duas tabulações



print("Numero de vogais: ", nvogais)