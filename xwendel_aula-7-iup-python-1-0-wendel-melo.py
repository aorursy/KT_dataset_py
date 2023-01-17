#definimos uma listas com numeros

numeros = [1,2,3,4,5,6]
#podemos construir uma lista com os quadrados dos numeros, aplicando uma operação a cada membro de uma lista

quadrados = [ k**2 for k in numeros ]

print(quadrados)
#podemos aplicar um filtro para só obter os números pares da lista apontada por numeros

#(números ímpares tem 1 como o resto da divisão por 2)

impares = [ i for i in numeros if i % 2 == 1 ]

print(impares)
#podemos combinar a aplicação de operações com filtragem. Por exemplo, o dobro dos números pares

dobroPares = [ 2*w for w in numeros if w % 2 == 0 ]

print(dobroPares)
#exemplo: função que recebe um número n e retorna uma lista com seus divisores

def divisores(n):

    return [x for x in range(1,n+1) if x % n == 0]     #dá para melhorar isso aqui...
#exemplo: programa que lê números inteiros do teclado e os exibe em ordem decrescente



n = int( input("Entre com a quantidade de numeros: ") )



numeros = [0]*n   #Gera uma lista com n objetos 0



for i in range(0, n):

    numeros[i] = int( input("Entre com numero %s: "%(i+1) ) )



numeros.sort(reverse = True) #ordena em modo decrescente



#imprimindo

for valor in numeros:

    print( valor, " ", end="")  #o argumento end="" é para que ele não pule linha (o default é end="\n")

#fazendo o mesmo exemplo com compreensão de lista



n = int( input("Entre com a quantidade de numeros: ") )



numeros = [ int( input("Entre com o numero %s: "%i ) ) for i in range(0, n) ]



numeros.sort(reverse = True)



[ print(valor) for valor in numeros ]  #aqui geramos uma lista, mas não fazemos nada com ela



conjunto = {23, "jessica", 3.1,  (1,2,3) }  #conjunto com quatro elementos
c = {1, 3, 3, 3, 3, 3, 3}   #repetições são ignoradas e este conjunto só terá dois elementos.

print(c)
conjunto = {23, "jessica", 3.1,  (1,2,3) }  #conjunto com quatro elementos

print(conjunto)
#assim, não faz sentido usar índices com conjuntos:

print( conjunto[0] )   #erro!
teste = { 4, [1,2] } #erro! Conjuntos não podem abrigar objetos mutáveis
#Exemplo: código que sorteia 6 dezenas distintas entre 1 e 60 para megasena



import random



ndezenas = 6



sorteio = set()  #gera um conjunto vazio



while len(sorteio) < ndezenas:

    aleatorio = random.randint(1,60)

    #print("aleatorio: ", aleatorio)

    sorteio.add( aleatorio )



print("sorteio: ", sorteio)