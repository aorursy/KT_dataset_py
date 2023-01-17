#exemplo: programa que lê um numero n e informa se n é primo

#um número é primo se só é divísível por 1 or ele mesmo.

#neste programa, usaremos o operador %, que fornece o resto da divisão entre números inteiros.

# 15 % 4 calcula o resto da divisão de 15 por 4, que é 3.



n = int( input("Entre com o numero: ") )



ndivisores = 0



for k in range(2, n):

    if n % k == 0:  #se o resto da divisão de n por k ézero, significa que k é divisor de n

        ndivisores += 1



if ndivisores == 0:

    print("%s é numero primo!"%(n) )

else:

    print("%s não é número primo!"%(n) )

#exemplo: programa que lê um numero n e informa se n é primo

#um número é primo se só é divísível por 1 or ele mesmo.

#nessa versão, vamos usar a cláusula break, que interrompe a execução de um laço



n = int( input("Entre com o numero: ") )



ndivisores = 0



for k in range(2, n):

    if n % k == 0:  #se o resto da divisão de n por k ézero, significa que k é divisor de n

        ndivisores += 1

        break       #força o encerramento do laço



if ndivisores == 0:

    print("%s é numero primo!"%(n) )

else:

    print("%s não é número primo!"%(n) )
primos = (2,3,5,7)

for i in primos:

    if i == 5:

        continue

    print(i)



#note que no código acima, o valor 5 está na tupla primos e não é impresso, pois quando i vale 5, é executada a cláusula continue antes da impressão de i

#a cláusua contonue força o programa a pular para a próxima iteração do for sem a execução do restante do bloco de repetição na iteração corrente.
#O operador + realiza de concatenação de containers ordenados

nome = "wendel" + "melo"

print(nome)
#O operador * realiza repetição de containers ordenados

texto =  3*"ai"    #repete a string 3 vezes

print(texto)
#o operador % permite a inclusão de valores externos a uma string. O código %s permite a inclusão de objetos de qualquer tipo à string.

nome = "Jessica"

idade = 29

texto = "%s tem %s anos"%(nome, idade)    #os valores apontados por nome e idade serão inseridos nos lugares dos %s

print(texto)
#o construtor str permite converter obtejos para string

aurea = 1.6180

t = str(aurea)

t
#é possível usar operadores de comparação para comparar strings

nome == "lueli"  #retorna True se a string apontada por nome for igual a string "lueli"
#também é possível usar os operadores <, <=, >, >= com strings. Nesse caso, a ordem dos caracteres na codificação será utilizada.

#na codificação ASCI, caracteres maiúsculos vem antes dos minúsculos

"jessica" > "carol"
help(str)
nome = "ana"

nome.upper()    # chama o método upper, que retorna uma cópia da string com caracteres maiúsculos
nome   #nome continua sendo uma string com caracteres minúsculos
texto = "KAT !@#$"

texto.isupper()
dado = "cAROL"

dado.isupper()
nome.islower()
nome.isdigit()
t = "1991"

t.isdigit()
print(texto)

texto.isalpha()
nome.isalpha()
print(texto)

texto.lower()
nome.upper()
valor = "universo"

valor.find("verso")  #retorna o índice onde a substring "verso" aparece na string apontada por valor.
valor.find("lua")
frase = "Meu $$$ texto que $$$ escrevi"

lista = frase.split("$$$")    #separa a string usando a substring "$$$"

print(lista)
sentenca = "Vovô viu a uva!"

palavras = sentenca.split(" ")   #separa a string usando o caractere espaço em branco

print(palavras)
sentenca.split()   #se o separador não for informado, separa nos espaços em branco
trechos = ("jessica", "é", "magnífica")

tudo = " ".join(trechos)    #concatenará todas as strings em trechos usando " " como separador

print(tudo)
h = "Quem casa quer casa"

g = h.replace("casa", "fala")    #sibstitui aparições de "casa" por "fala"

print(g)
h.replace("casa", "fala", 1)
#para remover a aparição de uma substring, basta usar a string vazia como destino

h.replace("casa", "")
h.count("casa")
#exemplo: código que conta vogais, consoantes e caracteres não alfabéticos de uma string. Assumimos que a string não é acentuada.



texto = input("Entre com o texto: ")

texto = texto.lower()   #assim, só precisamos nos preocupar com carateres minúsculos



nvogais = 0

nconsoantes = 0

noutros = 0



for c in texto:

    if c.isalpha():

        if c in "aeiou":

            nvogais += 1

        else:

            nconsoantes += 1

    else:

        noutros += 1



print("numero de vogais: ", nvogais)

print("numero de consoantes: ", nconsoantes)

print("outros caracteres: ", noutros)