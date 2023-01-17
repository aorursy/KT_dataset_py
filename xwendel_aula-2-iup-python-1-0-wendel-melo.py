#A função print realiza impressão (exibição) de dados na tela do terminal.

ano = 2020

print(ano)
#Pode-se imprimir diversos de uma vez, separdo-se os objetos por vírgula

print("Ano atual:", ano)
#Obs: A função print por padrão separa os objetos com um espaço em branco e pula uma linha.

#Para não separar os objetos com espaço, basta fazer:

print("Ano atual:", ano, sep="")



#Para não pular linha, basta fazer:

print("Ano atual:", ano, end="")

print(". Não pulou linha")
#Isto é um comentário: Tudo o que for escrito em uma linha depois de um caracter hashtag é considerado texto direcionado a seres humanos (comentário)

#assim, o interpretador Python ignorará esses comentários



#Primeiro programa em python: imprime "Alo mundo"

#autor: Wendel Melo



print("Alo mundo!")
#segundo programa em Python: lê nome do usuário e o saúda



nome = input("Digite o seu nome: ")

print("Ola", nome, "! ")

print("Tenha um bom dia!")
#programa que lê raio de um circulo e calcula diametro, perimetro e área



raio = input("Entre com o raio do circulo: ")

raio = float(raio)  #input sempre retorna string. Como desejamos fazer contas com o dado, precisamos fazer a conversão para um tipo numérico



pi = 3.1415



diametro = 2*raio

perimetro = 2*pi*raio

area = pi*(raio**2) #o operador ** realiza potenciação



print("diametro: ", diametro)

print("perimetro: ", perimetro)

print("area: ", area)



print("Tenha um bom dia!")

#programa que lê uma nota e informa se a nota é vermelha



nota = input("Entre com uma nota: ")

nota = float(nota)



if nota < 5.0:

    print("Nota vermelha")     #note o TAB no início linha. A indentação é o único modo de dizer ao Python que um bloco de instruções está subordinado a um if

    print("Você precisa estudar mais")

    print("Saia do PC e vá estudar")



print("Tenha um bom dia!")

#programa que lê uma nota e informa se a nota é vermelha



nota = input("Entre com uma nota: ")

nota = float(nota)



if nota < 5.0:

    print("Nota vermelha")     #note o TAB no início linha. A indentação é o único modo de dizer ao Python que um bloco de instruções está subordinado a um if

    print("Você precisa estudar mais")

    print("Saia do PC e vá estudar")

else:

    print("Nota azul")

    print("Você está de parabéns")



print("Tenha um bom dia!")
#programa que lê uma nota e informa se a nota é vermelha



nota = input("Entre com uma nota: ")

nota = float(nota)



if nota >= 0 and nota < 5.0 :

    print("Nota vermelha")     #note o TAB no início linha. A indentação é o único modo de dizer ao Python que um bloco de instruções está subordinado a um if

    print("Você precisa estudar mais")

    print("Saia do PC e vá estudar")

    

elif 5 <= nota < 8:   #python permite testar intervalos assim. NÃO FAÇA ISSO EM LINGUAGENS COMO C OU JAVA

    print("Nota azul")

    print("Você está de parabéns")

    

elif 8 <= nota <= 10:

    print("Super nota")

    print("Você é muito bom")

    print("Continue assim")

    

else:

    print("Nota inválida")



print("Tenha um bom dia!")
altura = input("Entre com a sua altura: ")

altura = float(altura)



if altura >= 1.80:

    print("você é alto")

    

    if altura >= 2.00:

        print("Está quente ai em cima?! ")



print("Tenha um bom dia!")