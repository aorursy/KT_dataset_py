print("Hello World!")
print("Welcome to Python!")
print("The product of 7 and 8 is", 7 * 8)
# Your code here:

a = 5
print(a)
a + 7
type(a)
type(a)
x = 5.12312983

type(x)
value = True # Camelcase

type(value)
# Your code here
print("This will be printed")

# print("This will not be printed")

print("Another example") # of a comment 
a = 5

b = 2
a + b
a - b
a * b
a**b
a / b
a // b
a % b
# Your code here
# Your code here
# Your code here
10 == 10    # Note: '==', not '='
10 > 5
5 > 10
a != b
b < 5
b <= 5
a = 10

b = 2



if a > b:

    print('Yes')
if a < b:

    print('Yes')
c = 20

if c > a and c > b:

    print("C is big!")
d = 7

if d > a and d > b:

    print("D is big")

if d > a or d > b:

    print("D is bigger than a or b")
import random

grade = random.randint(0, 10)



# Your code goes here
# Your code here
n = random.randint(0, 20)



if n % 2 == 0:

    print(n, 'is an even number')

else:

    print(n, 'is not an even number')

    

print('End of program')
kmh = random.randint(80, 150)



# Your code goes here80
trials = 1

random_num = random.randint(1, 100)



while random_num <= 90:        # condition

    print(random_num)           # indented block

    random_num = randint(1,100) # indented block

    trials = trials + 1

print ('Found a number greater than 90 (', random_num, ') after ', trials, 'trials.')
# 11 represents J, 12 represents Q, 13 represents K, 14 represents A

card_one = random.randint(0, 14)

card_two = random.randint(0, 14)



# Your code goes in the brackets of the parentheses.

while():

    card_one = random.randint(0, 14)

    card_two = random.randint(0, 14)

    

print("Yes! got aces", card_one, card_two)
x = "Jupyter"

y = 'I love Python'

print(x)

print(y)
type(x)
cheeseshop_dialog ="""Customer: 'Not much of a cheese shop really, is it?'

Shopkeeper: 'Finest in the district, sir.'

Customer: 'And what leads you to that conclusion?'

Shopkeeper: 'Well, it's so clean.'

Customer: 'It's certainly uncontaminated by cheese.'

"""

print(cheeseshop_dialog)

type(cheeseshop_dialog)
print(x + "2019")
n = len(dna)

print("The length of the DNA variable is", n)



dna = dna + "AGCTGA"

print("Now it is", len(dna))
x = 10

x *= 7

print(x)
text = "A musician wakes from a terrible nightmare."
print(text[0])

print(text[5])
print(text[-1])

print(text[-4])
# get the 1st and 6th letters

print(text[0])

print(text[5])
# get the last letter

print(text[-1])

# get 5th letter from the end

print(text[-5])
# get the 3rd to 8th letters

print(text[2:8])
# get the first 5 letters

print(text[0:5])

# or simply:

print(text[:5])



# get 3rd to last letters:

print(text[3:])



# last 3 letters

print(text[-3:])
seq = "CAAGTAATGGCAGCCATTAA"



# Your code goes here
# a list of strings

apes = ["Human", "Gorilla", "Chimpanzee"]

print(apes)
# a list of numbers

nums = [7, 13, 2, 400]

print(nums)
# a mixed list

mixed = [12, 'Mouse', True]

print(mixed)
print(apes[0])

print(apes[-1])
new_apes = apes.copy() # make a copy of the apes list

new_apes[2] = 'Bonobo'

print(new_apes)
apes.append("Macaco")

print(apes)
apes.insert(2, "Kofiko")

print(apes)
apes.remove("Human")

print(apes)
print(apes.pop(3))

print(apes)
i = apes.index('Kofiko')

print(i, apes[i])
# get the first 3 objects

shopping_list = ['tomatoes', 'sugar', 'tea', 'cereal', 'bananas', 'rice']

print(shopping_list[:3])
birds = ['Owl', 'Sparrow', 'Parrot']

snakes = ['Viper', 'Mamba', 'Python']

# Your code goes here
for ape in apes:

    print(ape, "is an ape")
dna = "ACGTAAAACGTACAAGA"

count_a = 0



for letter in dna:

    if letter == "G":

        count_a = count_a + 1

        

print("# of G's", count_a)
sentence = 'Please count the number of whitespace in this sentence'

count_space = 0



# Your code here



print("Number of spaces in our sentence is :", count_space)
for i in range(10): 

    print(i)
n = 97 # try other numbers











capitals = {

    'Israel': 'Jerusalem', 

    'France': 'Paris', 

    'England': 'London', 

    'USA': 'Washington DC',

    'Korea' : 'Seoul',

    'Spain' : 'Madrid'

}
print(capitals['Korea'])

print(capitals['Spain'])
capitals['Spain'] = 'Barcelona'

print(capitals['Spain'])
capitals['Germany'] = 'Berlin'

print(capitals['Germany'])
for country in capitals:

    print(capitals[country], "is the capital of", country)
'Israel' in capitals
'Japan' in capitals
# Your code goes here
def multiply(x, y):

    z = x * y

    return z
x = 3

y = multiply(x, 2)

print(y)
z = multiply(7, 5)

print(z)
secret = """Mq osakk le eh ue usq qhp, mq osakk xzlsu zh Xcahgq,

mq osakk xzlsu eh usq oqao ahp egqaho,

mq osakk xzlsu mzus lcemzhl gehxzpqhgq ahp lcemzhl oucqhlus zh usq azc, mq osakk pqxqhp ebc Zokahp, msauqjqc usq geou dat rq,

mq osakk xzlsu eh usq rqagsqo,

mq osakk xzlsu eh usq kahpzhl lcebhpo,

mq osakk xzlsu zh usq xzqkpo ahp zh usq oucqquo,

mq osakk xzlsu zh usq szkko;

mq osakk hqjqc obccqhpqc, ahp qjqh zx, mszgs Z pe heu xec a dedqhu rqkzqjq, uszo Zokahp ec a kaclq iacu ex zu mqcq obrfblauqp ahp ouacjzhl, usqh ebc Qdizcq rqtehp usq oqao, acdqp ahp lbacpqp rt usq Rczuzos Xkqqu, mebkp gacct eh usq oucbllkq, bhuzk, zh Lep’o leep uzdq, usq Hqm Meckp, mzus akk zuo iemqc ahp dzlsu, ouqio xecus ue usq cqogbq ahp usq kzrqcauzeh ex usq ekp."""



# A dictionary to crack the code.

code = {'w': 'x', 'L': 'G', 'c': 'r', 'x': 'f', 'G': 'C', 'E': 'O', 'h': 'n', 'O': 'S', 'y': 'q', 'R': 'B', 'd': 'm', 'f': 'j', 'i': 'p', 'o': 's', 'g': 'c', 'a': 'a', 'u': 't', 'k': 'l', 'q': 'e', 'r': 'b', 'V': 'Z', 'X': 'F', 'N': 'K', 'B': 'U', 'T': 'Y', 'M': 'W', 'U': 'T', 'm': 'w', 'C': 'R', 'J': 'V', 't': 'y', 'S': 'H', 'v': 'z', 'e': 'o', 'D': 'M', 'p': 'd', 'K': 'L', 'A': 'A', 'P': 'D', 'l': 'g', 's': 'h', 'W': 'X', 'H': 'N', 'j': 'v', 'z': 'i', 'I': 'P', 'b': 'u', 'Z': 'I', 'F': 'J', 'Y': 'Q', 'Q': 'E', 'n': 'k'}



def decode (secret, code):

    decipher = ""

    

    # Your code goes here

    

    print (decipher)



decode(secret, code)
