# bloody dependencies

from collections import Counter

from math import log2

import matplotlib.pyplot as plt
# load data



file_path = "/kaggle/input/beale-ciphers/beale.txt"

with open(file_path) as f:

    beale_cipher = [[int(x) for x in line.split(",")] for line in f.readlines()]



# to make readable

readable = lambda lst: " ".join([str(x) for x in lst])



# show data

print("\nCipher #1:")

print(readable(beale_cipher[0]))

print("\nCipher #2:")

print(readable(beale_cipher[1]))

print("\nCipher #3:")

print(readable(beale_cipher[2]))
# load cipherkey

file_path = "/kaggle/input/beale-ciphers/declaration.txt"

with open(file_path) as f:

    declaration = f.readlines()

    

# preprocess cipherkey

declaration = " ".join(declaration)  # one big block

declaration = declaration.replace("'", "")  # remove apostrophes

declaration = declaration.replace("&", "and")  # replace ampersand

declaration = declaration.lower()  # lowercase

declaration = "".join([l if l.isalpha() else " " for l in declaration])  # remove non-alpha characters

declaration = declaration.split()  # tokenize

declaration = [word for word in declaration if len(word) > 0]

declaration = ["[0]"] + declaration



# manual fixes (REF: https://en.wikipedia.org/wiki/Beale_ciphers#Deciphered_message)

remove_indices = [245, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 653, 818]

declaration = [word for idx, word in enumerate(declaration) if idx not in remove_indices]

declaration = declaration[:155] + ["a"] + declaration[155:]

declaration[811]  = "yfundamentally"

declaration[1005] = "xhave"



# show cipherkey

print("\nDeclaration of Independence:")

print(readable(declaration))
# how to decode

def decode(ciphertext: list, cipherkey: list) -> str:

    return "".join([cipherkey[x][0] if x < len(cipherkey) else "?" for x in ciphertext])



# test

print("Cipher #2:")

plaintext = decode(beale_cipher[1], declaration)

print("\n" + plaintext)
# does the declaration decode the other two ciphers?

print("\nCipher #1:")

print(decode(beale_cipher[0], declaration))

print("\nCipher #3:")

print(decode(beale_cipher[2], declaration))
# shannon entropy

def entropy(ciphertext: list) -> float:

    """Measures the Shannon entropy content of a ciphertext, H(x) = sum(-log_2(p(x)))"""

    counts = Counter(ciphertext)

    total = sum(counts.values())

    frequencies = {key: val/total for key,val in counts.items()}

    return - sum([log2(x) for x in frequencies.values()])



# accumulating entropy of each cipher

entropy_data = lambda lst: [entropy(lst[:x]) for x in range(len(lst))]

entropies = [entropy_data(ciphertext) for ciphertext in beale_cipher]



# visualize it

plt.figure(figsize=(20,10))

for i, x in enumerate(entropies):

    label = round(max(x))

    print("Entropy of cipher #"+str(i+1)+":", label)

    plt.plot(x, label = "Cipher #"+str(i+1))

    plt.scatter([len(x)], [max(x)], label = label)

plt.xlabel("Characters")

plt.ylabel("Shannon Entropy (bits)")

plt.title("Entropy Content of the Ciphers")

plt.legend()

plt.show()
# n_types = f( m_tokens )

lexicon_size = lambda x: len(set(x))

lexicon_size_data = lambda lst: [lexicon_size(lst[:x]) for x in range(len(lst))]



# accumulating density of each cipher

lexicon_sizes = [lexicon_size_data(ciphertext) for ciphertext in beale_cipher]

    

# visualize it

plt.figure(figsize=(20,10))

for i, x in enumerate(lexicon_sizes):

    label = int(100 * max(x) / len(x))

    print("Lexical Density of cipher #"+str(i+1)+":", label)

    plt.plot(x, label = "Cipher #"+str(i+1))

    plt.scatter([len(x)], [max(x)], label = label)

plt.xlabel("Characters (tokens)")

plt.ylabel("Unique Characters (types)")

plt.title("Lexical Density (type-token relation)")

plt.legend()

plt.plot()