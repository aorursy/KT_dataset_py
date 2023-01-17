mylist = [1, 2, 3, 4, 5, 6]



for x in mylist:

    print(x)
# Your notebook already knows about mylist. Sum its values by adding the code below this comment.



total = 0



for x in mylist:

    total = total + x



print(total)
s = "This is a test string for HCDE 530"

# Add your code below



words = s.split() # Splitting the string into words



for word in words: # Printing each word

    print(word)
# Add your code here



mylist[3] = "four" # There are multiple ways to replace an element in a list. Here, I'm assigning a different value to the 4th element (aka replacing the old value with a new value).

print(mylist) # Printing the list with a replaced element
# The code below allows you to access your Kaggle data files

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# create a file handle called fname to open and hold the contents of the data file

fname = open('/kaggle/input/a2data/test.txt', 'r')



# Add your code below

for line in fname: # Not sure why there's an error message when I see the directory in question on the right. The code should be correct through.

    print(line.rstrip())



# It's good practice to close your file when you are finished. This is in the next line.

fname.close()
dogList = ["Akita","Alaskan Malamute","Australian shepherd","Basset hound","Beagle","Boston terrier","Bulldog","Chihuahua","Cocker Spaniel","Collie","French Bulldog","Golden Retriever","Great Dane","Poodle","Russell Terrier","Scottish Terrier","Siberian Husky","Skye terrier","Smooth Fox terrier","Terrier","Whippet"]



# Add your code below

for word in dogList:

    if word.find("terrier"):

        print(word.find("terrier"))

    elif word.find("Terrier"):

        print(word.find("Terrier"))   

    else:

        print(-1)
binList = [0,1,1,0,1,1,0]



# Add your code below

for x in binList:

    if x == 1:

        print("One")



        
for word in dogList:

    if (word.find("bulldog") != -1 or word.find("Bulldog")) != -1:

        print(word)
numChars = 0

numLines = 0

numWords = 0



# create a file handle called fname to open and hold the contents of the data file

# make sure to upload your a2feed.txt file first

fname = open('/kaggle/input/a2data/test.txt', 'r')



# Add your code below to read each line in the file, count the number of characters, lines, and words

# updating the numChars, numLines, and numWords variables.

for line in fname:

    numLines = numLines + 1

    numWords = numWords + len(line.split())

    numChars = numChars + len(line.rstrip()) # Removing the newline characters

    

# output code below is provided for you; you should not edit this

print('%d characters'%numChars)

print('%d lines'%numLines)

print('%d words'%numWords)



# It's good practice to close your file when you are finished. This is in the next line.

fname.close()
# Add your code below

numChars = 0

numLines = 0

numWords = 0



# create a file handle called fname to open and hold the contents of the data file

# make sure to upload your a2feed.txt file first

fname = open('/kaggle/input/a2data/sherlock.txt', 'r')



# Add your code below to read each line in the file, count the number of characters, lines, and words

# updating the numChars, numLines, and numWords variables.

for line in fname:

    numLines = numLines + 1

    numWords = numWords + len(line.split())

    numChars = numChars + len(line.rstrip()) # Removing the newline characters

    

# output code below is provided for you; you should not edit this

print('%d characters'%numChars)

print('%d lines'%numLines)

print('%d words'%numWords)



# It's good practice to close your file when you are finished. This is in the next line.

fname.close()