import re

import sys

import random

import string



import pandas as pd
hin_vowels = ["अ", "आ", "इ", "ई", "उ", "ऊ", "ए", "ऐ", "ओ", "औ"]

hin_sonorants = ["ऋ", "ॠ", "ऌ"]

hin_anuswara = ["अं"]

hin_nukta = ["़"]

hin_consonants = [

    "क", "ख", "ग", "घ", "ङ",

    "च", "छ", "ज", "झ", "ञ",

    "ट", "ठ", "ड", "ढ", "ण",

    "त", "थ", "द", "ध", "न",

    "प", "फ", "ब", "भ", "म",

    "य", "र", "ल", "व",

    "श", "ष", "स", "ह"

]
all_country_codes = {

    0, 1, 7, 20, 27, 30, 31, 32, 33, 34, 36, 39, 40, 41, 43, 44, 45, 46, 47,

    48, 49, 51, 52, 53, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 65, 66, 81,

    82, 84, 86, 90, 91, 92, 93, 94, 95, 98, 211, 212, 213, 216, 218, 220,

    221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234,

    235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248,

    249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 260, 261, 262, 263,

    264, 265, 266, 267, 268, 269, 290, 291, 297, 298, 299, 350, 351, 352,

    353, 354, 355, 356, 357, 358, 359, 370, 371, 372, 373, 374, 375, 376,

    377, 378, 379, 380, 381, 382, 383, 385, 386, 387, 389, 420, 421, 423,

    500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 590, 591, 592, 593,

    594, 595, 596, 597, 598, 599, 670, 672, 673, 674, 675, 676, 677, 678,

    679, 680, 681, 682, 683, 685, 686, 687, 688, 689, 690, 691, 692, 800,

    808, 850, 852, 853, 855, 856, 870, 878, 880, 881, 882, 883, 886, 888,

    960, 961, 962, 963, 964, 965, 966, 967, 968, 970, 971, 972, 973, 974,

    975, 976, 977, 979, 992, 993, 994, 995, 996, 998

}



def get_codes_starting_with(prefix):

    """

    Prints all the country codes starting with the `prefix`.

    """

    found_codes = []

    for code in all_country_codes:

        if str(code).startswith(prefix):

            found_codes.append(code)

    return found_codes





check_codes = ["91", "1", "7", "41", "57"]

for check_code in check_codes:

    print("Prefix to check:", check_code)

    print("Found match:", *get_codes_starting_with(check_code))

    print()
random.seed(2019)

rand_codes = random.choices(list(all_country_codes), k=10)

print("Random country codes:", *rand_codes)

rand_codes_combined = "".join(map(str, rand_codes))

print("Concatenated codes string:", rand_codes_combined)



orig_rand_codes = []

current_code = ""

for i in rand_codes_combined:

    current_code += i

    if int(current_code) in all_country_codes:

        orig_rand_codes.append(current_code)

        current_code = ""

    

print("Decoded parts:", *orig_rand_codes)
print("Roman characters")

roman_chars = string.ascii_letters[:26]

for i, roman_char in enumerate(roman_chars):

    print((roman_char, len(roman_char.encode('utf8'))), end=" ")

    if (i+1)%10 == 0:

        print()

print()



print("\nDevanagari characters")

devanagari_chars = hin_vowels + hin_sonorants + hin_anuswara + hin_consonants

for i, devanagari_char in enumerate(devanagari_chars):

    print((devanagari_char, len(devanagari_char.encode('utf8'))), end=" ")

    if (i+1)%10 == 0:

        print()

print()
hin2wx_vowels = {

    "अ": "a",

    "आ": "A",

    "इ": "i",

    "ई": "I",

    "उ": "u",

    "ऊ": "U",

    "ए": "e",

    "ऐ": "E",

    "ओ": "o",

    "औ": "O",

    "ै": "E",

    "ा": "A",

    "ो": "o",

    "ू": "U",

    "ु": "u",

    "ि": "i",

    "ी": "I",

    "े": "e",

}

hin2wx_sonorants = {

    "ऋ": "q",

    "ॠ": "Q",

    "ऌ": "L"

}

hin2wx_anuswara = {"अं": "M", "ं": "M"}

hin2wx_consonants = {

    "क": "k",

    "ख": "K",

    "ग": "g",

    "घ": "G",

    "ङ": "f",

    "च": "c",

    "छ": "C",

    "ज": "j",

    "झ": "J",

    "ञ": "F",

    "ट": "t",

    "ठ": "T",

    "ड": "d",

    "ढ": "D",

    "ण": "N",

    "त": "w",

    "थ": "W",

    "द": "x",

    "ध": "X",

    "न": "n",

    "प": "p",

    "फ": "P",

    "ब": "b",

    "भ": "B",

    "म": "m",

    "य": "y",

    "र": "r",

    "ल": "l",

    "व": "v",

    "श": "S",

    "ष": "R",

    "स": "s",

    "ह": "h",

}

hin2wx_all = {

    **hin2wx_vowels, **hin2wx_anuswara,

    **hin2wx_sonorants, **hin2wx_consonants

}
def is_vowel_hin(char):

    """

    Checks if the character is a vowel.

    """

    if char in hin2wx_anuswara or char in hin2wx_vowels:

        return True

    return False





def hin2wx(hin_string):

    """

    Converts the Hindi string to the WX string.

    

    This function goes through each character from the hin_string and

    maps it to a corresponding Roman character according to the

    Devanagari to Roman character mapping defined previously.

    """

    wx_string = []

    for i, current_char in enumerate(hin_string[:-1]):

        # skipping over the character as it's not included

        # in the mapping

        if current_char == "्":

            continue



        # get the Roman character for the Devanagari character

        wx_string.append(hin2wx_all[current_char])



        # Handling of "a" sound after a consonant if the next

        # character is not "्" which makes the previous character half

        if not is_vowel_hin(current_char):

            if hin_string[i+1] != "्" and not is_vowel_hin(hin_string[i+1]):

                wx_string.append(hin2wx_all["अ"])



    wx_string.append(hin2wx_all[hin_string[-1]])

    if not is_vowel_hin(hin_string[-1]):

        wx_string.append(hin2wx_all["अ"])



    wx_string = "".join(wx_string)

    

    # consonant + anuswara should be replaced by

    # consonant + "a" sound + anuswara

    reg1 = re.compile("([kKgGfcCjJFtTdDNwWxXnpPbBmyrlvSRsh])M")

    wx_string = reg1.sub("\g<1>aM", wx_string)



    # consonant + anuswara should be replaced by

    # consonant + "a" sound + anuswara

    reg1 = re.compile("([kKgGfcCjJFtTdDNwWxXnpPbBmyrlvSRsh])M")

    wx_string = reg1.sub("\g<1>aM", wx_string)



    return wx_string

pairs = [

    ("शहरों", "SaharoM"),

    ("खूबसूरत", "KUbasUrawa"),

    ("बैंगलोर", "bEMgalora"),

    ("कोलकाता", "kolakAwA"),

    ("हैदराबाद", "hExarAbAxa"),

    ("कोझिकोडे", "koJikode"),

    ("सफर", "saPara"),

    ("उसमे", "usame"),

    ("संभावनाओं", "saMBAvanAoM"),

    ("मुंबई", "muMbaI"),

    ("नई", "naI"),

    ("मंगलवार", "maMgalavAra"),

    ("घंटे", "GaMte"),

    ("ट्रंप", "traMpa"),

    ("डोनाल्ड", "donAlda"),

    ("स्टेट", "steta"),

    ("संगठन", "saMgaTana"),

    ("प्रतिबंध", "prawibaMXa"),

    ("एंड", "eMda"),

    ("अंदेशे", "aMxeSe")

]



test_df = pd.DataFrame(pairs, columns=["Hindi String", "Actual WX"])

test_df["Our WX"] = test_df["Hindi String"].apply(hin2wx)

test_df["Both WX eq?"] = test_df["Actual WX"] == test_df["Our WX"]

test_df.index = test_df.index + 1

test_df
wx2hin_vowels = {

    "a": "अ",

    "A": "आ",

    "i": "इ",

    "I": "ई",

    "u": "उ",

    "U": "ऊ",

    "e": "ए",

    "E": "ऐ",

    "o": "ओ",

    "O": "औ"

}

wx2hin_vowels_half = {

    "A": "ा",

    "e": "े",

    "E": "ै",

    "i": "ि",

    "I": "ी",

    "o": "ो",

    "U": "ू",

    "u": "ु"

}

wx2hin_sonorants = {

    "q": "ऋ",

    "Q": "ॠ",

    "L": "ऌ"

}

wx2hin_anuswara = {"M": "अं"}

wx2hin_anuswara_half = {"M": "ं"}

wx2hin_consonants = {

    "k": "क",

    "K": "ख",

    "g": "ग",

    "G": "घ",

    "f": "ङ",

    "c": "च",

    "C": "छ",

    "j": "ज",

    "J": "झ",

    "F": "ञ",

    "t": "ट",

    "T": "ठ",

    "d": "ड",

    "D": "ढ",

    "N": "ण",

    "w": "त",

    "W": "थ",

    "x": "द",

    "X": "ध",

    "n": "न",

    "p": "प",

    "P": "फ",

    "b": "ब",

    "B": "भ",

    "m": "म",

    "y": "य",

    "r": "र",

    "l": "ल",

    "v": "व",

    "S": "श",

    "R": "ष",

    "s": "स",

    "h": "ह",

}

wx2hin_all = {

    **wx2hin_vowels,

    **wx2hin_vowels_half,

    **wx2hin_sonorants,

    **wx2hin_anuswara,

    **wx2hin_anuswara_half,

    **wx2hin_consonants

}
def is_vowel_wx(char):

    if char in {"a", "A", "e", "E", "i", "I", "o", "O", "u", "U", "M"}:

        return True

    return False

    



def wx2hin(wx_string):

    """

    Converts the WX string to the Hindi string.

    

    This function goes through each character from the wx_string and

    maps it to a corresponding Devanagari character according to the

    Roman to Devanagari character mapping defined previously.

    """

    wx_string += " "

    hin_string = []

    for i, roman_char in enumerate(wx_string[:-1]):

        if is_vowel_wx(roman_char):

            # If current character is "a" and not the first character

            # then skip

            if roman_char == "a" and i != 0:

                continue



            if roman_char == "M":

                hin_string.append(wx2hin_anuswara_half[roman_char])

            elif i == 0 or wx_string[i-1] == "a":

                hin_string.append(wx2hin_vowels[roman_char])

            else:

                hin_string.append(wx2hin_vowels_half[roman_char])

        else:

            hin_string.append(wx2hin_all[roman_char])

            if not is_vowel_wx(wx_string[i+1]) and wx_string[i+1] != " ":

                hin_string.append("्")

    return "".join(hin_string)
test_df = pd.DataFrame(pairs, columns=["Hindi String", "Actual WX"])

test_df["Our Hin"] = test_df["Actual WX"].apply(wx2hin)

test_df["Both Hin eq?"] = test_df["Hindi String"] == test_df["Our Hin"]

test_df.index = test_df.index + 1

test_df
!pip install wxconv
from wxconv import WXC



hin2wx = WXC(order='utf2wx', lang="hin").convert



test_df = pd.DataFrame(pairs, columns=["Hindi String", "Actual WX"])

test_df["Our WX"] = test_df["Hindi String"].apply(hin2wx)

test_df["Both WX eq?"] = test_df["Actual WX"] == test_df["Our WX"]

test_df.index = test_df.index + 1

test_df
wx2hin = WXC(order='wx2utf', lang="hin").convert

test_df = pd.DataFrame(pairs, columns=["Hindi String", "Actual WX"])

test_df["Our Hin"] = test_df["Actual WX"].apply(wx2hin)

test_df["Both Hin eq?"] = test_df["Hindi String"] == test_df["Our Hin"]

test_df.index = test_df.index + 1

test_df