## Faster method to perform Text Preprocessing 

## by joining all tweets text into one single long tweet string



import pandas as pd

import numpy as np

import re

import string

import os 

from stop_words import get_stop_words # more comprehensive than nltk

from nltk.corpus import stopwords

import time 

import pdb
# read in csv

df = pd.read_csv('../input/customer-support-on-twitter/twcs/twcs.csv')

df
# join tweets text as one single long string 

# IMPT: join tweets with a newline + spacing '\n ' so we can split them back into individual tweets after preprocessing

text_list = df['text'].values.tolist()

text_list = [re.sub('\n', '', x) for x in text_list] # IMPT: remove some existing '\n'

text_one_long = '\n '.join(text_list)

assert len(text_one_long.split('\n ')) == len(text_list) # assert to ensure join later



# check 

text_one_long[0:500]
# faster method to preprocess as one long string but keep newlines



def fast_clean_text_keep_newline(text):

    print("0 Start")

    assert isinstance(text, str), "Needs to be one long joined string isntead of list of strings, if not slow"

    # lower

    text = text.lower()

    print("1")

    # first replace weird quotes for stopwords removal

    text = re.sub("‘", "'", text) # weird left quote

    text = re.sub("’", "'", text) # weird right quote

    print("2")

    # remove twitter handle without removing emails

    text = re.sub("\\B@[A-Za-z_]+", ' ', text)

    print("3")

    # remove urls 

    text = re.sub("(https?://|https?://www|www)\S+", ' ', text) # \S+ used instead since .*? more suitable regex for in between words

    print("4")

    # remove punctuation 

    text = re.sub("[—¡“”…{}]".format(string.punctuation), ' ', text)

    print("5")

    # remove emoticons 

    emo_regex = '🔣|🛰|👝|👖|🕋|🌦|❔|📅|🌂|😇|🏂|🚆|🗑|🐒|🖼|💔|🈳|🍺|🚋|🏰|😩|㊙|🐶|🦊|🎑|👊|🤸|🌨|🍩|🏆|🌰|🎋|🐐|🚑|📱|💨|🥘|😆|😟|🏍|🎪|📈|💣|🐥|🔩|🌳|🔲|🤴|💞|🏟|😐|🤥|😍|💑|🏏|🕳|✍|👧|♂|💊|😹|🚇|🙁|🐡|❎|🅿|💙|🕌|↗|💅|🕤|🎷|🔆|🏗|💯|📧|➕|😉|🕣|🌽|♥|🍗|🐭|☃|🎅|🚖|🕖|⛄|🇿|🌁|🐏|🌤|✝|⚡|🖱|🍿|♀|🕸|🏐|🍫|👢|🙈|😤|🌧|💂|😮|⛽|😵|🍨|🐙|🕍|😥|♉|🔕|🚪|💪|🍖|📿|📺|♊|🔬|🏯|🎞|🈯|🍝|😱|🕴|⏯|🛋|🏣|✖|⌚|🔈|☸|🍹|🌍|📟|⛓|🛏|🏢|🔁|🐻|⚒|🚄|😙|😒|🇨|🔞|👹|👙|🐳|🈲|🛎|🍽|⏺|⚛|🐿|🌖|🍱|🦆|📙|⛳|⛸|🆔|🌾|🇱|😚|🎗|⚪|🚞|🎹|🌚|🗯|🇫|🚚|📤|🎰|💃|✒|📔|🔪|💤|🏈|🍪|🔦|🗄|ℹ|🛍|✌|🍴|🎠|🚢|➰|🌀|🍠|🦌|♒|💧|🕜|😠|🤳|🅱|🌸|👷|🖲|🐘|🥚|🚭|⛈|🏥|🎓|🌩|🏳|🌻|🕒|🌘|🕚|♑|💖|😓|📋|8|🗒|☹|🔡|🚒|🆘|🏁|⛷|💸|🐪|👔|🦎|🏵|🐓|🎇|🐅|🔃|😘|🥔|🤓|🏎|💐|⚔|🦍|🤠|😀|❣|🔗|🔚|🖖|🇦|😧|🎁|🍐|🏅|🌉|👛|🐷|🌊|🍉|🐜|🕐|🍇|🇳|☠|💡|🚯|🏜|🏷|🐃|⛩|⚽|🚈|🇹|⛎|🏴|🛃|🍄|👕|⚗|🖨|🌭|⛵|🛣|⬆|☮|🔫|📀|🤧|🏧|🤶|⛅|😅|⏫|☔|🌕|↔|🛁|🕠|💄|💹|🛂|🎊|🌐|🏒|↩|🔜|▶|😗|🖊|📝|💋|🌜|🔔|🔵|🕔|💓|😋|🔒|🥗|🛐|🥂|📮|🥜|🏾|📸|💌|😪|📯|🍜|💵|🈹|🐬|🎩|🐉|💇|🎸|🥝|👄|❇|🚘|🆓|👿|🕵|📭|🍈|🥓|🏓|🚽|⚫|🕦|🇧|✉|📰|🦋|🍘|🚾|🖍|🕹|📫|🕊|💲|🐰|🚨|☯|🎡|☘|💻|🔷|🎥|🔏|🛢|❄|✅|‼|👗|👳|➿|🥋|⤵|🛵|➖|📆|🎙|🏪|👏|💘|💮|🛩|⁉|😑|🚹|💰|🎐|㊗|🛥|✔|🏩|💦|🐤|🕎|📲|🐌|🏤|💜|🙊|👋|🐸|💭|📥|🛶|✋|💀|🎳|📽|🤖|↕|👆|🗂|🚼|❤|🌏|🆚|📍|3|📉|🚂|🍋|😈|🌇|☎|🎲|🎄|🔨|🔴|🤡|🥑|️|🙄|👯|🅾|🤾|♈|🕕|⏬|◽|⏰|🏼|💫|🎚|❓|↖|🔌|🚱|✡|👬|🔅|🐖|🔇|🔋|👂|🥞|🛑|🐺|🌓|🏡|1|🐗|🦑|😺|♿|⌛|🌃|🌈|🎬|👫|🍚|Ⓜ|🌫|😛|⚰|🏦|✳|🌌|🐽|🔭|😊|🚕|➡|🚁|🌪|🚓|🐟|🚷|🎻|🔂|♨|🚴|🚩|🗣|😝|🍵|🤞|🎢|🔙|🌹|💶|⏮|🌆|🆎|🔮|📛|🤺|⛹|🔺|5|😨|♎|😻|🈶|🛠|❌|⚕|🏄|🚜|🤵|👓|🏚|🈁|🆗|👸|📻|😄|🔐|◻|🥖|🀄|👮|🔻|🌛|⛰|💝|👻|🚍|🚤|⏸|💕|⏩|😶|🎽|🛫|⚠|📘|🦀|💟|🌗|👐|🤘|4|™|🛤|😰|👟|🕡|🍓|📑|😂|🐝|🌑|⛴|🎱|🚿|☂|🍯|⤴|📕|🍦|⛲|🚰|🚛|🇩|🤦|👁|🦐|🈵|🐕|⃣|🐛|🏸|🌋|🥐|😯|📡|💽|🐋|👒|📞|🗝|📦|🔀|🚺|⚖|👤|🦅|✏|🚉|👰|💍|©|📶|🥒|🐞|🍕|📒|🕺|📚|🍔|🕘|🕙|➗|9|🏝|🦇|🛌|🇺|🛡|🌔|🍃|🇲|😎|🏀|🐵|💛|🍥|👺|👘|🦂|®|🚲|🗡|🍟|📗|😏|🇾|👦|🖋|🚮|🦉|🍞|💁|⬜|😬|👭|🍍|🙅|🌿|🍰|🕛|👌|👩|🤤|🆑|⏪|👃|😡|🏕|🛀|🎮|🌶|🐂|🕑|🙀|🏹|💚|😦|🐄|📼|🎦|👵|💢|🔘|💥|🚸|🔉|🐊|🃏|🍭|⏲|⏳|🍳|📐|👥|🖇|🕗|💗|📷|🆖|👼|⚾|🕉|🐣|😕|🤰|🕯|🥕|💱|🏊|🌞|🐠|👽|🍬|🍼|👅|🛴|🌱|🐇|◀|⚓|🍣|🚣|♋|🚶|🤔|🙃|▫|🐧|〰|🍶|🈷|📪|🛒|♍|🇯|😌|🚝|🍑|📨|🕓|🤽|🚠|🐆|🤚|🔍|🏋|🛬|🎌|📄|📖|🔟|🖕|😭|👀|🤣|🏉|🎿|🌠|🔱|🌷|🥙|💎|👪|🎭|🐯|🗨|💾|🔠|🙌|🔼|🤝|👎|🙆|🚏|🏛|🙉|🇷|😲|🏨|🚧|📎|📵|♓|🍲|🔖|♦|😾|💠|🖤|🔶|☪|🐼|🌯|👡|🏔|▪|🎆|🇬|🏻|🕥|📂|🌟|🐎|🌡|#|🇰|🗃|⌨|😼|🍅|🇪|💈|🎧|🏫|🌬|🚊|🐁|✨|💳|🔧|❕|❗|✈|🍻|🍏|🤢|🤼|🚫|🖐|☺|‍|🗿|🐾|🈸|0|🥉|👨|🤙|🛅|🏭|🍢|♠|🥈|🥇|🇶|🗜|💬|🙇|🇵|🔳|🌄|🤐|🤕|🔑|🤑|🚃|🚅|🎛|🍊|🐹|🏞|🎉|🎶|🔛|🔤|😜|🏙|🖥|🥊|🔝|↘|🕰|💴|😿|🎒|🙂|⬛|👱|🙋|🥀|🇼|🎃|🐩|🗽|📬|💺|🎀|🉐|🎟|🇻|🏺|🎯|🥁|📃|🕢|⛺|🌮|🎣|🌎|😃|🉑|2|🆒|🌼|💆|🤜|👍|🕝|◼|♣|🎂|😴|🌺|☣|↙|🐫|🚻|🐀|🥅|🎍|🥄|🤗|🚥|🔄|🍀|⭐|🌥|📠|😸|🖌|📴|💉|🐑|🍒|🎼|☀|🌝|📁|🚡|🎈|☦|👞|🦈|☝|🅰|⬇|👶|🌅|7|🐢|⏱|😔|🈚|🐈|🎤|🚔|📩|📳|🍂|☕|🏽|🏌|👴|🎵|🏮|🔥|🎎|♐|🍾|🆕|😁|🙎|🍸|🌲|🐦|🥃|🇸|☄|🔎|🤛|🚟|🍡|🚎|⛔|♌|👇|📏|🦄|😢|💷|🏑|👉|♏|🔢|📇|👑|💏|🛳|🚬|🚵|🤷|🚗|🏿|📌|📓|⛪|◾|🍎|🕷|💒|👚|🎨|🌵|🌴|🕧|📢|💩|👣|😖|🍮|🗓|✊|🇭|👜|🏖|🥛|🈴|🔯|📣|🍧|🐮|🐔|🈺|🗼|☑|⬅|⛏|🗞|😷|🌙|🐚|6|⏭|🐍|🕞|🌒|🗳|🗾|🎖|♻|🚦|😽|🎏|🤹|🚐|🔸|🚙|🐨|🔰|📊|🚳|🍁|👈|🙍|🦁|🗺|⭕|👲|⏹|🍷|☢|📜|🇽|🦃|⛱|🛄|🇴|🍌|🧀|🏬|👠|↪|〽|🚀|🍤|🔊|🕶|🍛|🎾|🆙|🐲|🏃|🦏|💿|👾|🔓|🕟|⚙|😫|🏇|🇮|⏏|🍆|🤒|🎴|🏠|🗻|🙏|🐴|⛑|🎺|☁|😞|🔹|📹|🔽|🎫|😳|⚱|⚜|✴|✂|🚌|💼|🐱|🈂|😣|🍙'

    text = re.sub(emo_regex, ' ', text)

    print("6")

    # remove stop_words last

    # IMPT: split by thick quotes " " to keep newlines \n

    text_split_keep_newline = text.split(" ")

    text_split_keep_newline = [w for w in text_split_keep_newline if w != ''] # IMPT: remove spaces '' when split

    stop_words = get_stop_words('en') # get_stop_words more comprehensive than nltk

    nltk_words = stopwords.words('english')

    stop_words.extend(nltk_words)

    stop_words = list(set(stop_words)) 

    text = ' '.join([w for w in text_split_keep_newline if w not in stop_words])

    print("7")

    # remove extra spacing behind '\n' caused by ' '.join above 

    text = re.sub('(?<=\S) \n', '\n', text) # IMPT: adding (?<=\S) positive look-behind (<) does two things: 

                                            # (i) matches end of sentence char before ' \n' but does not replace that char  

                                            # (ii) stops matching start of an empty space sentence

    print("8")

    

    # DONE!

    print("DONE!")

    return text
# fast clean one long string

start = time.time() 

text_one_cleaned = fast_clean_text_keep_newline(text_one_long)

end = time.time()

print("Fast clean took: %s seconds" % (end-start))
# split preprocessed text back into individual tweets by newline + spacing '\n '

text_cleaned_split = text_one_cleaned.split('\n ') 



# check

text_cleaned_split 
# join back with original df 

df_cleaned = df

df_cleaned['text_cleaned'] = text_cleaned_split 



# check, DONE!

df_cleaned
# write out as csv

df_cleaned.to_csv('twcs_text_cleaned.csv')