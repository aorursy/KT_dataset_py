import nltk

# nltk.download("all") # just 2 minutes! Yeah!
nltk.download("book")
from nltk.book import *
text1
text1.concordance("monstrous")
text2.concordance("affection")
text3
text3.concordance("lived")
text4.concordance("nation")
text4
text5
text5.concordance('ur')
text1.similar("monstrous")
text2
text2.similar("monstrous")
text2.common_contexts(["monstrous", "very"])
text3.similar("lived")
text3.common_contexts(["lived", "dwelt"])
%matplotlib inline
text4.dispersion_plot(["citizens", "democracy", "freedom", "duties", "America"])
text3.dispersion_plot(["lived", "dwelt"])