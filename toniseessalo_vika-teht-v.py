import pandas as pd
# Koodi-Suomi : 
# haluaisin importoida eli tilata käyttööni "pandas" nimisen koodikirjaston.
import spacy
# Koodi-Suomi : 
# haluaisin importoida eli tilata käyttööni "spacy" nimisen koodikirjaston.
data = pd.read_csv("../input/spammit1/spam_text_messages.csv")
# Koodi-Suomi : 
# tiedän että pandas-kirjasto sisältää csv-muotoisen tiedoston lataamista helpoittavan koodipätkän nimeltään "read_csv"
# voisitko antaa tälle koodipätkälle parametreina avattavan tiedoston nimen
# voisitko sitten sujauttaa tiedostosta luetun taulukon kirjekuoreen nimeltään "data"

data.head(10)
# Koodi - Suomi : 
# pyysin sinua äsken sujauttamaan "spam_text_messages.csv"-tiedostossa olleen taulukon "data"-nimiseen kirjekuoreen/muuttujaan
# voisitko nyt tulostaa ruudulle 10 ensimmäistä riviä tästä kirjekuoresta/muuttujasta.
nlp = spacy.load("en_core_web_sm")
# Koodi-Suomi : 
# Tiedän että Spacy-kirjasto sisältää työkalun nimeltään "load" jonka avulla voi ladata kielimallin
# Voisitko tallentaa kielimallin "nlp" nimiseen kirjekuoreen/muuttujaan.
data["text"] = data["text"].apply(lambda row: " ".join([w.lemma_ for w in nlp(row)]))
# Koodi-Suomi :
# voisitko suorittaa ns. "labda-funktion", eli jonkun tietyn prosessi kaikille taulukon sarakkeen data["text"] riveille.
# funktio/prosessi jota haluan käyttää on [w.lemma_ for w in nlp(row)], eli rivin jokaisen sanan lemmatisointi
data.head(10)
# Koodi - Suomi : 
# voisitko vielä tulostaa ruudulle 10 ensimmäistä riviä tästä kirjekuoresta/muuttujasta.
import sklearn
from sklearn.model_selection import train_test_split
# Koodi - Suomi : 
# Tiedän että "train_test_split" - komento löytyy "sklearn"-työkalupakin "model_selection" -työkalusarjasta
# voisitko tuoda kyseisen työkalun koodiin käyttööni

ennustavat_muuttujat_opetus, ennustavat_muuttujat_testaus, ennustettava_muuttuja_opetus, ennustettava_muuttuja_testaus = sklearn.model_selection.train_test_split(data['text'], data['label'], test_size=0.1, random_state=1) # 70% training and 30% test
# Koodi - Suomi : 
# osoittaisitko train_test_splitille seuraavan nimiset kirjekuoret joihin se voi tallentaa valmiiksi jaetun datan :
#      1. ennustavat_muuttujat_opetus, 2. ennustavat_muuttujat_testaus, 
#      3. ennustettava_muuttuja_opetus, 4.ennustettava_muuttuja_testaus 

# antaisitko train_test_split -komennolle parametreina sulkujen sisällä 1. ennustavat muuttujat, 2. ennustettavan muuttujan
# .. sekä myös 3. "test_size" parametrin arvolla 0.1 

# antaisitko myös parametrina random_state=1 ( ei ole olennaista ymmärtää täsää vaiheessa)
from sklearn.feature_extraction.text import CountVectorizer
#Koodi-Suomi
# Haluaisin importoida sklearn nimisen kirjaston sisältämästä feature_extraction-työkalupakista löytyvästä "text"-työkalusarjasta työkalun CountVectorizer

vektorisoija = CountVectorizer() 
#Koodi-Suomi
# Haluaisin luoda uuden CountVectorizer työkalun ja tipauttaa sen vektorisoija

ennustavat_muuttujat_opetus = vektorisoija.fit_transform(ennustavat_muuttujat_opetus)
#Koodi-Suomi
# Haluaisin muuttaa tekstin vektorimuotoon käyttämällä CountVectorizer:n tarjoamaa työkalua fit_transform...
# ...ja antamalla sille parametreina ennustavat_muuttujat_opetus-kirjekuoren sisällön (opetusdatan osuus tekstiviesteistä)
# Voisitko tipauttaa vektorisoidun opetusdatan takaisin kuoreen/muuttujaan nimeltään ennustavat_muuttujat_opetus
from sklearn.neural_network import MLPClassifier
# Koodi - Suomi
# Voisitko tuoda sklearn kirjaston neural_network työkalupakista työkälun nimeltään MLPClassifier

model = MLPClassifier(solver='lbfgs',
                    hidden_layer_sizes=(1000, 1000))
# Koodi-Suomi
# Haluaisin käyttää MLPClassifier nimistä työkalua luodakseni uuden neuroverkon
# Haluaisin antaa solver-parametriksi 'lbfgs', joka on hyvä vaihtoehto gradient descentille kun dimensioita on paljon.
# Haluaisin että neuroverkossa olisi sisääntulokerroksen ja ulostulokerroksen välissä 2 välikerrosta. 
# Haluaisin että ensimmäisessä välikerroksessa olisi 800 solua ja jälkimmäisessä 180)
# Haluan että uusi neuroverkko sujautetaan kuoreen nimeltä "model"
#800/180

model.fit(ennustavat_muuttujat_opetus, ennustettava_muuttuja_opetus)
# Koodi-Suomi
# Haluaisin nyt opettaa äsken luomani neuroverkon käyttämällä opetusdataa.
from sklearn.metrics import confusion_matrix

ennustavat_muuttujat_testaus_vektorina = vektorisoija.transform(ennustavat_muuttujat_testaus)
## Koodi - Suomi :
# Haluaisin vektorisoida nyt myös testaukseen käytettävän "ennustavat_muuttujat_testaus"-datan 
#... jotta voin käyttää sitä tekoälytyökalun kanssa ennustamiseen. 


ennuste = model.predict(ennustavat_muuttujat_testaus_vektorina)
## Koodi - Suomi :
# Haluaisin käyttää predict-työkalua ennustamaan testi-datalla.

totuustaulu = confusion_matrix(ennustettava_muuttuja_testaus, ennuste)
# Koodi - Suomi :
# Haluaisin käyttää"confusion_matrix"-työkalua joka vertaa ennustetta oikeisiin tuloksiin
# Antaisin sille parametreina testamiseen tarkoitetun ennustettavat muuttujat sisältävän kirjekuoren(oikeat vastaukset)
#... ja sitten itse ennusteen
# voisitko sujauttaa vertailun lopputuloksen kirjekuoreen nimeltään "totuustaulu"

print(totuustaulu)
# Koodi - Suomi :
#voisitko tulostaa ruudulle "totuustaulu"-kirjekuoren sisällön
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
import numpy as np

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='TOTUUSTAULU',
                          cmap=plt.cm.Blues):
    
    # plt.cm.Oranges .. eli muitakin varivaihtoehtoja loytyy
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.figure(figsize = (10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size = 24)
    plt.colorbar(aspect=4)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, size = 14)
    plt.yticks(tick_marks, classes, size = 14)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    
    # Labeling the plot
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize = 20,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        
    plt.grid(None)
    plt.tight_layout()
    plt.ylabel('ennuste', size = 18)
    plt.xlabel('totuus', size = 18)
cm = confusion_matrix(ennustettava_muuttuja_testaus, ennuste)
plot_confusion_matrix(cm, classes = ['ham', 'spam'],
                      normalize = False,
                      title = 'TOTUUSTAULU')
