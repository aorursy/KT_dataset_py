!pip uninstall spacy -y
!pip uninstall neuralcoref
!pip install -q spacy==2.1.0 --user
!python -m spacy download en
!pip install -q neuralcoref --no-binary neuralcoref
import logging;

logging.basicConfig(level=logging.INFO)



# Load your usual SpaCy model (one of SpaCy English models)

import spacy

nlp = spacy.load('en')



# Add neural coref to SpaCy's pipe

import neuralcoref

neuralcoref.add_to_pipe(nlp)

# You're done. You can now use NeuralCoref as you usually manipulate a SpaCy document annotations.

doc = nlp(u'Who is Abraham Lincoln? When was he born? Where is his hometown?')



print(doc._.has_coref)

print(doc._.coref_clusters)
# You're done. You can now use NeuralCoref as you usually manipulate a SpaCy document annotations.

doc = nlp(u'What’s the weather like in Hong Kong? How much are the flights to get there?')



print(doc._.has_coref)

print(doc._.coref_clusters)
QA = {"Who is Abraham Lincoln?":"An American statesman and lawyer who served as the 16th President of the United States.",

      "When was Abraham Lincoln born?":"February 12, 1809.",

      "Where is Abraham Lincoln's hometown?":"Hodgenville, Kentucky"}
def my_coref(orig_text, to_replace):

    left = 0

    processed_text = ""

    for beg,end,mention in to_replace:

        processed_text += orig_text[left:beg] + mention

        left = end

    processed_text += orig_text[left:]

    return processed_text
def answer(question):

    print('Question: ', question)

    global session_context

    start_pos = len(session_context)

    session_context += (question + " ")

    # print("context:",context)

    if question in QA: 

        return QA[question]

    else:

        doc = nlp(session_context)

        if doc._.has_coref:

            # print(doc._.coref_clusters)

            to_replace = []

            for clust in doc._.coref_clusters:

                main_mention = clust.main

                for mention in clust.mentions:

                    beg, end = mention.start_char - start_pos, mention.end_char - start_pos

                    if end > 0:                                     # 是本句中的指代

                        if mention.text in ["its","his","her","my","your","our","their"]:

                            to_replace.append((beg,end,main_mention.text+"'s"))

                        else:

                            to_replace.append((beg,end,main_mention.text))

            to_replace = sorted(to_replace)                         # 按照起始位置升序排序，为逐个替换做准备

            question2 = my_coref(question,to_replace)

            print("Coreferenced question:",question2)

            if question2 in QA:

                return QA[question2]

                    

    return "I don't know."
session_context = ""
answer("Who is Abraham Lincoln?")
answer("When was he born?")
answer("Where is his hometown?")