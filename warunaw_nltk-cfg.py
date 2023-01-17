import nltk
grammar = nltk.CFG.fromstring("""

S -> Aux NP VP

S -> VP

VP -> V NP

VP -> V NP NP

NP -> Det Nominal

NP -> Nominal

NP -> Pronoun

Nominal -> Noun

Nominal -> Proper-Noun Nominal

V -> 'book'

Det -> 'that'

Noun -> 'flight' | 'flights'

Proper-Noun -> 'TWA'

Aux -> 'can'

Pronoun -> 'you'

""")



grammar.start()

grammar.productions()
print('*'* 20 + ' Top Down parsing ' + '*' * 20)

rd_parser = nltk.RecursiveDescentParser(grammar, trace =2)



sentenses = ['book that flight','can you book TWA flights']



for sent in sentenses:

    tokens = sent.split()

    print('\r\n' + sent + '------------>')

    for tree in rd_parser.parse(tokens):

        print(tree)    
print('*'* 20 + ' Bottom Up parsing ' + '*' * 20)

sr_parse = nltk.ShiftReduceParser(grammar, trace=2)



for sent in sentenses:

    tokens = sent.split()

    print(tokens)

    print(sent + '------------->\r\n')

    for tree in sr_parse.parse(tokens):

        print(tree)
parser = nltk.LeftCornerChartParser(grammar, trace = 2)



for sent in sentenses:

    print('\r\n' + sent + '==============================>\r\n')

    

    tokens = sent.split()

    print ((parser.parse(tokens)))