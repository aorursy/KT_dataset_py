import Bio

from Bio.Seq import Seq

dna = Seq("ACGTTGCAC")

print(dna)
from Bio.Alphabet import IUPAC

dna = Seq("AGTACACTGGT", IUPAC.unambiguous_dna)
dna.reverse_complement()

rna = dna.transcribe()

rna.translate()
from Bio.SeqUtils import GC

GC(dna)
from Bio.SeqUtils import molecular_weight

molecular_weight("ACCCGT")
from Bio.Seq import Seq



#create a sequence object

my_seq = Seq("CATGTAGACTAG")



#print out some details about it

print("seq %s is %i bases long" % (my_seq, len(my_seq)))

print("reverse complement is %s" % my_seq.reverse_complement())

print("protein translation is %s" % my_seq.translate())