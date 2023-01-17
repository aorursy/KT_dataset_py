from Bio.Seq import Seq
from Bio.Alphabet import IUPAC
my_seq = Seq("AGTACACTGGT", IUPAC.unambiguous_dna)
my_seq
my_seq.alphabet
my_seq.count("A")
from Bio.SeqUtils import GC
GC(my_seq)
my_mRNA = my_seq.transcribe()
my_seq.translate()
str(my_seq)
my_seq.complement()
my_seq.reverse_complement()
from Bio.Seq import Seq
from Bio.Alphabet import IUPAC
messenger_rna = Seq("AUGGCCAUUGUAAUGGGCCGCUGAAAGGGUGCCCGAUAG", IUPAC.unambiguous_rna)
messenger_rna
messenger_rna.translate()