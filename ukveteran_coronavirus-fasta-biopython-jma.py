from Bio import SeqIO

for seq_record in SeqIO.parse('../input/customer-frauddata/coronavirus.fasta', "fasta"):

    print(seq_record.id)

    print(seq_record.seq)
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from Bio.Seq import Seq

from Bio.Alphabet import IUPAC
seq_record.seq.count("A")
from Bio.SeqUtils import GC

GC(seq_record.seq)
from Bio.SeqUtils import molecular_weight

molecular_weight(seq_record.seq)
len(seq_record.seq)