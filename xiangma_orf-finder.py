from Bio import SeqIO
records = SeqIO.parse('../input/dna2.fasta', 'fasta')
for record in records:
    for strand, seq in (1, record.seq), (-1, record.seq.reverse_complement()):
        for frame in range(3):
            length = 3 * ((len(seq)-frame) // 3)
            for pro in seq[frame:frame+length].translate(table = 1).split("*")[:-1]:
                if 'M' in pro:
                    orf = pro[pro.find('M'):]
                    pos = seq[frame:frame+length].translate(table=1).find(orf)*3 + frame +1
                    if len(orf)*3 +3 > 1300:
                        print("{}...{} - length {}, strand {}, frame {}, pos {}, name {}".format\
                               (orf[:3], orf[-3:], len(orf)*3+3, strand, frame, pos, record.id))


from Bio import SeqIO
import re
records = SeqIO.parse('../input/dna2.fasta', 'fasta')

for record in records:
    for strand, seq in (1, record.seq), (-1, record.seq.reverse_complement()):
        for frame in range(3):
            index = frame
            while index < len(record) - 6:
                match = re.match('(ATG(?:\S{3})*?T(?:AG|AA|GA))', str(seq[index:]))
                if match:
                    orf = match.group()
                    index += len(orf)
                    if len(orf) > 1300:
                        pos = str(record.seq).find(orf) + 1 
                        print("{}...{} - length {}, strand {}, frame {}, pos {}, name {}".format\
                               (orf[:6], orf[-3:], len(orf), strand, frame, pos, record.id))
                else: index += 3
                             
from Bio import SeqIO
records = list(SeqIO.parse('../input/dna2.fasta', 'fasta'))
list_query = list()
for r in records:
    for i in range(len(r)):
        if len(r.seq[i: i+7]) < 7: continue
        list_query.append(str(r.seq[i: i+7]))
query = set(list_query)

count = dict()
for q in query:
    num = 0
    for r in records:
        num += r.seq.count_overlap(q) 
    count[q] = num
print(sorted([(v,k) for k,v in count.items() if v > 30], reverse = True))
    