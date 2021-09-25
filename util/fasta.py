
def save_as_fasta(data, path):
    fasta = to_fasta(data)
    with open(path,'w') as file:
        file.write(fasta)

def to_fasta(data):
    fasta = ''
    for seq,labels in data:
        fasta += '>{}\n{}\n'.format(' '.join(labels), seq)
    return fasta

import pandas as pd
def fasta_file_to_df(path):
    with open(path,'r') as file:
        fasta = file.read()
    df = []
    name, seq = False, ''
    for line in fasta.splitlines():
        line = line.rstrip()
        if line.startswith(">"):
            if name:
                df.append({'name':name, 'sequence':seq})
            name = line[1:]
            seq = ''
        else: seq += line
    df.append({'name':name, 'sequence':seq})
    return pd.DataFrame(df)
