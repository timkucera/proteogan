# -*- coding: utf-8 -*-
def fasta_to_dict(fasta):
    d = {}
    name, seq = False, ''
    for line in fasta.splitlines():
        line = line.rstrip()
        if line.startswith(">"):
            if name: d[name] = seq
            name = line[1:].split(' ')[0]
            seq = ''
        else: seq += line
    d[name] = seq
    return d

def fasta_to_list(fasta):
    l = []
    name, seq = False, ''
    for line in fasta.splitlines():
        line = line.rstrip()
        if line.startswith(">"):
            if name: l.append([name,seq])
            name = line[1:].split(' ')[0]
            seq = ''
        else: seq += line
    l.append([name,seq])
    return l
