# -*- coding: utf-8 -*-
'''
HOW TO
######
To define a custom filter, create a function <function_name> with <args>. It should
wrap another function that accepts a data.dataset.RawDataset class
(you can access the protein data inside via RawDataset.df) and returns
a pandas DataFrame. You can then apply the filter in the data/config.json
file with:

{"filter_fx": <function_name>, "filter_args": <args>}

Please don't forget to register your function in the dictionary at the end of this file.

'''


###########################
# Filters protein sequences shorter than a cut-off value
def maximum_length(length):

    def fx(raw_data):
        df = raw_data.df
        df = df.loc[df['sequence'].map(lambda l: len(l)) <= length]
        return df

    return fx

###########################
# Truncate sequence length (for VAE)
def truncate(length):

    def fx(raw_data):
        df = raw_data.df
        df['sequence'] = df['sequence'].map(lambda l: l[:length])
        return df

    return fx

###########################
# Filters labels with a minimum number of labels
def minimum_class_member_threshold(threshold):

    def filter_terms(labels, target_labels):
        return list(labels.intersection(target_labels))

    def fx(raw_data):
        df = raw_data.df
        terms = raw_data.terms
        target_labels = set(terms[terms['count'] >= threshold]['term'].tolist())
        df['labels'] = df.apply(lambda row: filter_terms(set(row['labels']), target_labels), axis=1)
        df = df.loc[df['labels'].map(lambda l: len(l)) > 0]
        return df

    return fx

###########################
# Filters the n largest GO classes (by number of members)
def n_largest_classes(n):

    def filter_terms(labels, target_labels):
        return list(labels.intersection(target_labels))

    def fx(raw_data):
        df = raw_data.df
        terms = raw_data.terms.sort_values(by=['count'], ascending=False)
        target_labels = set(terms['term'].tolist()[:n])
        df['labels'] = df.apply(lambda row: filter_terms(set(row['labels']), target_labels), axis=1)
        df = df.loc[df['labels'].map(lambda l: len(l)) > 0]
        return df

    return fx

###########################
# Filters only a given list of GO terms
def only_terms(terms):

    def filter_terms(labels):
        return list(labels.intersection(set(terms)))

    def fx(raw_data):
        df = raw_data.df
        df['labels'] = df.apply(lambda row: filter_terms(set(row['labels'])), axis=1)
        df = df.loc[df['labels'].map(lambda l: len(l)) > 0]
        return df

    return fx


###########################
# Filters sequences not containing a list of amino acids. Used to filter uncommon aminos
def amino_acids(do_not_contain='OUBZXJ'):

    def fx(raw_data):
        df = raw_data.df
        df = df[~df['sequence'].str.contains('|'.join(do_not_contain))]
        return df

    return fx





###########################
# Register functions. Use the name you want to use in the config file as a key.
filter = {
    'maximum_length': maximum_length,
    'minimum_class_member_threshold': minimum_class_member_threshold,
    'only_terms': only_terms,
    'amino_acids': amino_acids,
    'n_largest_classes': n_largest_classes,
    'truncate': truncate
}
