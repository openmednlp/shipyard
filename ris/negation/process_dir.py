import bedrock
import operator
from functools import reduce
import pandas as pd
import re

t = '''
Befund und Beurteilung
Zum Vergleich Röntgen-Thorax vom 2015 vorliegend.

Kein Pneumothorax oder Pleuraerguss.
'''

speculation_patterns = [
    u'DD',
    u'könn',
    u'differential',
    u'eingeschränkte',
    u'beurteilbarkeit',
    u'suspekt',
    u'möglich',
    u'verdächtig',
    u'vermutl',
    u'verdacht',
    u'nicht[a-zöäüA-Z0-9[:blank:]]+ausgeschlossen',
    u'nicht[a-zöäüA-Z0-9[:blank:]]+ausschließbar'
]

pattern_dict = {
    'INFILTRAT': "\w*[iI]nfiltrat\w*",
    'PNEUMONIE': "\w*[pP]neumoni\w*",
    'DEKOKMPENSATION': "[dD]ekompens\w*",
    'KOMPENSATION': "[kK]ompens\w*",
    'STAUUNG': "\w*[sS]tauung\w*",
    'EMBOLIE': "\w*[eE]mbol\w*",
    'SPECULATION': '^(kein|ohne)$',
    'NEGATION': '^(' + '|'.join(speculation_patterns) + ')$'
}


def label_findings(words):
    labels = []
    for word in words:
        label = None
        for key in pattern_dict.keys():
            if re.fullmatch(pattern_dict[key], word):
                label = key
                break
        labels.append(label)
    return labels


def get_data():
    input_path = '/home/giga/dev/python/shipyard/omnlp/negation/data/sample'
    headers = ['beurteilung', 'befund und beurteilung']

    # Read all the files in a dir and assign the section labels for each row
    labeled = bedrock.collection.section_label_dir(
        input_path,
        headers,
        extensions=['']
    )

    # Create as many ids as there are rows for that accession
    ids = [[a_id]*len(l) for a_id, l in zip(labeled[0], labeled[1])]

    # Create columns out of lists (with repetition)
    ids_ext = reduce(operator.concat, ids)
    texts = reduce(operator.concat, labeled[1])
    labels = reduce(operator.concat, labeled[2])

    texts = [t.strip() for t in texts]

    # Make df out of lists
    df = bedrock.common.lists_to_df(
        [ids_ext, texts, labels],
        ['id', 'text', 'label']
    )

    # remove non-interesting rows (non-labeled)
    df = df[df['label'] != 'n/a']

    # Remove illogical rows (in this case HTML tags
    df = df[~df['text'].str.contains('^<.+>$')]

    return df


def process(df):
    # Split sentences and flatten the data
    sentences_df = pd.DataFrame(
        bedrock.process.sentence_tokenize(df['text']),
        index=df.index
    ).stack()

    df = df.join(
        pd.Series(
            index=sentences_df.index.droplevel(1),
            data=sentences_df.values,
            name='sentence'
        )
    )

    # Tokenize
    df['tokens'] = bedrock.process.tokenize(df['sentence'])

    df['word_labels'] = df['tokens'].apply(label_findings)

    return df