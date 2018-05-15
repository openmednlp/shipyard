import re
import dill
from configparser import ConfigParser

config = ConfigParser()
config.read('config.ini')


def process_report(text):
    import bedrock
    import pandas as pd

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

    headers = ['beurteilung', 'befund und beurteilung']

    if type(text) is not list:
        text_lines = text.splitlines()
    else:
        text_lines = text

    # Read all the files in a dir and assign the section labels for each row
    content_labeles = bedrock.collection.section_label(text, headers)

    text_lines = [t.strip() for t in text_lines]

    # Make df out of lists
    df = bedrock.common.lists_to_df(
        [text_lines, content_labeles],
        ['text', 'content_label']
    )

    # remove non-interesting rows (non-labeled)
    df = df[df['content_label'] != 'n/a']

    # Remove illogical rows (in this case HTML tags
    df = df[~df['text'].str.contains('^<.+>$')]

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


def run():
    with open(config['DEFAULT']['LABELER'], 'wb') as f:
        dill.dump(process_report, f)


if __name__ == '__main__':
    print('running')
    run()
    print('done')
