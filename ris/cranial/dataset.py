import bedrock
import numpy as np


def crawl_reports(dir_path='/home/giga/dev/python/shipyard/ris/cranial/data'):
    headers = [
        'anamnese',
        'fragestellung',
        'technik',
        'befund',
        'beurteilung'
    ]

    result_ids = []
    result_lines = []
    result_sent_id = []
    result_labels = []

    file_ids, text_lines_list, labels_list = bedrock.collection.section_label_dir(dir_path, headers)

    for file_id, text_lines, labels in zip(file_ids, text_lines_list, labels_list):
        text_lines_sentences = bedrock.process.sentence_tokenize(text_lines)
        for text_line_sentences, label in zip(text_lines_sentences, labels):
            sentences_count = len(text_line_sentences)

            result_ids.extend([file_id]*sentences_count)
            result_labels.extend([label] * sentences_count)
            result_sent_id.extend(list(range(sentences_count)))
            result_lines.extend(text_line_sentences)

    lists = [result_ids, result_labels, result_sent_id, result_lines]
    columns = 'accession_id section_id sentence_id sentence'.split()
    sent_df = bedrock.common.lists_to_df(lists, columns)
    sent_df.to_csv('cranial_sentences.csv', encoding='utf-8')

    print(sent_df.sample())
    print('Bye!')


def load_data(csv_path):
    import pandas as pd
    df = pd.read_csv(csv_path, encoding='utf-8', low_memory=False)
    return df


def map_label(df, target_column):
    replace_dict = {np.nan: 0, 2: 1, 3: 1, 4: 1, 5: 1}
    df[target_column].replace(replace_dict, None, inplace=True)
    return df


def csv_to_dataset(csv_path, row_limit=None):
    df = load_data(csv_path)

    if row_limit is not None:
        df = limit_df(df, row_limit)

    df = fix_df_encoding(df)

    return df


def fix_df_encoding(df):
    df['sentence'] = df['sentence'].apply(lambda x: str(x).encode('cp1252').decode('utf8'))
    df['sentence'] = bedrock.process.lemmatize(df['sentence'])
    return df


def limit_df(df, row_limit):
    df = df[:row_limit]
    return df


def filter_df_by_sections(df, target_column, section_ids):
    if type(section_ids) == str:
        section_ids = [section_ids]
    df = df[df['section_id'].isin(section_ids)]
    df = map_label(df, target_column)
    return df


def get_data(df, target_column, section_ids, test_size=0.7):
    if type(section_ids) == str:
        section_ids = [section_ids]

    df = df[df['section_id'].isin(section_ids)]

    df = map_label(df, target_column)

    x, x_val, y, y_val = bedrock.collection.train_validate_split_df(
        df,
        'sentence',
        target_column,
        'accession_id',
        test_size=test_size
    )
    return x, x_val, y, y_val