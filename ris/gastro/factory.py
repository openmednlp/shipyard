import bedrock
from ris.gastro.models.standard import bnb
import dill
from configparser import ConfigParser

config = ConfigParser()
config.read('config.ini')


def to_sentences(text):
    import bedrock
    return bedrock.process.sentence_tokenize(text)


def text_process_pipeline(sentences):
    import bedrock
    processed = bedrock.process.lemmatize(sentences)
    processed = bedrock.process.remove_short(processed, 5)
    return processed


def get_data(csv_path):
    df = bedrock.collection.file_to_df(csv_path)

    with open(config['DEFAULT']['preprocessor'], 'wb') as f:
        dill.dump(text_process_pipeline, f)

    df['processed'] = text_process_pipeline(df['sentence'])

    train_test_data = bedrock.collection.train_validate_split_df(
        df,
        'sentence',
        'ground_truth',
        'impression_id'
    )
    vectorizer = bedrock.feature.train_tfidf_vectorizer(
        train_test_data[0],
        config['DEFAULT']['vectorizer']
    )
    x = vectorizer.transform(train_test_data[0])
    x_val = vectorizer.transform(train_test_data[1])
    y = train_test_data[2]
    y_val = train_test_data[3]

    return x, x_val, y, y_val


def train_model(model_name, train_test_data, persist_path=None):
    if model_name == 'bnb':
        return bnb(
            train_test_data[0],
            train_test_data[1],
            train_test_data[2],
            train_test_data[3],
            persist_path,
            True
        )
    raise ValueError('No model named {} found'.format(model_name))


def run():
    print('GO GO GO!')
    with open(config['DEFAULT']['sentence_tokenizer'], 'wb') as f:
        dill.dump(to_sentences, f)

    x, x_val, y, y_val = get_data(config['DEFAULT']['data'])
    data = x, x_val, y, y_val
    train_model('bnb', data, config['DEFAULT']['model'])


if __name__ == '__main__':
    run()
