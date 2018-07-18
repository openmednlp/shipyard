import pandas as pd
from configparser import ConfigParser
import re
from gensim.models.word2vec import Word2Vec

config = ConfigParser()
config.read('config.ini')


def vectorize_dataset(x, x_val, y, y_val, stratify=False):
    from bedrock.feature import train_tfidf_vectorizer
    from bedrock.collection import balance_df

    train_df = pd.DataFrame({'x': x, 'y': y})

    if stratify:
        balanced_train_df = balance_df(train_df, 'y')
        x = balanced_train_df['x']
        y = balanced_train_df['y']

    vectorizer = train_tfidf_vectorizer(
        x,
        config['DEFAULT']['vectorizer']
    )

    x_balanced_vec = vectorizer.transform(x)
    x_val_vec = vectorizer.transform(x_val)

    return x_balanced_vec, x_val_vec, y, y_val


def regex_label_sentences(sentences, pattern_dict):
    labels = []
    for sentence in sentences:
        label = None
        for key in pattern_dict.keys():
            if re.search(pattern_dict[key], sentence):
                label = key
                break
        labels.append(label)
    return labels


def word2vec(sentences):
    from gensim.models.word2vec import Word2Vec

    print('doing w2v')
    model = Word2Vec(sentences, workers=6, size=200, min_count=1, window=15, sample=1e-3)
    words = model.wv.vocab
    vectors = model[words]
    # df = pd.DataFrame(data=vectors.transpose(), columns=words)
    return words, vectors
