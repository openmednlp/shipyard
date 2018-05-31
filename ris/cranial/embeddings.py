import pandas as pd
import bedrock
from configparser import ConfigParser
import re
from gensim.models.word2vec import Word2Vec

config = ConfigParser()
config.read('config.ini')


def vectorize_dataset(x, x_val, y, y_val, stratify=False):
    train_df = pd.DataFrame({'x': x, 'y': y})

    if stratify:
        balanced_train_df = bedrock.collection.balance_df(train_df, 'y')
        x = balanced_train_df['x']
        y = balanced_train_df['y']

    vectorizer = bedrock.feature.train_tfidf_vectorizer(
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
    print('doing w2v')
    model = Word2Vec(sentences, workers=6, size=200, min_count=1, window=15, sample=1e-3)
    words = model.wv.vocab
    vectors = model[words]
    # df = pd.DataFrame(data=vectors.transpose(), columns=words)
    return words, vectors
