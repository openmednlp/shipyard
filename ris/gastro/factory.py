import bedrock
from models.standard import bnb


def get_data(csv_path):
    df = bedrock.collection.file_to_df(csv_path)
    # id, impression_id, sentence_id, sentence, ground_truth
    data = bedrock.collection.train_validate_split_df(
        df,
        'sentence',
        'ground_truth',
        'impression_id'
    )

    vectorizer = bedrock.feature.train_tfidf_vectorizer(data[0], 'output/vectorizer.pickle')
    x = vectorizer.transform(data[0])
    x_val = vectorizer.transform(data[1])
    return x, x_val, data[2], data[3]


def train_model(model_name, data, persist_path):
    if model_name == 'bnb':
        return bnb(data[0],data[1],data[2],data[3], 'output/model.pickle', True)
    raise ValueError('No model named {} found'.format(model_name))


if __name__ == '__main__':
    print('woohoo')
    x, x_val, y, y_val = get_data('/home/giga/dev/python/shipyard/ris/gastro/data/sentence_dataset.csv')
    data = x, x_val, y, y_val
    model = train_model('bnb', data, 'output/model.pickle')
    y_hat = model.predict(x_val)