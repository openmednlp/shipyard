import pandas as pd
from dataset import load_xls, assign_key
import time
import gensim
from tqdm import tqdm
import collections
from scipy.spatial import cKDTree


def extract_data(input_path, output_path):
    xls_df = load_xls(
        input_path,
        ['Definitiver Bericht']
    )
    composite_key_fields = [
        'E-Laufnummer', 'Eingangsdatum', 'Entnahmedatum', 'Patienten-ID', 'Sequenz'
    ]
    xls_df = assign_key(xls_df, composite_key_fields)

    print(xls_df.columns.values)
    print(xls_df.sample(1))

    xls_df.to_csv(output_path)

    return xls_df


def import_csv(csv_path):
    # start_time = time.time()
    csv_df = pd.read_csv(csv_path)
    # print("--- %s seconds --- load csv time" % (time.time() - start_time))
    csv_df = csv_df.fillna('')
    return csv_df


def tag_diagnoses(diagnoses, ids=None):
    start_time = time.time()

    id_with_diagnoses = enumerate(diagnoses) if ids is None else zip(ids, diagnoses)
    tagged_diagnoses = [
        gensim.models.doc2vec.TaggedDocument(
            gensim.utils.simple_preprocess(diag),
            [idx]
        )
        for idx, diag in id_with_diagnoses
    ]
    print("--- %s seconds --- preprocessing time" % (time.time() - start_time))
    return tagged_diagnoses


def train_test_split(tagged_diagnoses, ratio=.9):
    n = len(tagged_diagnoses)
    split_idx = int(n * ratio)
    train_corpus = tagged_diagnoses[:split_idx]
    test_corpus = [d[0] for d in tagged_diagnoses[split_idx:]]
    return train_corpus, test_corpus


def fit_doc2vec(train_corpus, save_path, epochs=10):
    model = gensim.models.doc2vec.Doc2Vec(
        vector_size=50,
        min_count=2,
        epochs=epochs
    )
    model.build_vocab(train_corpus)

    start_time = time.time()
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
    print("--- %s seconds --- gensim train time" % (time.time() - start_time))

    model.save(save_path)


def find_k_closest(centroids, data, k=1, distance_norm=2):
    """
    Arguments:
    ----------
        centroids: (M, d) ndarray
            M - number of clusters
            d - number of data dimensions
        data: (N, d) ndarray
            N - number of data points
        k: int (default 1)
            nearest neighbour to get
        distance_norm: int (default 2)
            1: Hamming distance (x+y)
            2: Euclidean distance (sqrt(x^2 + y^2))
            np.inf: maximum distance in any dimension (max((x,y)))

    Returns:
    -------
        indices: (M,) ndarray
        values: (M, d) ndarray
    """

    kdtree = cKDTree(data)
    distances, indices = kdtree.query(centroids, k, p=distance_norm)
    if k > 1:
        indices = indices[:,-1]
    values = data[indices]
    return indices, values


def some_checks(model):
    print(
        model.infer_vector(
            [
                'ramus', 'circumflexus', 'mit', 'aktuell',
                'durchgängigen', 'bypässen', 'und', 'suffizienten',
                'anastomosen', 'allgemeine', 'atherosklerose',
                'der', 'nieren', 'akute', 'stauung', 'in', 'beiden'
            ]
        )
    )

    start_time = time.time()
    ranks = []
    second_ranks = []
    for doc_id in tqdm(range(len(train_corpus) // 1000)):
        inferred_vector = model.infer_vector(train_corpus[doc_id].words)
        sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
        rank = [docid for docid, sim in sims].index(doc_id)
        ranks.append(rank)

        second_ranks.append(sims[1])

    print(collections.Counter(ranks))  # Results vary between runs due to random seeding and very small corpus
    print("--- %s seconds --- similarity check on train" % (time.time() - start_time))

    doc_id = 0
    print(
        (
            'Document (){}): «{}»\n'.format(
                doc_id, ' '.join(
                    train_corpus[doc_id].words
                )
            )
        )
    )
    inferred_vector = model.infer_vector(train_corpus[doc_id].words)
    sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
    start_time = time.time()

    print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
    for label, index in [('MOST', 0), ('SECOND-MOST', 1), ('MEDIAN', len(sims) // 2), ('LEAST', len(sims) - 1)]:
        print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_corpus[sims[index][0]].words)))

    print("--- %s seconds --- last check" % (time.time() - start_time))


input_path = '/home/giga/dev/python/shipyard/ris/al/data/Patho_2017.xlsx'

output_path = 'output/patho.csv'

# extract_data(input_path, output_path)

df = import_csv(output_path)
diagnoses = df['R_Diagnose'].tolist()
ids = df['id']
tagged_diagnoses = tag_diagnoses(diagnoses, ids)
train_corpus = tagged_diagnoses  # , test_corpus = train_test_split(tagged_diagnoses)

save_path = 'output/doc2vec.model'
# fit_doc2vec(train_corpus, save_path, epochs=40)

model = gensim.models.doc2vec.Doc2Vec.load(save_path)


from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

kmeans_model = KMeans(n_clusters=50, init='k-means++', max_iter=100)
X = kmeans_model.fit(model.docvecs.doctag_syn0)
labels=kmeans_model.labels_.tolist()

l = kmeans_model.fit_predict(model.docvecs.doctag_syn0)
pca = PCA(n_components=2).fit(model.docvecs.doctag_syn0)
datapoint = pca.transform(model.docvecs.doctag_syn0)


plt.figure

label1 = ["#FFFF00", "#008000", "#0000FF", "#800080", "#FF5505", "#00C505", "#5055FF", "#805585"]

import numpy as np
colormap=plt.cm.rainbow(np.linspace(0, 1, len(kmeans_model.cluster_centers_)))
#color = [label1[0][i] for i in labels]

plt.scatter(datapoint[:, 0], datapoint[:, 1], c=colormap[kmeans_model.labels_])

centroids = kmeans_model.cluster_centers_
centroidpoint = pca.transform(centroids)
plt.scatter(centroidpoint[:, 0], centroidpoint[:, 1], marker='^', s=150, c='#000000')
plt.show()


order_ids, values = find_k_closest(centroids, model.docvecs.doctag_syn0)
centroid_ids = ids[order_ids]
print(centroid_ids)

df_centroids = df[df['id'].isin(centroid_ids)]
print(df_centroids)
for id, diagnose in zip(df_centroids['id'], df_centroids['R_Diagnose']):
    with open('output/R_Diagnose/' + id + '.txt', 'w') as f:
        f.write(diagnose)

# for idx in indices:
#     print('------', idx, '-------')
#     inferred_vector = model.infer_vector(train_corpus[idx].words)
#     print(idx, train_corpus[idx].words)
#     sims = model.docvecs.most_similar([inferred_vector], topn=10)
#
#     for sim in sims:
#         print(sim, train_corpus[sim[0]].words)



print('done')