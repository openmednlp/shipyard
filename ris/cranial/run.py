from models import *
from dataset import *
from embeddings import *

target_columns = [
    'ICB',
    'fracture',
    'hydrocephalus',
    'midline',
    'vessels'
]
target_sections_map = [
    ['beurteilung'],  # ICB
    ['beurteilung'],  # fracture
    ['befund', 'beurteilung'],  # hydrocephalus
    ['befund', 'beurteilung'],  # midline
    ['befund', 'beurteilung']  # vessels
]


def spacy_stuff():
    import spacy
    from spacy import displacy
    t = '''
    Etwas eingeschränkte Beurteilbarkeit der Halsgefässe im REA-Protokoll, soweit kein Anhalt für eine Dissektion.
    '''

    nlp = spacy.load('de')
    doc = nlp(t)

    graph_html = displacy.render(doc, style='dep', minify=True, page=True)
    print(graph_html)
    [
        print(c.text, c.root.text, c.root.dep_, sep=', ')
        for c in doc.noun_chunks
    ]


def run_test_models(csv_path, min_row_count, max_row_limit, interval, splits, repeats):
    import viz
    overall_result_df = pd.DataFrame()
    df_all = csv_to_dataset(csv_path, max_row_limit)
    for row_limit in range(min_row_count, max_row_limit+1, interval):
        df = limit_df(df_all, row_limit)
        for target_column, label_target_sections in zip(target_columns, target_sections_map):
            print(':: ' + target_column + ' ::')

            #x, x_test, y, y_test = get_data(df, target_column, target_section)

            x, y = get_x_and_y(df, target_column, label_target_sections)

            # TODO make it 1/0 split istead of 0.7/03
            #x_vec, x_vec_test, y_vec, y_vec_test = vectorize_dataset(x, x_test, y, y_test, stratify=False)
            # from scipy.sparse import vstack
            # x_vec_both = vstack((x_vec, x_vec_test))
            # y_vec_both = y_vec.append(y_vec_test)

            test_results_df = model_testing(
                x,
                y,
                label_name=target_column,
                corpus_size=row_limit,
                splits=splits,
                repeats=repeats
            )
            overall_result_df = overall_result_df.append(test_results_df)

            header_template = 'Algorithm comparison | Label: {} | Corpus size: {}'
            viz_header = header_template.format(target_column, row_limit)
            viz.box_plot(
                test_results_df['model'],
                test_results_df['accuracy_scores'],
                viz_header,
                show_plot=False,
                persist=True
            )

    overall_result_df.to_csv('output/result.csv')


def run_linear_svm(csv_path, row_limit):
    df_all = csv_to_dataset(csv_path)
    df_train = limit_df(df_all, row_limit)
    models = []
    # predictions = []
    df_res = df_all
    for target_column, target_sections in zip(target_columns, target_sections_map):
        print(':: ' + target_column + ' ::')

        x, y = get_x_and_y(df_train, target_column, target_sections)

        # Train model
        linearsvm = get_model_map()['LinearSVC']
        model = linearsvm.fit(x,y)
        models.append(model)

        # Predict
        df_filtered = df_all[df_all['section_id'].isin(target_sections)]
        df_filtered_mapped = map_label(df_filtered, target_column)

        x_all = df_filtered_mapped['sentence']
        y_hat = model.predict(x_all)

        df_filtered_mapped['predicted_'+target_column] = y_hat
        df_res = pd.concat([df_res, df_filtered_mapped['predicted_'+target_column]], axis=1)

    df_res.to_csv('output/linear_svm_predicted_result.csv')


def run_document_lod(csv_path):
    df = pd.read_csv(csv_path, encoding='utf-8', low_memory=False)
    df_grouped = df.groupby('accession_id').any()
    return df_grouped



def run_classificators(df):
    for target_column, target_section in zip(target_columns, target_sections_map):
        print(':: ' + target_column + ' ::')
        x, x_test, y, y_test = get_data(df, target_column, target_section)
        x_vec, x_vec_test, y_vec, y_vec_test = vectorize_dataset(x, x_test, y, y_test, stratify=False)
        # bnb(x_vec, x_vec_test, y_vec, y_vec_test, do_viz=True)

        grid_search(x_vec, y_vec)

        # tpot_search(x_vec, x_vec_test, y_vec, y_vec_test, target_column)
        # tpot_models(x_vec, x_vec_test, y_vec, y_vec_test, target_column, do_viz=True)


def run_regex(df):
    from regex_patterns import fraktur_map
    target_column = target_columns[1]
    target_section = target_sections_map[1]
    x, x_test, y, y_test = get_data(df, target_column, target_section)

    x = x.append(x_test)
    y = y.append(y_test)

    findings = regex_label_sentences(x, fraktur_map)
    y_hat = [0 if f is None else 1 for f in findings]

    bedrock.viz.show_stats(y, y_hat)

    comp_df = pd.DataFrame(
        {
            'sentence': x,
            'ground_truth': y,
            'prediction': y_hat
        }
    )

    comp_df.to_csv('comparison.csv')


if __name__ == '__main__':
    data_abs_path = '/home/giga/dev/python/shipyard/ris/cranial/data/csv/'
    file_name = 'cranial_sentences_05312018_NEU.csv'
    csv_path = data_abs_path + file_name


    # run_classificators(df)
    # run_regex(df)

    # run_test_models(
    #     csv_path,
    #     min_row_count=1008,
    #     max_row_limit=8008,
    #     interval=1000,
    #     splits=10,
    #     repeats=10
    # )

    # run_linear_svm(csv_path, 8008)
    run_document_lod('output/linear_svm_predicted_result.csv')

    # sentences = df['sentence']
    # import viz
    # import embeddings
    # tokenized_sentences = [s.split() for s in sentences if len(s)>5]
    # x_w2v, words = embeddings.word2vec(tokenized_sentences)
    # viz.tsne_run(x_w2v)

