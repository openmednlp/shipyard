from models import *
from dataset import *
from embeddings import *

from sklearn.model_selection import GroupKFold
from os.path import join

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
    subsample_sizes = range(min_row_count, max_row_limit + 1, interval)
    for row_limit in subsample_sizes:
        df = limit_df(df_all, row_limit)
        for target_column, label_target_sections in zip(target_columns, target_sections_map):
            print(':: ' + target_column + ' ::')

            #x, x_test, y, y_test = get_data(df, target_column, target_section)

            df_filtered = filter_df_by_sections(df, target_column, label_target_sections)
            x = df_filtered['sentence']
            y = df_filtered[target_column]
            groups = df_filtered['accession_id']
            # TODO make it 1/0 split istead of 0.7/03
            #x_vec, x_vec_test, y_vec, y_vec_test = vectorize_dataset(x, x_test, y, y_test, stratify=False)
            # from scipy.sparse import vstack
            # x_vec_both = vstack((x_vec, x_vec_test))
            # y_vec_both = y_vec.append(y_vec_test)

            # cv = RepeatedKFold(2, 2, 42)
            cv = GroupKFold(10)
            # cv = DataFrameCV(df_filtered, 'accession_id', n_splits=10)

            test_results_df = model_testing(
                x,
                y,
                groups,
                label_name=target_column,
                corpus_size=row_limit,
                cv=cv)

            # cv = GroupKFold(10)
            # group_results_df = accession_level_model_testing(
            #     x,
            #     y,
            #     cv,
            #     groups,
            #     )

            # test_results_df['gac_scores'] = group_results_df['gac_scores']
            # test_results_df['gac_mean'] = group_results_df['gac_mean']

            overall_result_df = overall_result_df.append(test_results_df)

            header_template = 'Algorithm comparison | Label: {} | Corpus size: {}'
            viz_header = header_template.format(target_column, row_limit)
            viz.box_plot(
                test_results_df['model'],
                test_results_df['accuracy_scores'],
                viz_header,
                show_plot=False,
                persist=True,
                x_label='Algorithm',
                y_label='Accuracy'
            )

    overall_result_df.to_csv('output/result.csv')


import viz


def vizardry(
        plot_type,
        score_types=['accuracy_scores', 'f1_scores', 'gac_scores'],
        x_label='Number of segments',
        y_label=''):

    label_dict = {
        'accuracy_scores': 'Segment level accuracy',
        'f1_scores': 'F1',
        'gac_scores': 'Report level accuracy'
    }

    df = pd.read_csv('output/result.csv') #,  dtype={'accuracy_scores': np.})

    for score_type in score_types:
        df[score_type] = df[score_type].astype(np.ndarray).apply(lambda x: eval(x))

    if plot_type == 'line_report_accuracy_comparison':
        df_group = df.groupby(['label'])
        print(df_group)

        for label, scores in df_group:
            grouped_scores = scores.groupby(['model'])
            model_dict = {}

            fields = ['corpus_size', 'accuracy_mean', 'gac_mean']
            for model, scores in grouped_scores:
                model_dict[model] = scores[fields].sort_values('corpus_size')

            print(label, model_dict, sep=': ')

            model_names = list(model_dict.keys())
            x = model_dict[model_names[0]]['corpus_size']
            ys = [model_dict[key]['accuracy_mean'] for key in model_names]

            # Segment level
            viz.line_plots(
                x,
                ys,
                model_names,
                label,
                False,
                True,
                'Corpus size',
                'Mean segment level accuracy',
                'line_comparison_segment'
            )

            # Report level
            ys = [model_dict[key]['gac_mean'] for key in model_names]
            viz.line_plots(
                x,
                ys,
                model_names,
                label,
                False,
                True,
                'Corpus size',
                'Mean report level accuracy',
                'line_comparison_report'
            )

        return

    df_group = df.groupby(['model', 'label'])
    print(df_group)

    for alg_label_pair, pair_scores in df_group:
        algorithm_name, label_name = alg_label_pair
        if plot_type == 'box':
            for score_type in score_types:
                header = '{}_{}_{}'.format(
                    algorithm_name,
                    label_name,
                    score_type
                )
                viz.box_plot(
                     pair_scores['corpus_size'].values,
                     results=pair_scores[score_type].values,
                     header=header,
                     show_plot=False,
                     persist=True,
                     x_label=x_label,
                     y_label= y_label if y_label else label_dict[score_type],
                     save_dir_name=plot_type
                 )
        elif plot_type == 'line_segment_accuracy':
            results = [
                pair_scores['accuracy_min'].values,
                pair_scores['accuracy_max'].values,
                pair_scores['accuracy_mean'].values
            ]

            header = '{}_{}_{}'.format(
                algorithm_name,
                label_name,
                'accuracy_min_max_mean'
            )

            viz.line_plot(
                pair_scores['corpus_size'].values,
                results=results,
                header=header,
                show_plot=False,
                persist=True,
                x_label=x_label,
                y_label='Segment level accuracy',
                save_dir_name=plot_type
            )
        elif plot_type == 'line_report_accuracy':
            results = [
                pair_scores['gac_scores'].apply(min),
                pair_scores['gac_scores'].apply(max),
                pair_scores['gac_mean'].values
            ]

            header = '{}_{}_{}'.format(
                algorithm_name,
                label_name,
                'report_accuracy_min_max_mean'
            )

            viz.line_plot(
                pair_scores['corpus_size'].values,
                results=results,
                header=header,
                show_plot=False,
                persist=True,
                x_label=x_label,
                y_label='Report level accuracy',
                save_dir_name=plot_type
            )
        else:
            raise ValueError('Plot type cannot be ' + str(plot_type))


def run_linear_svm(csv_path, row_limit):
    df_all = csv_to_dataset(csv_path)
    df_train = limit_df(df_all, row_limit)
    models = []
    # predictions = []
    df_res = df_all
    for target_column, target_sections in zip(target_columns, target_sections_map):
        print(':: ' + target_column + ' ::')

        df_filtered = filter_df_by_sections(df_train, target_column, target_sections)
        x = df_filtered['sentence']
        y = df_filtered[target_column]

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




def run_document_lod(input_csv_path, output_csv_path='output/lod_doc.csv'):
    df = pd.read_csv(input_csv_path, encoding='utf-8', low_memory=False)
    df_grouped = df.groupby('accession_id').any()
    df_grouped.to_csv(output_csv_path)
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

from sklearn.model_selection._split import _BaseKFold


class DataFrameCV(_BaseKFold):
    def __init__(self, df, split_field, n_splits, random_state=None, shuffle=False):
        super().__init__(n_splits, shuffle, random_state)
        self.df = df
        self.split_field = split_field
        self.train_indices, self.test_indices = self.cv_by_field(
            self.df,
            fold_count=self.n_splits,
            field=self.split_field,
            reset_index=False
        )


    def cv_by_field(self, df, fold_count, field, reset_index=False):
        from random import shuffle

        if reset_index:
            df = df.reset_index()

        df_grouped = df.groupby(field)
        distinct_field_values = list(df_grouped.groups.keys())

        # group_max_targets = list(df_grouped[target_field].max())

        shuffle(distinct_field_values)

        field_value_folds = [
            distinct_field_values[idx::fold_count]
            for idx
            in range(fold_count)
        ]

        index_folds = []
        for fold in field_value_folds:
            df_fold_indices = [
                idx
                for fv
                in fold
                for idx
                in df[df[field] == fv].index.values
            ]

            index_folds.append(df_fold_indices)

        train_fold_ids = []
        test_fold_ids = []

        for fold_test_idx in range(fold_count):
            train_folds = [
                idx
                for fold_idx
                in range(fold_count)
                if fold_idx != fold_test_idx
                for idx
                in index_folds[fold_idx]
            ]
            train_fold_ids.append(train_folds)

            test_fold = [idx for idx in index_folds[fold_test_idx]]
            test_fold_ids.append(test_fold)

        return train_fold_ids, test_fold_ids

    def _iter_test_indices(self, X=None, y=None, groups=None):
        test_indices = self.test_indices

        for fold_test_indices in test_indices:
            yield fold_test_indices

    def _iter_test_masks(self, X=None, y=None, groups=None):
        for test_index in self._iter_test_indices(X, y, groups):
            # test_mask = np.zeros(len(X), dtype=np.bool)
            # test_mask[test_index] = True
            yield X.index.isin(test_index)


if __name__ == '__main__':
    data_abs_path = '/home/giga/dev/python/shipyard/ris/cranial/data/csv'
    file_name = 'cranial_sentences_05312018_NEU.csv'
    csv_path = join(data_abs_path, file_name)


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

    vizardry(plot_type='line_report_accuracy_comparison')
    #vizardry(plot_type='box')
    # vizardry(plot_type='line_report_accuracy')
    # vizardry(plot_type='line_segment_accuracy')


    # run_linear_svm(csv_path, 8008)
    # run_document_lod('output/linear_svm_predicted_result.csv', 'output/lod_doc.csv')


    # run_linear_svm_doc_level(csv_path, 8008)

    # df = pd.DataFrame(
    #     {
    #         'a': [1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5],
    #         'b': [11, 22, 22, 33, 33, 33, 44, 44, 44, 44, 55, 55, 55, 55, 55]
    #     }
    # )
    # tr,te = cv_by_field(df, 3, 'a')
    #
    # print(tr)
    # print(te)

    # sentences = df['sentence']
    # import viz
    # import embeddings
    # tokenized_sentences = [s.split() for s in sentences if len(s)>5]
    # x_w2v, words = embeddings.word2vec(tokenized_sentences)
    # viz.tsne_run(x_w2v)


