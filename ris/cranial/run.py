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

    if 'accuracy_comparison' in plot_type:
        df_group = df.groupby(['label'])
        print(df_group)

        for label, scores in df_group:
            grouped_scores = scores.groupby(['model'])
            model_dict = {}

            fields = [
                'corpus_size',
                'accuracy_mean',
                'gac_mean',
                'accuracy_scores',
                'gac_scores'
            ]

            for model, scores in grouped_scores:
                model_dict[model] = scores[fields].sort_values('corpus_size')

            print(label, model_dict, sep=': ')

            model_names = list(model_dict.keys())
            x = model_dict[model_names[0]]['corpus_size']

            # Segment level
            if 'line' in plot_type:
                ys = [model_dict[key]['accuracy_mean'] for key in model_names]
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
            elif 'box' in plot_type:
                ys = [model_dict[key]['accuracy_scores'] for key in model_names]
                for corpus_id in range(len(x)):
                    viz.box_plot(
                        names=model_names,
                        results=[ys_per_corpus.iloc[corpus_id] for ys_per_corpus in ys],
                        header='box comparison segment {}'.format(x.iloc[corpus_id]),
                        show_plot=False,
                        persist=True,
                        x_label='Corpus size',
                        y_label='Segment level accuracy',
                        save_dir_name='box_segment_comparison'
                    )

            # Report level
            if 'line' in plot_type:
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
            elif 'box' in plot_type:
                ys = [model_dict[key]['gac_scores'] for key in model_names]
                for corpus_id in range(len(x)):
                    viz.box_plot(
                        names=model_names,
                        results=[ys_per_corpus.iloc[corpus_id] for ys_per_corpus in ys],
                        header='box comparison report {}'.format(x.iloc[corpus_id]),
                        show_plot=False,
                        persist=True,
                        x_label='Corpus size',
                        y_label='Report level accuracy',
                        save_dir_name='box_report_comparison'
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
        linearsvm = get_model_map()['SVM']
        model = linearsvm.fit(x, y)
        models.append(model)

        # Predict
        df_filtered = df_all[df_all['section_id'].isin(target_sections)]
        df_filtered_mapped = map_label(df_filtered, target_column)

        x_all = df_filtered_mapped['sentence']
        y_hat = model.predict(x_all)

        df_filtered_mapped['predicted_'+target_column] = y_hat
        df_res = pd.concat([df_res, df_filtered_mapped['predicted_'+target_column]], axis=1)

    file_name = 'linear_svm_predicted_result.csv'
    df_res.to_csv('output/' + file_name)


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


def calculate_p_value(
        calculated_values_path='/home/giga/dev/python/shipyard/ris/cranial/output/linear_svm_predicted_result.csv',
        patient_list_file_path='/home/giga/dev/python/shipyard/ris/cranial/data/csv/patientlist_07152018.xlsx',
        tab_name='Tabelle1'):

    patient_list_df = pd.read_excel(
        patient_list_file_path,
        tab_name,
        usecols='C,I,J,K,L,M'
    )
    accession_numbers = patient_list_df['accession number']

    patient_list_df.rename(
        index=str,
        columns={
            'accession number': 'accession_id',
            'intracranial bleeding in Beurteilung; 0=not mentioned; 1=no; 5=yes': 'ICB',
            'fracture in Beurteilung': 'fracture',
            'Liquorzirkulation in Befund/Beurteilung': 'hydrocephalus',
            '       midline in Befund/Beurteilung': 'midline',
            'vessels in Befund/Beurteilung': 'vessels'
        },
        inplace=True
    )

    print(patient_list_df.sample(1))

    fields = [
        'accession_id',
        'ICB',
        'fracture',
        'hydrocephalus',
        'midline',
        'vessels'
    ]
    patient_list_columns_df = patient_list_df[fields]

    patient_list_positive_count = patient_list_columns_df.astype(bool).sum(axis=0)
    patient_list_count = patient_list_columns_df.count()

    predicted_df = pd.read_csv(calculated_values_path, low_memory=False)
    predicted_fields = [
        'accession_id',
        'predicted_ICB',
        'predicted_fracture',
        'predicted_hydrocephalus',
        'predicted_midline',
        'predicted_vessels'
    ]
    predicted_columns_df = predicted_df[predicted_fields]

    rename_dict=dict(zip(predicted_fields, fields))
    predicted_columns_df.rename(
        index=str,
        columns=rename_dict,
        inplace=True
    )

    predicted_columns_df.loc[:, 'accession_id'] = predicted_columns_df['accession_id'].apply(lambda x: int(x[:-4]))

    predicted_for_patients_df = predicted_columns_df[
        predicted_columns_df['accession_id'].isin(list(accession_numbers))
    ]

    predicted_for_patients_no_nan_df = predicted_for_patients_df.fillna(0)

    predicted_grouped = predicted_for_patients_no_nan_df.groupby('accession_id').agg('max')
    predicted_sum = predicted_grouped.sum().apply(int)
    predicted_count = predicted_grouped.count().apply(int)
    print(
        'predicted sum:\n{}\n predicted count:\n{}\n'.format(
            predicted_sum,
            predicted_count
        )
    )

    all_grouped = predicted_columns_df.groupby('accession_id').agg('max')
    all_sum = all_grouped.sum().apply(int)
    all_count = all_grouped.count().apply(int)
    print(
        'all sum:\n{}\n all count:\n{}\n'.format(
            all_sum,
            all_count
        )
    )

    missing_data = accession_numbers[~accession_numbers.isin(predicted_for_patients_df['accession_id'])]
    print('The missing data for:\n', missing_data)

    from scipy.stats import chisquare

    # positive_f_obs = predicted_sum[0] / predicted_count[0]
    # negative_f_obs = 1 - positive_obs
    #
    # positive_f_exp = all_sum[0] / all_count[0]
    # negative_f_exp = 1 - positive_f_exp



    p_value_info = {
        'label': [],
        'observed present': [],
        'observed not present': [],
        'observed all': [],
        'expected present': [],
        'expected not present': [],
        'total expected present': [],
        'total expected not present': [],
        'total expected': [],
        'critical value': [],
        'p-value': [],
        'difference present': [],
        'abs difference present': [],
        'difference not present': [],
        'abs difference not present': [],
        'p<=0.10': [],
        'p<=0.05': [],
        'p<=0.005': []

    }

    p_value_paper = {
        'label': [],
        'p_man_vs_all_aut': [],
        'p_man_vs_sample_aut': [],
        'critical_man_vs_all_aut': [],
        'critical_man_vs_sample_aut': [],
    }

    print('patient_list_positive_count\n', patient_list_positive_count)

    for idx in fields[1:]:
        sample_aut_positive = predicted_sum[idx]
        sample_aut_negative = predicted_count[idx] - sample_aut_positive

        expected_positive = predicted_count[idx] * all_sum[idx]/all_count[idx]
        expected_negative = predicted_count[idx] - expected_positive

        p = chisquare(
            f_obs=[sample_aut_positive, sample_aut_negative],
            f_exp=[expected_positive, expected_negative]  # [positive_f_exp, negative_f_exp]
        )

        p_value_info['label'].append(idx)
        p_value_info['observed present'].append(predicted_sum[idx])

        observed_not_present = predicted_count[idx] - predicted_sum[idx]
        p_value_info['observed not present'].append(observed_not_present)
        p_value_info['observed all'].append(predicted_count[idx])
        p_value_info['expected present'].append(expected_positive)
        p_value_info['expected not present'].append(expected_negative)
        p_value_info['total expected present'].append(all_sum[idx])
        p_value_info['total expected not present'].append(all_count[idx] - all_sum[idx])
        p_value_info['total expected'].append(all_count[idx])
        p_value_info['critical value'].append(p[0])
        p_value_info['p-value'].append(p[1])

        diff_present = expected_positive - predicted_sum[idx]
        p_value_info['difference present'].append(diff_present)
        p_value_info['abs difference present'].append(abs(diff_present))

        diff_not_present = expected_negative - observed_not_present
        p_value_info['difference not present'].append(diff_not_present)
        p_value_info['abs difference not present'].append(abs(diff_not_present))

        p_value_info['p<=0.10'].append('reject' if p[1] <= 0.1 else 'fail to reject')
        p_value_info['p<=0.05'].append('reject' if p[1] <= 0.05 else 'fail to reject')
        p_value_info['p<=0.005'].append('reject' if p[1] <= 0.005 else 'fail to reject')

        print(
            (
                '-------------------\n'
                'idx:\t{}\n'
                'present:\t{}\n'
                'expected present:\t{}\n'
                'p-value:\t{:3f}\n'
                'total\t{}'
            ).format(
                idx,
                predicted_sum[idx],
                expected_positive,
                p[1],
                predicted_count[idx]
            )
        )

        print(p)

        manual_positive = patient_list_positive_count[idx]
        manual_negative = patient_list_count[idx] - manual_positive

        p_value_paper['label'].append(idx)
        p_man_vs_all_aut = chisquare(
            f_obs=[manual_positive, manual_negative],
            f_exp=[expected_positive, expected_negative]
        )
        p_value_paper['p_man_vs_all_aut'].append(p_man_vs_all_aut[1])
        p_value_paper['critical_man_vs_all_aut'].append(p_man_vs_all_aut[0])


        p_man_vs_sample_aut = chisquare(
            f_obs=[manual_positive, manual_negative],
            f_exp=[sample_aut_positive, sample_aut_negative]
        )
        p_value_paper['p_man_vs_sample_aut'].append(p_man_vs_sample_aut[1])
        p_value_paper['critical_man_vs_sample_aut'].append(p_man_vs_sample_aut[0])


    p_df = pd.DataFrame(p_value_info)
    p_df.to_csv('output/p_value.csv')

    print(p_value_paper)
    p_paper_df = pd.DataFrame(p_value_paper)
    p_paper_df.to_csv('output/p_value_paper.csv', float_format='%.10f')

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

    # calculate_p_value()

    # vizardry(plot_type='line_accuracy_comparison')
    vizardry(plot_type='box_accuracy_comparison')

    # vizardry(plot_type='box')
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


