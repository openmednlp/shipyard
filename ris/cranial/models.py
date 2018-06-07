import bedrock
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE
from sklearn.naive_bayes import BernoulliNB
from tpot import TPOTClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFwe, f_classif
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from tpot.builtins import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def tpot_search(X_train, X_test, y_train, y_test, target_column):
    pipeline_optimizer = TPOTClassifier(
        generations=30,
        population_size=30,
        cv=5,
        random_state=42,
        verbosity=2,
        config_dict='TPOT sparse'
    )

    pipeline_optimizer.fit(X_train, y_train)
    # print(pipeline_optimizer.score(X_test, y_test))

    pipeline_optimizer.export('output/tpot_exported_pipeline_' + target_column + '.py')

# from sklearn.metrics import classification_report
# target_names = ['class 0', 'class 1', 'class 2']
# print(classification_report(y_true, y_pred, target_names=target_names))


def grid_search(x, y):
    bernoulli = BernoulliNB()
    parameters = {
        'alpha': np.arange(0.01, 2, 0.1),
        'binarize': np.arange(0, 1, 0.1),
        'fit_prior': [True, False]
    }
    grid = GridSearchCV(bernoulli, parameters)
    grid.fit(x, y)


def cross_bnb(x, y):
    clf = BernoulliNB()
    skf = StratifiedKFold(n_splits=10)
    # skf.get_n_splits(x,y)

    scores = cross_val_score(clf, x, y, cv=skf)
    print(scores)
    n, bins, patches = plt.hist(scores, 5)
    plt.show()


def bnb(X_train, X_test, y_train, y_test, persist_path=None, do_viz=False):
    # Feature selection before SVC
    # print('BeronulliNB')

    pipeline = make_pipeline(
        RFE(
            estimator=ExtraTreesClassifier(
                criterion="gini",
                max_features=0.1,
                n_estimators=100
            ),
            step=0.4
        ),
        BernoulliNB(alpha=0.01, fit_prior=False)
    )

    pipeline.fit(X_train, y_train)
    predicted = pipeline.predict(X_test)

    if do_viz:
        bedrock.viz.show_stats(y_test, predicted)

    from sklearn.metrics import accuracy_score

    print('Accuracy', accuracy_score(y_test, predicted))

    bedrock.common.save_pickle(pipeline, persist_path)

    # returns train and test classified
    return pipeline


tpot_pipelines = {
    'ICB': make_pipeline(
        TfidfVectorizer(),
        KNeighborsClassifier(n_neighbors=11, p=1, weights="distance")), # LinearSVC(C=10.0, dual=True, loss="hinge", penalty="l2", tol=0.001),
    'fracture': make_pipeline(
        TfidfVectorizer(),
        LinearSVC(C=10.0, dual=False, loss="squared_hinge", penalty="l1", tol=0.01)),
    'hydrocephalus': make_pipeline(
        TfidfVectorizer(),
        SelectFwe(score_func=f_classif, alpha=0.006),
        RandomForestClassifier(bootstrap=False, criterion="gini", max_features=0.3, min_samples_leaf=2, min_samples_split=17, n_estimators=100, )
    ),
    'midline': make_pipeline(
        TfidfVectorizer(),
        RandomForestClassifier(bootstrap=False, criterion="entropy", max_features=0.25, min_samples_leaf=2, min_samples_split=20, n_estimators=100)),
    'vessels': make_pipeline(
        TfidfVectorizer(),
        RFE(estimator=ExtraTreesClassifier(criterion="entropy", max_features=0.15000000000000002, n_estimators=100), step=0.6500000000000001),
        #OneHotEncoder(minimum_fraction=0.25),
        LinearSVC(C=0.5, dual=True, loss="hinge", penalty="l2", tol=0.001)
    )
}


def tpot_models(X_train, X_test, y_train, y_test, target_column, persist_path=None, do_viz=False):

    pipeline = tpot_pipelines[target_column]
    pipeline.fit(X_train, y_train)
    y_hat = pipeline.predict(X_test)

    if do_viz:
        bedrock.viz.show_stats(y_test, y_hat)

    bedrock.common.save_pickle(pipeline, persist_path)


def get_model_map():
    return {
        'BernoulliNB': make_pipeline(
            TfidfVectorizer(),
            BernoulliNB()
        ),
        'LinearSVC': make_pipeline(
            TfidfVectorizer(),
            LinearSVC(),
        ),
        'RandomForestClassifier': make_pipeline(
            TfidfVectorizer(),
            RandomForestClassifier()
        ),
        'ExtraTreesClassifier': make_pipeline(
            TfidfVectorizer(),
            ExtraTreesClassifier()
        )
    }

def model_testing(
        x,
        y,
        label_name,
        corpus_size,
        seed=42,
        splits=10,
        repeats=20):
    # TODO: Trying to disable a warning that is out of my control
    import pandas as pd
    pd.options.mode.chained_assignment = None

    model_map = get_model_map()

    all_models = model_map  # {**model_map, **tpot_pipelines}

    scoring = ['accuracy', 'f1', 'precision', 'recall']
    # results = []
    res = {
        'label': [],
        'corpus_size': [],
        'model': [],
        'accuracy_scores': [],
        'accuracy_mean': [],
        'accuracy_std': [],
        'accuracy_max': [],
        'accuracy_min': [],
        'precision_scores': [],
        'precision_mean': [],
        'recall_scores': [],
        'recall_mean': [],
        'f1_scores': [],
        'f1_mean': [],

    }

    for key in all_models.keys():
        print(key)
        model = all_models[key]
        kfold = RepeatedKFold(n_splits=splits, n_repeats=repeats, random_state=seed)
        cv_results = cross_validate(
            model,
            x,
            y,
            cv=kfold,
            scoring=scoring,
            return_train_score=False,
        )

        #cv_results = cross_val_score(model, x, y, cv=kfold, scoring=scoring)
        # results.append(cv_results)

        res['label'].append(label_name)
        res['corpus_size'].append(corpus_size)
        res['model'].append(key)

        accuracy_scores = cv_results['test_accuracy']
        res['accuracy_scores'].append(list(accuracy_scores))
        res['accuracy_mean'].append(accuracy_scores.mean())
        res['accuracy_std'].append(accuracy_scores.std())
        res['accuracy_min'].append(accuracy_scores.min())
        res['accuracy_max'].append(accuracy_scores.max())

        f1_scores = cv_results['test_f1']
        res['f1_scores'].append(list(f1_scores))
        res['f1_mean'].append(f1_scores.mean())

        precision_scores = cv_results['test_precision']
        res['precision_scores'].append(list(precision_scores))
        res['precision_mean'].append(precision_scores.mean())

        precision_scores = cv_results['test_recall']
        res['recall_scores'].append(list(precision_scores))
        res['recall_mean'].append(precision_scores.mean())

        # msg = "%s: mean %f (std %f) [max: %f, min: %f]" % (
        #     key, cv_results.mean(),
        #     cv_results.std(),
        #     cv_results.max(),
        #     cv_results.min()
        # )
        # print(msg)

    import pandas as pd
    df = pd.DataFrame(res)

    return df