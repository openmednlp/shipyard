from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV


def create_algorithm_dict(uid, func, name, desc, creator, other):
    return {
        'id': uid,
        'func': func,
        'name': name,
        'description': desc,
        'creator': creator,
        'other': other
    }


def train_rfc(X, y):
    pipeline = make_pipeline(
        TfidfVectorizer(),
        RandomForestClassifier()
    )
    pipeline.fit(X, y)
    return pipeline


def train_cv_rfc(X, y):
    pipeline = make_pipeline(
        TfidfVectorizer(),
        RandomForestClassifier()
    )

    parameters = {
        'tfidfvectorizer__use_idf': (True, False),
        'randomforestclassifier__n_estimators': [5, 10, 20],
        'randomforestclassifier__min_samples_split': [2, 3, 5]
    }

    cv = GridSearchCV(pipeline, cv=3, n_jobs=4, param_grid=parameters)
    cv.fit(X, y)
    return cv


def train_svc(X, y):
    pipeline = make_pipeline(
        TfidfVectorizer(),
        LinearSVC()
    )

    pipeline.fit(X, y)
    return pipeline


def train_cv_svc(X, y):
    pipeline = make_pipeline(
        TfidfVectorizer(),
        LinearSVC()
    )
    parameters = {
        'tfidfvectorizer__use_idf': (True, False),
        'linearsvc__C': [1, 10, 100]
    }

    cv = GridSearchCV(pipeline, cv=3, n_jobs=4, param_grid=parameters)
    cv.fit(X, y)
    return cv


def get_algorithms():
    return [
        create_algorithm_dict(
            0,
            train_svc,
            'Default',
            'Basic Linear SVC',
            'admin',
            'Let\'s check what needs to ne added here'),
        create_algorithm_dict(
            1,
            train_rfc,
            'Basic Random Forest',
            'Just a basic RFC',
            'admin',
            'Let\'s check what needs to ne added here'),
        create_algorithm_dict(
            2,
            train_svc,
            'SVC',
            'Basic Linear SVC',
            'admin',
            'Let\'s check what needs to ne added here'),
        create_algorithm_dict(
            3,
            train_cv_rfc,
            'Grid Search RFC',
            'Grid Search optimized RFC',
            'admin',
            'Let\'s check what needs to ne added here'),
        create_algorithm_dict(
            4,
            train_cv_svc,
            'Grid Search Linear SVC',
            'Grid Search optimized Linear SVC',
            'admin',
            'Let\'s check what needs to ne added here'),

    ]

algorithms = get_algorithms()

functions_dict = {
    a['id']: a['func']
    for a in algorithms
}

algorithms_info = {}
for algorithm in algorithms:
    del algorithm['func']
    uid = algorithm['id']
    algorithms_info[uid] = algorithm

