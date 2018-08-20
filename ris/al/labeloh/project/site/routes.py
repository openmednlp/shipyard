from flask import Blueprint, render_template, request, redirect, flash, jsonify
from flask import current_app, url_for
from flask import Response

from flask_wtf.file import FileField, FileRequired, FileAllowed
from flask_user import login_required, roles_required

from wtforms import StringField, SelectField
from wtforms.validators import DataRequired, Length

from flask_wtf import FlaskForm
from werkzeug.utils import secure_filename

import requests

from pandas import DataFrame
from pandas.io.json import json_normalize

import json


mod = Blueprint('site', __name__, template_folder='templates')

# TODO: Needs to be split in multiple files - blueprints already in place so it's simple


def _get_link(links, rel_value):
    for l in links:
        if l['rel'] == rel_value:
            return l['href']
    raise ValueError('No file relation found in the links list')


@mod.route('/')
def root():
    return render_template('site/index.html')


# The Members page is only accessible to authenticated users
@mod.route('/members')
@login_required  # Use of @login_required decorator
def member():
    print('Member.')
    return render_template('site/index.html')


# The Admin page requires an 'Admin' role.
@mod.route('/admin')
@roles_required('Admin')  # Use of @roles_required decorator
def admin():
    print('Admin!')
    return render_template('site/index.html')


class DatasetForm(FlaskForm):
    name = StringField(
        'Dataset Name',
        [Length(min=5, max=25)]
    )
    description = StringField(
        'Dataset Description',
        [Length(min=5, max=35)]
    )
    file = FileField(
        'Dataset File',
        [FileRequired()]
    )


class PredictForm(FlaskForm):
    file = FileField(
        'Dataset File',
        [FileRequired()]
    )


@mod.route('/datasets', methods=['POST'])
def datasets_post():
    api_url = current_app.config['API_URI']
    datasets_api_url = api_url + '/api/v1/datasets'
    storage_api_url = api_url + '/api/v1/storage'

    form = DatasetForm()

    if form.validate_on_submit():
        # Don't forget the csrf, or it will fail.
        file = request.files['file']
        if not secure_filename(file.name):
            # TODO make this better
            flash('Wrong file name', 'error')
            return render_template("upload_file.html", form=form, url=url_for('site.datasets'))

        r = requests.post(
            storage_api_url,
            file.read(),
            headers={'Content-Type': 'application/octet-stream'}
        )

        dataset_json = {
            'name': form.name.data,
            'description': form.description.data,
            'file_id': int(r.text)
        }
        requests.post(datasets_api_url, json=dataset_json)

        flash('Thanks for uploading the file')
        return redirect(url_for('site.datasets'))

    return redirect(url_for('site.datasets')) # fix the checking part


@mod.route('/datasets', methods=['GET'])
def datasets():
    api_url = current_app.config['API_URI']
    datasets_api_url = api_url + '/api/v1/datasets'

    start = request.args.get('start')
    limit = request.args.get('limit')
    if not start:
        start = 1
    else:
        start = int(start)
    if not limit:
        limit = 5
    else:
        limit = int(limit)

    form = DatasetForm()

    # Don't init with request.form
    # flask-wtf does this for you

    r = requests.get(
        '{}?start={}&limit={}'.format(
            datasets_api_url, start, limit)
    )
    data = json.loads(r.text)
    datasets_df = json_normalize(data['content'])
    urls = [
        url_for('site.dataset', dataset_id=dataset_id)
        for dataset_id
        in datasets_df['id']
    ]
    names = datasets_df['name']

    prev_attributes = [
        l['href']
        for l in data['links']
        if l['rel'] == 'previous'
    ][0].split('?')[-1]

    prev_url = None
    if prev_attributes:
        prev_url = '{}?{}'.format(
            request.path,
            prev_attributes
        )

    next_attributes = [
        l['href']
        for l in data['links']
        if l['rel'] == 'next'
    ][0].split('?')[-1]

    next_url = None
    if next_attributes:
        next_url = '{}?{}'.format(
            request.path,
            next_attributes
        )

    return render_template(
        'upload_file.html',
        form=form,
        url=url_for('site.datasets'),
        links=zip(urls, names),
        prev_url=prev_url if prev_url else None,
        next_url=next_url if next_url else None
    )



@mod.route('/datasets/<int:dataset_id>')
def dataset(dataset_id):

    api_url = current_app.config['API_URI']
    url = api_url + '/api/v1/datasets/' + str(dataset_id)
    r = requests.get(url)

    json_data = json.loads(r.content)
    url = _get_link(json_data['links'], 'file')
    delete_url = url_for('site.dataset_delete', dataset_id=dataset_id) #TODO: local url and then from that page 2 actions - delete from datasets and then from storage (or make a job for that) _get_link(json_data['links'], 'delete') # requests.delete()
    return '''
    <p>
    Id: {}</br>
    Name: {}</br>
    Description:<br/> 
    {}</br>
    <a href={}>link</a> <a href={}>delete</a></p>
    '''.format(
        json_data['content']['id'],
        json_data['content']['name'],
        json_data['content']['description'],
        url,
        delete_url
    )


@mod.route('/datasets/<int:dataset_id>/delete')
def dataset_delete(dataset_id):
    api_url = current_app.config['API_URI']
    dataset_url = api_url + '/api/v1/datasets/' + str(dataset_id)
    r = requests.get(dataset_url)

    json_data = json.loads(r.content)
    file_url = _get_link(json_data['links'], 'file')

    r_dataset = requests.delete(dataset_url)
    r_file = requests.delete(current_app.config['API_URI'] + file_url)

    return 'Delete is not implemented on the API side atm' # TODO: make real response


@mod.route('/models', methods=['GET'])
def models_get():
    api_url = current_app.config['API_URI']
    models_api_url = api_url + '/api/v1/machine-learning/models'
    r = requests.get(models_api_url)

    data = json.loads(r.text)
    if not data['content']:
        return 'De nada'

    models_df = json_normalize(data['content'])

    urls = [
        url_for('site.dataset', model_id=model_id)
        for model_id
        in models_df['id']
    ]
    names = models_df['name']

    downlaod_section = render_template(
        "url_list.html",
        links=zip(urls, names)
    )

    return "<h1>Fake page</h1>"


@mod.route('/models', methods=['POST'])
def models_post():
    return 'Karamba, you cannot do that! Go to the machine learning site section to train the model.'


@mod.route('/models/<int:model_id>')
def model(model_id):
    api_url = current_app.config['API_URI']
    url = api_url + '/api/v1/machine-learning/models/' + str(model_id)
    r = requests.get(url)

    json_data = json.loads(r.content)
    url = _get_link(json_data['links'], 'file')
    # url = api_url + '/api/v1/storage/{}'.format(json_data['file_id'])

    return '''
    <p>
    Id: {}</br>
    Name: {}</br>
    Description:<br/> 
    {}</br>
    <a href={}>link</a></p>
    '''.format(
        json_data['content']['id'],
        json_data['content']['name'],
        json_data['content']['description'],
        url
    )


class TrainForm(FlaskForm):
    dataset = SelectField(label='Dataset', coerce=str)
    algorithm = SelectField(label='Algorithm', coerce=str)


@mod.route('/ml', methods=['GET'])
def ml():
    urls = [
        url_for('site.ml_train'),
        url_for('site.ml_predict')
    ]
    names = [
        'train',
        'predict'
    ]
    return render_template('ml.html', links=zip(urls, names))


@mod.route('/ml/train', methods=['GET', 'POST'])
def ml_train():
    api_url = current_app.config['API_URI']
    # storage_api_url = api_url + '/api/v1/storage'

    dataset_url = api_url + '/api/v1/datasets'
    r = requests.get(dataset_url)
    datasets = json.loads(r.content)['content']

    dataset_options = []
    for c in datasets:
        dataset_options.append(
            (
                str(c['id']),
                c['name']
            )
        )

    url = api_url + '/api/v1/machine-learning/algorithms'
    r = requests.get(url)
    algorithms = json.loads(r.content)['content']
    algorithm_options = []
    for a in algorithms:
        algorithm_options.append(
            (str(a['id']), a['name'])
        )

    form = TrainForm()
    form.algorithm.choices = algorithm_options
    form.dataset.choices = dataset_options

    if form.validate_on_submit():
        dataset_id = form.dataset.data
        algorithm_id = form.algorithm.data

        model_api_url = api_url + "/api/v1/machine-learning/models"

        json_request = {
            "dataset_id": dataset_id,
            "algorithm_id": algorithm_id
        }

        r = requests.post(
            model_api_url,
            json=json_request
        )

        flash(r.json())

        return redirect(url_for('site.ml_train'))

    return render_template(
        'train_form.html',
        form=form,
        url=url_for('site.ml_train')
    )

    # return render_template(
    #     'ml_all.html',
    #     dataset_form=dataset_form,
    #     dataset_storage_url=url_for('site.datasets'),
    #     model_options=[],
    #     dataset_options=dataset_options,
    #     algorithm_options=algorithm_options,
    #     predict_form=predict_form,
    #     model_url='None'
    # )


class PredictForm(FlaskForm):
    model = SelectField(label='Model', coerce=str)
    dataset_file = FileField('Dataset', [FileRequired()])


@mod.route('/ml/predict', methods=['GET', 'POST'])
def ml_predict():
    models_api_url = current_app.config['MODELS_API']
    r = requests.get(models_api_url)
    models = json.loads(r.content)['content']
    options = [
        (
            str(m['id']),
            'data-{}-algo-{}'.format(
                m['dataset_id'],
                m['algorithm_id']
            )
        ) for m in models
    ]

    form = PredictForm()
    form.model.choices = options
    if form.validate_on_submit():

        # Don't forget the csrf, or it will fail.
        file = request.files['dataset_file']
        if not secure_filename(file.name):
            # TODO make this better
            flash('Wrong file name', 'error')
            return render_template(
                "predict_form.html",
                form=form,
                url=url_for('site.ml_predict')
            )

        api_url = '{}/{}'.format(
                models_api_url,
                form.model.data
        )

        r = requests.post(
            api_url,
            file.read(),
            headers={'Content-Type': 'application/octet-stream'}
        )

        result = r.json()['content']

        df = DataFrame({'X': result['X'], 'y': result['y']})
        csv = df.to_csv()

        return Response(
            csv,
            mimetype="text/csv",
            headers={"Content-disposition":
                         "attachment; filename=file.csv"})


    return render_template(
        'predict_form.html',
        form=form,
        url=url_for('site.ml_predict')
    )


@mod.route('/labeling', methods=['GET', 'POST'])
def labeling():
    return "<h1>Fake page</h1>"
