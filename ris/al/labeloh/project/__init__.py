from flask import Flask
from labeloh.model import db, User, Role
import datetime
from flask_user import UserManager

app = Flask(__name__, instance_relative_config=True)
app.config.from_object('labeloh.default_settings')
app.config.from_pyfile('application.cfg', silent=False)

db.init_app(app)
db.app = app

# Setup Flask-User and specify the User dataset-model
user_manager = UserManager(app, db, User)

# Create all database tables
db.create_all()

# Create 'member@example.com' user with no roles
if not User.query.filter(User.email == 'member@example.com').first():
    user = User(
        username='member',
        email='member@example.com',
        email_confirmed_at=datetime.datetime.utcnow(),
        password=user_manager.hash_password('Password1'),
    )
    db.session.add(user)
    db.session.commit()

# Create 'admin@example.com' user with 'Admin' and 'Agent' roles
if not User.query.filter(User.email == 'admin@example.com').first():
    user = User(
        username='admin',
        email='admin@example.com',
        email_confirmed_at=datetime.datetime.utcnow(),
        password=user_manager.hash_password('Password1'),
    )
    user.roles.append(Role(name='Admin'))
    user.roles.append(Role(name='Agent'))
    db.session.add(user)
    db.session.commit()


from labeloh.project.api.datasets import data_blueprint
from labeloh.project.api.machine_learning.models import models_blueprint
from labeloh.project.api.machine_learning.algorithms import algorithms_blueprint

from labeloh.project.api.samples import al_blueprint
from labeloh.project.api.storage import storage_blueprint
from labeloh.project.site.routes import mod as site_mod

app.register_blueprint(data_blueprint, url_prefix='/api')
app.register_blueprint(algorithms_blueprint, url_prefix='/api')
app.register_blueprint(models_blueprint, url_prefix='/api')
app.register_blueprint(al_blueprint, url_prefix='/api')
app.register_blueprint(storage_blueprint, url_prefix='/api')
app.register_blueprint(site_mod, url_prefix='/site')
