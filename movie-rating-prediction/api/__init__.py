from ddtrace import patch_all
from flask import Flask
from flask_assets import Environment

patch_all()

def init_app():
    app = Flask(__name__, instance_relative_config=False)
    app.config.from_object("config.Config")
    assets = Environment()
    assets.init_app(app)

    with app.app_context():
        from . import routes
        from .assets import compile_static_assets
        from .dash.dashboard import init_dashboard

        app = init_dashboard(app)
        compile_static_assets(assets)
        return app

