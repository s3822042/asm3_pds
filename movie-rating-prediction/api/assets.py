"""Compile static assets."""
from flask import current_app as app
from flask_assets import Bundle

def compile_static_assets(assets):
    assets.auto_build = True
    assets.debug = False
    css_bundle = Bundle(
        "css/*.css",
        filters="cssmin",
        output="dist/css/style.min.css",
        extra={"rel": "stylesheet/css"},
    )
    assets.register("css_all", css_bundle)
    if app.config["FLASK_ENV"] == "development":
        css_bundle.build()
    return assets