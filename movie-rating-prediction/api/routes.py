"""Routes for parent Flask app."""
from flask import current_app as app
from flask import render_template

@app.route("/evaluate")
def evaluate():
    return render_template(
        "evaluate.jinja2",
        title="Evaluation Metric",
        # template="evaluation-template",
        body="This is a evaluation page.",
    )

@app.route("/predict")
def predict():
    return render_template(
        "predict.jinja2",
        title="Predict value",
        # template="predict-template",
        body="This is a predict page.",
    )