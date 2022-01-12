from get_data import all_movies, ml_data
from dash import Dash, html, dcc
from dash.dependencies import Input, Output
from flask import Flask, flash, request, redirect, jsonify
import pathlib
import numpy as np
import pandas as pd
import pickle
import traceback

# Chart

import plotly.express as px
import plotly.graph_objs as go
# ML Algorithm

from sklearn.model_selection import train_test_split
from sklearn import tree
import lightgbm as lgb
import xgboost as xgb
# Evaluation metric

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score


PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("data").resolve()
MODEL_PATH = PATH.joinpath("models").resolve()

server = Flask(__name__)
server.secret_key = 'super secret key'
server.config['SESSION_TYPE'] = 'filesystem'

app = Dash(
    __name__, server=server, meta_tags=[{"name": "viewport", "content": "width=device-width"}],
)

df = all_movies.copy()
ml = ml_data.copy()

X = ml.budget.values[:, None]
y = ml['rating']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42)

MODELS = {
    'Decision Tree': tree.DecisionTreeClassifier,
    'Light GBM': lgb.LGBMClassifier,
    'XG Boost': xgb.XGBClassifier
}

# Routing
@server.route("/evaluate", methods=['POST','GET'])
def evaluate():
    if request.method == 'GET':
       return "Please use Postman to input the test file and get the evaluation metric"
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        try:
            file = request.files['file']
            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)
            df_test_file =  pd.read_csv(file, delimiter=',')
            X_test = df_test_file.drop(columns='rating').copy()
            y_test = df_test_file['rating'].copy()
            y_predict_xgbr = xgbr_model.predict(X_test)
            # XG Boost  Algorithm
            xgbr_rmse = np.sqrt(mean_squared_error(y_test, y_predict_xgbr))
            xgbr_mae = mean_absolute_error(y_test, y_predict_xgbr)
            xgbr_r2 = r2_score(y_test, y_predict_xgbr)
            xgbr_accuracy = accuracy_score(y_test, y_predict_xgbr)

            metric_xgbr = [
                {
                    "RMSE": xgbr_rmse,
                    "MAE:": xgbr_mae,
                    "R2:": xgbr_r2,
                    "Accuracy:": xgbr_accuracy,
                }
            ]

            return jsonify({
               "Evaluation Metric for XG Boost Algorithm":metric_xgbr,
            })

        except:
           return jsonify({
               "trace": traceback.format_exc()
               })


@server.route("/predict", methods=['POST','GET'])
def predict():
    if request.method == 'GET':
       return "Please use Postman to input test set file and get the predicted rating value"
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        try:
            file = request.files['file']
            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)
            df_test_file =  pd.read_csv(file, delimiter=',')
            X_test = df_test_file.drop(columns='rating').copy()
            prediction_xgbr = xgbr_model.predict(X_test)

             # Take the first value of prediction
            output_xgbr = prediction_xgbr[0]

            return jsonify({
               "Prediction for rating using XG Boost Algorithm":str(output_xgbr)
            })

        except:
           return jsonify({
               "trace": traceback.format_exc()
               })


app.title = "Visualization Dashboard"

df1 = df.sort_values(by='popularity', ascending=False).head(5)

unique = list(df.original_language.unique())
list_ratio = []
for each in unique:
    x = df[df["original_language"] == each]
    ratio_popularity = sum(x.popularity)/len(x)
    list_ratio.append(ratio_popularity)

df2 = pd.DataFrame({"language": unique, "ratio": list_ratio})
new_index = (df2.ratio.sort_values(ascending=False).head(5)).index.values
sorted_data = df2.reindex(new_index)

# Create app layout
app.layout = html.Div(
    [
        # empty Div to trigger javascript file for ml_graph resizing
        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.H3(
                                    "Visualization Dashboard",
                                    style={
                                        "margin-bottom": "0px",
                                    },
                                ),
                                html.H5(
                                    "Movie Rating Prediction",
                                     style={
                                         "margin-top": "0px"
                                    },
                                ),
                            ]
                        )
                    ],
                    className="column",
                    id="title",
                ),
            ],
            id="header",
            className="row flex-display",
            style={"margin-bottom": "25px"},
        ),
        html.Div(
            [
                html.P("Select Model:"),
                dcc.Dropdown(
                    id='model-name',
                    options=[{'label': x, 'value': x}
                             for x in MODELS],
                    value='Decision Tree',
                    clearable=False
                ),
            ],
            className="four columns",
            style={
                "margin-top": "10px",
            },
            id="cross-filter-options",
        ),
        html.Div(
            [
                html.Div(
                    [dcc.Graph(id="ml_graph")],
                    id="GraphContainer",
                    className="pretty_container",
                ),
            ],
            id="right-column",
            className="twelve columns",
        ),
        html.Div(
            [
                html.Div(
                    [dcc.Graph(
                        figure=px.bar(df1, x="popularity", y="title")
                    )],
                    className="pretty_container six columns",
                ),
                html.Div(
                    [dcc.Graph(
                        figure=px.bar(sorted_data, x="language", y="ratio")
                    )],
                    className="pretty_container six columns",
                ),
            ],
            className="row flex-display",
        ),
    ],
    id="mainContainer",
    style={"display": "flex", "flex-direction": "column"},
)


@app.callback(
    Output("ml_graph", "figure"),
    [Input('model-name', "value")])
def train_and_display(name):
    model = MODELS[name]()
    model.fit(X_train, y_train)

    x_range = np.linspace(X.min(), X.max(), 100)
    y_range = model.predict(x_range.reshape(-1, 1))

    fig = go.Figure([
        go.Scatter(x=X_train.squeeze(), y=y_train,
                   name='train', mode='markers'),
        go.Scatter(x=X_test.squeeze(), y=y_test,
                   name='test', mode='markers'),
        go.Scatter(x=x_range, y=y_range,
                   name='prediction')
    ])

    return fig


# Main
if __name__ == "__main__":
    # Load model
    xgbr_model = pickle.load(open(MODEL_PATH.joinpath("xgbr.pkl"), "rb"))
    print("model xgbr loaded")
    server.run(debug=True)
