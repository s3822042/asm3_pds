from controls import MOVIE_STATUSES
import itertools as it
from dash import Dash, html, dcc
from flask import Flask, flash, request, redirect, jsonify
import copy
from dash.dependencies import Input, Output, State, ClientsideFunction
import datetime as dt
import pathlib
import numpy as np
import pandas as pd
import pickle
import traceback
import plotly.express as px
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


df = pd.read_csv(
    DATA_PATH.joinpath("data.csv"),
    low_memory=False,
)


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

df['year'] = pd.to_datetime(df['year'], format='%Y', errors='coerce')
df = df[df["year"] > dt.datetime(1960, 1, 1)]

layout = dict(
    autosize=True,
    automargin=True,
    margin=dict(l=30, r=30, b=20, t=40),
    hovermode="closest",
    plot_bgcolor="#F9F9F9",
    paper_bgcolor="#F9F9F9",
    legend=dict(font=dict(size=10), orientation="h"),
)


movie_status_options = [
    {"label": str(MOVIE_STATUSES[movie_status]), "value": str(movie_status)}
    for movie_status in MOVIE_STATUSES
]

# Create app layout
app.layout = html.Div(
    [
        # empty Div to trigger javascript file for graph resizing
        html.Div(id="output-clientside"),
        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.H3(
                                    "Visualization Dashboard",
                                    style={"margin-bottom": "0px"},
                                ),
                                html.H5(
                                    "Movie Rating Prediction", style={"margin-top": "0px"}
                                ),
                            ]
                        )
                    ],
                    className="one-half column",
                    id="title",
                ),
            ],
            id="header",
            className="row flex-display",
            style={"margin-bottom": "25px"},
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.P(
                            "Filter by movie release date (or select range in histogram):",
                            className="control_label",
                        ),
                        dcc.RangeSlider(
                            id="year_slider",
                            min=1960,
                            max=2017,
                            value=[1990, 2010],
                            className="dcc_control",
                        ),
                        html.P("Filter by movie status:",
                               className="control_label"),
                        dcc.RadioItems(
                            id="movie_status_selector",
                            options=[
                                {"label": "All ", "value": "all"},
                                {"label": "Released", "value": "Released"},
                                {"label": "Customize ", "value": "custom"},
                            ],
                            value="Released",
                            labelStyle={"display": "inline-block"},
                            className="dcc_control",
                        ),
                        dcc.Dropdown(
                            id="movie_statuses",
                            options=movie_status_options,
                            multi=True,
                            value=list(MOVIE_STATUSES.keys()),
                            className="dcc_control",
                        ),
                    ],
                    className="pretty_container four columns",
                    id="cross-filter-options",
                ),
                html.Div(
                    [
                        html.Div(
                            [dcc.Graph(id="count_graph")],
                            id="countGraphContainer",
                            className="pretty_container",
                        ),
                    ],
                    id="right-column",
                    className="eight columns",
                ),
            ],
            className="row flex-display",
        ),
    ],
    id="mainContainer",
    style={"display": "flex", "flex-direction": "column"},
)


def filter_dataframe(df, movie_statuses, year_slider):
    dff = df[
        df["status"].isin(movie_statuses)
        & (df["year"] > dt.datetime(year_slider[0], 1, 1))
        & (df["year"] < dt.datetime(year_slider[1], 1, 1))
    ]
    return dff


# Create callbacks
app.clientside_callback(
    ClientsideFunction(namespace="clientside", function_name="resize"),
    Output("output-clientside", "children"),
    [Input("count_graph", "figure")],
)


@app.callback(
    Output("movie_statuses", "value"), [Input("movie_status_selector", "value")]
)
def display_status(selector):
    if selector == "all":
        return list(MOVIE_STATUSES.keys())
    elif selector == "Released":
        return ["RL"]
    return []

@app.callback(Output("year_slider", "value"), [Input("count_graph", "selectedData")])
def update_year_slider(count_graph_selected):

    if count_graph_selected is None:
        return [1990, 2010]

    nums = [int(point["pointNumber"])
            for point in count_graph_selected["xgbr_model"]]
    return [min(nums) + 1960, max(nums) + 1961]

@app.callback(
    Output("count_graph", "figure"),
    [
        Input("movie_statuses", "value"),
        Input("year_slider", "value"),
    ],
)
def make_count_figure(movie_statuses, year_slider):

    layout_count = copy.deepcopy(layout)

    dff = filter_dataframe(df, movie_statuses, [1960, 2017])
    g = dff[["movieId", "year"]]
    g.index = g["year"]
    g = g.resample("A").count()

    colors = []
    for i in range(1960, 2018):
        if i >= int(year_slider[0]) and i < int(year_slider[1]):
            colors.append("rgb(123, 199, 255)")
        else:
            colors.append("rgba(123, 199, 255, 0.2)")

    data = [
        dict(
            type="scatter",
            mode="markers",
            x=g.index,
            y=g["movieId"] / 2,
            name="All Movies",
            opacity=0,
            hoverinfo="skip",
        ),
        dict(
            type="bar",
            x=g.index,
            y=g["movieId"],
            name="All Movies",
            marker=dict(color=colors),
        ),
    ]

    layout_count["title"] = "Completed Movies/Year"
    layout_count["dragmode"] = "select"
    layout_count["showlegend"] = False
    layout_count["autosize"] = True

    figure = dict(data=data, layout=layout_count)
    return figure





# Main
if __name__ == "__main__":
    # Load model
    xgbr_model = pickle.load(open(MODEL_PATH.joinpath("xgbr.pkl"), "rb"))
    print("model xgbr loaded")
    server.run(debug=True)
