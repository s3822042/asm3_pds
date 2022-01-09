from dash import Dash, html, dcc
from flask import Flask
from dash.dependencies import Input, Output
import pathlib
import datetime as dt
import plotly.express as px


import numpy as np
import pandas as pd
import pickle
import copy


from controls import MOVIES_STATUSES


PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("data").resolve()
MODEL_PATH = PATH.joinpath("models").resolve()

server = Flask(__name__)

app = Dash(
    __name__, server=server, meta_tags=[{"name": "viewport", "content": "width=device-width"}],
)



# Routing
@server.route("/evaluate")
def Evaluate():
    return "Evaluate"

@server.route("/predict", methods=['POST'])
def Predict():
    return "Predict"

app.title = "Visualization Dashboard"



df = pd.read_csv(
    DATA_PATH.joinpath("data.csv"),
    low_memory=False,
)

available_status = df['status'].unique()

counting = df['status'].value_counts()
year = df['year'].unique()

app.layout = html.Div([
    html.Div([
        html.Div([
            dcc.Dropdown(
                id='status',
                options=[{'label': i, 'value': i} for i in available_status],
                value='Released'
            ),
        ])
    ]),
    html.Div([
        dcc.Graph(
            id="status_graph",
        )
    ]),
])

# @app.callback(
#     Output('status_graph', 'figure'),
#     Input('status', 'value'),
# )
# def update_status_graph(year_value):
#     dff = df[df['year'] == year_value]

#     fig = px.bar(x=dff[dff['year']])

#     return fig


# Main
if __name__ == "__main__":
    # Load model
    dt_model = pickle.load(open(MODEL_PATH.joinpath("dt.pkl"), "rb"))
    print('dt model loaded')
    lgbm_model = pickle.load(open(MODEL_PATH.joinpath("lgbm.pkl"), "rb"))
    print('lgbm model loaded')
    # xgbr_model = pickle.load(open(MODEL_PATH.joinpath("xgbr.pkl"), "rb"))

    app.run_server(debug=True)
