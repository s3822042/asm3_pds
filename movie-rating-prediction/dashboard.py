from dash import Dash, html, dcc
from flask import Flask, flash, request, redirect, jsonify, render_template
from dash.dependencies import Input, Output
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

app = Dash(
    __name__, server=server, meta_tags=[{"name": "viewport", "content": "width=device-width"}],
)

# Routing
@server.route("/evaluate", methods=['POST','GET'])
def evaluate():
    if request.method == 'GET':
       return "Evaluation page"
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
            y_predict_dt = dt_model.predict(X_test)
            y_predict_lgbm = lgbm_model.predict(X_test)
            # Decision Tree Algorithm
            dt_rmse = np.sqrt(mean_squared_error(y_test, y_predict_dt))
            dt_mae = mean_absolute_error(y_test, y_predict_dt)
            dt_r2 = r2_score(y_test, y_predict_dt)
            dt_accuracy = accuracy_score(y_test, y_predict_dt)
            # Light GBM Algorithm
            lgbm_rmse = np.sqrt(mean_squared_error(y_test, y_predict_lgbm))
            lgbm_mae = mean_absolute_error(y_test, y_predict_lgbm)
            lgbm_r2 = r2_score(y_test, y_predict_lgbm)
            lgbm_accuracy = accuracy_score(y_test, y_predict_lgbm)

            metric_dt = [
                {
                    "RMSE": dt_rmse,
                    "MAE:": dt_mae,
                    "R2:": dt_r2,
                    "Accuracy:": dt_accuracy,
                }
            ]

            metric_lgbm = [
                {
                    "RMSE": lgbm_rmse,
                    "MAE:": lgbm_mae,
                    "R2:": lgbm_r2,
                    "Accuracy:": lgbm_accuracy,
                }
            ]

            return jsonify({
               "Evaluation Metric for Decision Tree Algorithm":metric_dt,
               "Evaluation Metric for Light GBM Algorithm":metric_lgbm,
            })

        except:
           return jsonify({
               "trace": traceback.format_exc()
               })



@server.route("/predict", methods=['POST','GET'])
def predict():
    if request.method == 'GET':
       return "Prediction page"
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
            prediction_dt = list(dt_model.predict(X_test))
            prediction_lgbm = lgbm_model.predict(X_test)
        #   prediction_xgbr = xgbr_model.predict(X_test)

             # Take the first value of prediction
            output_dt = prediction_dt[0]
            output_lgbm = prediction_lgbm[0]

            return jsonify({
               "Prediction for rating using Decision Tree Algorithm":str(output_dt),
               "Prediction for rating using Light GBM Algorithm":str(output_lgbm)
            })

        except:
           return jsonify({
               "trace": traceback.format_exc()
               })


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
    # print('xgbr model loaded')
    server.secret_key = 'super secret key'
    server.config['SESSION_TYPE'] = 'filesystem'

    app.run_server(debug=True)
