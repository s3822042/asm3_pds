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
server.secret_key = 'super secret key'
server.config['SESSION_TYPE'] = 'filesystem'

app = Dash(
    __name__, server=server, meta_tags=[{"name": "viewport", "content": "width=device-width"}],
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

df['release_date'] = pd.to_datetime(df['release_date'], format='%Y-%m-%d')
df['year'] = df['release_date'].dt.year


# Main
if __name__ == "__main__":
    # Load model
    xgbr_model = pickle.load(open(MODEL_PATH.joinpath("xgbr.pkl"), "rb"))
    print('xgbr model loaded')

    server.run(debug=True)
