from dash import Dash, html, dcc, dash_table
import plotly.express as px  # (version 4.7.0 or higher)
import plotly.graph_objects as go

import numpy as np
import pandas as pd


from .data import create_dataframe
from .layout import html_layout

def init_dashboard(server):
    dash_app = Dash(
        server=server,
        routes_pathname_prefix="/",
        external_stylesheets=[
            "/static/dist/css/style.min.css",
            "https://fonts.googleapis.com/css?family=Lato",
        ],
    )

    # Load DataFrame
    df = create_dataframe()

    # Custom HTML layout
    dash_app.index_string = html_layout
    # Create Layout
    dash_app.layout = html.Div(
        children=[
            dcc.Graph(
                id="histogram-graph",
                figure={
                    "data": [
                        {
                            "x": df["original_language"],
                            "text": df["original_language"],
                            "customdata": df["movieId"],
                            "name": "Original language",
                            "type": "histogram",
                        }
                    ],
                    "layout": {
                        "title": "Original language",
                        "height": 500,
                        "padding": 150,
                    },
                },
            ),
        ],
        id="dash-container",
    )
    return dash_app.server
