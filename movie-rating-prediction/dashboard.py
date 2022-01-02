from dash import Dash, html, dcc
from dash.dependencies import Input, Output, State, ClientsideFunction
import pathlib
import datetime as dt


import numpy as np
import pandas as pd
import pickle
import copy


from controls import MOVIES_STATUSES


PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("data").resolve()
MODEL_PATH = PATH.joinpath("model").resolve()

app = Dash(
    __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}],
)
app.title = "Visualization Dashboard"
server = app.server

movie_status_options = [
    {"label": str(MOVIES_STATUSES[movie_status]), "value": str(movie_status)}
    for movie_status in MOVIES_STATUSES
]

# Load model
# model1 = pickle.load(open(MODEL_PATH.joinpath("model1.pkl"), "rb"))
# model2 = pickle.load(open(MODEL_PATH.joinpath("model2.pkl"), "rb"))
# model3 = pickle.load(open(MODEL_PATH.joinpath("model3.pkl"), "rb"))


df = pd.read_csv(
    DATA_PATH.joinpath("data.csv"),
    low_memory=False,
)

df["release_date"] = pd.to_datetime(df["release_date"])
df = df[df["release_date"] > dt.datetime(1960, 1, 1)]

df = df.sort_values('movieId', ascending=False)
df = df.drop_duplicates(subset='movieId', keep='first')

trim = df[["movieId", "title", "rating"]]
trim.index = trim["movieId"]
dataset = trim.to_dict(orient="index")

# Create global chart template
layout = dict(
    autosize=True,
    automargin=True,
    margin=dict(l=30, r=30, b=20, t=40),
    hovermode="closest",
    plot_bgcolor="#F9F9F9",
    paper_bgcolor="#F9F9F9",
    legend=dict(font=dict(size=10), orientation="h"),
)

# Create app layout
app.layout = html.Div(
    [
        dcc.Store(id="aggregate_data"),
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
                                    "Movie Overview", style={"margin-top": "0px"}
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
                            "Filter by release date (or select range in histogram):",
                            className="control_label",
                        ),
                        dcc.RangeSlider(
                            id="year_slider",
                            min=1960,
                            max=2021,
                            value=[1990, 2010],
                            className="dcc_control",
                        ),
                        html.P("Filter by movie status:", className="control_label"),
                        dcc.RadioItems(
                            id="movie_status_selector",
                            options=[
                                {"label": "All ", "value": "all"},
                                {"label": "Released ", "value": "release"},
                                {"label": "Customize ", "value": "custom"},
                            ],
                            value="release",
                            labelStyle={"display": "inline-block"},
                            className="dcc_control",
                        ),
                        dcc.Dropdown(
                            id="movies_statuses",
                            options=movie_status_options,
                            multi=True,
                            value=list(MOVIES_STATUSES.keys()),
                            className="dcc_control",
                        ),
                    ],
                    className="pretty_container four columns",
                    id="cross-filter-options",
                ),
                html.Div(
                    [
                        html.Div(
                            id="info-container",
                            className="row container-display",
                        ),
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
        html.Div(
            [
                html.Div(
                    [dcc.Graph(id="main_graph")],
                    className="pretty_container seven columns",
                ),
                html.Div(
                    [dcc.Graph(id="individual_graph")],
                    className="pretty_container five columns",
                ),
            ],
            className="row flex-display",
        ),
        html.Div(
            [
                html.Div(
                    [dcc.Graph(id="pie_graph")],
                    className="pretty_container seven columns",
                ),
                html.Div(
                    [dcc.Graph(id="aggregate_graph")],
                    className="pretty_container five columns",
                ),
            ],
            className="row flex-display",
        ),
    ],
    id="mainContainer",
    style={"display": "flex", "flex-direction": "column"},
)

def filter_dataframe(df, movies_statuses, year_slider):
    dff = df[
        df["status"].isin(movies_statuses)
        & (df["release_date"] > dt.datetime(year_slider[0], 1, 1))
        & (df["release_date"] < dt.datetime(year_slider[1], 1, 1))
    ]
    return dff

# Create callbacks
app.clientside_callback(
    ClientsideFunction(namespace="clientside", function_name="resize"),
    Output("output-clientside", "children"),
    [Input("count_graph", "figure")],
)

@app.callback(
    Output("movies_statuses", "value"), [Input("movie_status_selector", "value")]
)
def display_status(selector):
    if selector == "all":
        return list(MOVIES_STATUSES.keys())
    elif selector == "release":
        return ["RL"]
    return []

@app.callback(Output("year_slider", "value"), [Input("count_graph", "selectedData")])
def update_year_slider(count_graph_selected):

    if count_graph_selected is None:
        return [1990, 2010]

    nums = [int(point["pointNumber"]) for point in count_graph_selected["points"]]
    return [min(nums) + 1960, max(nums) + 1961]

@app.callback(
    Output("count_graph", "figure"),
    [
        Input("movies_statuses", "value"),
        Input("year_slider", "value"),
    ],
)
def make_count_figure(movies_statuses,year_slider):

    layout_count = copy.deepcopy(layout)

    dff = filter_dataframe(df, movies_statuses, [1960, 2017])
    g = dff[["movieId", "release_date"]]
    g.index = g["release_date"]
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
    app.run_server(debug=True)
