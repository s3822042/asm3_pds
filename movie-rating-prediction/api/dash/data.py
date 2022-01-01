import numpy as np
import pandas as pd


def create_dataframe():
    df = pd.read_csv("data/final_data.csv", delimiter=",")
    df.drop(columns=["overview", "poster_path"], inplace=True)
    return df