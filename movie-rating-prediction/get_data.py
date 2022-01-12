import pandas as pd
import pathlib
import itertools as it

PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("data").resolve()

all_movies = pd.read_csv(
    DATA_PATH.joinpath("data.csv"),
    low_memory=False,
)
all_movies['genres'].fillna(value='', inplace=True)

genres = set(list(it.chain.from_iterable(
    [g.split(',') for g in all_movies.genres if g])))

variable_labels = {
    'rating': 'Numeric Rating',
    'popularity': 'Popularity',
    'revenue': 'Dollars',
    'year': 'Year',
    'runtime': 'Length (minutes)'
}
