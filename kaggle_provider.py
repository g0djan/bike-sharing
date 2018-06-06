import pandas as pd
import numpy as np
from numpy import ndarray
from pandas import DataFrame, Series
from datetime import datetime
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler

DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S'


def column_to_vector(series: Series) -> ndarray:
    length = series.size
    return series.values.reshape(length, 1)


def preprocess_data(x):
    scaler = StandardScaler()
    scaler.fit(x)
    x = scaler.transform(x)
    return x


def _handle_season(matrix: ndarray, df: DataFrame, column_name) -> ndarray:
    categories = to_categorical(df[column_name])[:, 1:]
    return np.append(matrix, categories, axis=1)


def _copy_column(matrix: ndarray, df: DataFrame, column_name) -> ndarray:
    if matrix is None:
        return column_to_vector(df[column_name])
    return np.append(matrix, column_to_vector(df[column_name]), axis=1)


class KaggleProvider:
    def __init__(self, df: DataFrame, dt_format: str):
        self.df = df
        self.matrix = None
        self.dt_format = dt_format
        self.default_handler = _copy_column
        self.column_handlers = {'datetime': self._handle_datetime,
                                'season': _handle_season,
                                'weather': _handle_season}

    def load_data(self, is_train=True):
        matrix = self.get_matrix()
        if is_train:
            np.random.shuffle(self.matrix)
            return matrix[:, 0:-3].astype(np.float32), matrix[:, -1]  # TODO: think about 2 last columns
        return matrix.astype(np.float32)

    def get_matrix(self):
        if self.matrix is not None:
            return self.matrix
        for column in self.df:
            self.matrix = self.handle(column)
        return self.matrix

    def handle(self, column):
        handler = self.column_handlers.get(column, self.default_handler)
        return handler(self.matrix, self.df, column)

    @staticmethod
    def from_file(path: str, dt_format=DATETIME_FORMAT):
        return KaggleProvider(pd.read_csv(path), dt_format)

    def _handle_datetime(self, matrix: ndarray, df: DataFrame, column_name) -> ndarray:
        column = df[column_name].apply(lambda s: datetime.strptime(s, self.dt_format).hour)
        if matrix is None:
            return column_to_vector(column)
        return np.append(matrix, column_to_vector(column), axis=1)
