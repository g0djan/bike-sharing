import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

def load_data(path, train=True):
    """Load data from a XLSX File

    Parameters
    ----------
    path: str
        The path to the XLSX file

    train: bool (default True)
        Decide whether or not data are *training data*.
        If True, some random shuffling is applied.

    Return
    ------
    X: numpy.ndarray
        The data as a multi dimensional array of floats
    ids: numpy.ndarray
        A vector of ids for each sample
    """
    df = pd.ExcelFile(path).parse('in')
    df.datetime = df.datetime.apply(lambda dt: dt.hour)
    X = df.values.copy()
    if train:
        np.random.shuffle(X)
        X, labels = X[:, 0:-1].astype(np.float32), X[:, -1]
        return X, labels
    else:
        X, ids = X[:, 1:].astype(np.float32), X[:, 0].astype(str)
        return X, ids


def preprocess_data(X, scaler=None):
    """Preprocess input data by standardise features
    by removing the mean and scaling to unit variance"""
    if not scaler:
        scaler = StandardScaler()
        scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler


def preprocess_labels(labels, encoder=None, categorical=True):
    """Encode labels with values among 0 and `n-classes-1`"""
    if not encoder:
        encoder = LabelEncoder()
        encoder.fit(labels)
    y = encoder.transform(labels).astype(np.int32)
    if categorical:
        y = np_utils.to_categorical(y)
    return y, encoder


def open_categorical(data, columns):
    columns.sort()
    shift = 0
    for column in columns:
        categorical = np_utils.to_categorical(data[:, shift + column])
        data = np.insert(data, [shift + column + 1], categorical, axis=1)
        data = np.delete(data, [shift + column], axis=1)
        shift += categorical.shape[1] - 1
    return data
