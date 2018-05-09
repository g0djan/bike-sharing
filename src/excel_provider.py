import pandas as pd


def normalize(data_frame):
    return (data_frame - data_frame.mean()) / data_frame.std() #(data_frame.max() - data_frame.min())  # data_frame.std()


def load_data(filename, tablename, columns):
    data = pd.ExcelFile(filename).parse(tablename)
    data.datetime = data.datetime.apply(lambda dt: dt.timestamp())
    y_train = _get_y(data)
    x_train = _get_x(data, columns)
    return x_train, y_train


def load_x(filename, tablename, columns):
    data = pd.ExcelFile(filename).parse(tablename)
    data.datetime = data.datetime.apply(lambda dt: dt.timestamp())
    x_train = _get_x(data, columns)
    return x_train


def _get_y(data_frame):
    return data_frame.cnt.values


def _get_x(data_frame, columns):
    table = normalize(data_frame)
    matrix = table.as_matrix()
    return matrix[:, :columns]
