import os.path
from keras.models import Sequential
from keras.layers import Dense
from src.excel_provider import load_data


def create_regression_nn(x_train, y_train):
    model = Sequential()
    model.add(Dense(9, activation='relu', input_shape=(9,)))
    model.add(Dense(7, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    info = model.fit(x_train, y_train, epochs=20, batch_size=1, verbose=0)
    print(info.history)
    return model


def fit():
    x_train, y_train = load_data(os.path.join('data', 'train.xlsx'), 'in', -3)
    model = create_regression_nn(x_train, y_train)
    model.save("nn1.h5")


if __name__ == '__main__':
    fit()
