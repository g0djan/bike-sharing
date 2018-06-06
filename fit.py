import numpy as np
import keras.metrics
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from kaggle_provider import KaggleProvider
from predict import predict


def rmsle(y_true, y_pred):
    return K.sqrt(keras.metrics.mean_squared_logarithmic_error(y_true, y_pred))


def baseline_model():
    model = Sequential()
    model.add(Dense(dims, input_dim=dims, activation='relu'))
    model.add(Dense(dims, activation='relu'))
    model.add(Dense(1, activation='relu'))
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    return model


dims = None


def fit(is_train=True):
    global dims
    provider = KaggleProvider.from_file('train.csv')
    x_train, y_train = provider.load_data(is_train)
    dims = x_train.shape[1]

    # fix random seed for reproducibility
    seed = 42
    np.random.seed(seed)
    regressor = KerasRegressor(build_fn=baseline_model, epochs=10, batch_size=5, verbose=0)
    estimators = []
    estimators.append(('standardize', StandardScaler()))
    estimators.append(('mlp', regressor))
    pipeline = Pipeline(estimators)
    kfold = KFold(n_splits=5, random_state=seed)
    results = cross_val_score(pipeline, x_train, y_train, cv=kfold, scoring='neg_mean_absolute_error')
    print("Standardized: {} ({}) MSE".format(results.mean(), results.std()))

    if not is_train:
        regressor.fit(x_train, y_train)
        predict(regressor)


fit()
#
# from sklearn.model_selection import train_test_split
# from keras.callbacks import EarlyStopping, ModelCheckpoint
# X_train, X_val, Y_train, Y_val = train_test_split(X_train, labels, test_size=0.3, random_state=42)
#
# fBestModel = 'best_model.h5'
# early_stop = EarlyStopping(monitor='val_loss', patience=2, verbose=1)
# best_model = ModelCheckpoint(fBestModel, verbose=0, save_best_only=True)
#
# model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=400,
#           batch_size=128, verbose=True, callbacks=[best_model, early_stop])
