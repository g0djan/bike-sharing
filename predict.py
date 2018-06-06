import pandas as pd
from keras.models import load_model
from kaggle_provider import KaggleProvider, preprocess_data


def predict(model):
    output = 'submission.csv'
    provider = KaggleProvider.from_file('test.csv', '%m/%d/%y %I:%M %p')
    x_test = provider.load_data(is_train=False)
    x_test = preprocess_data(x_test)
    y_test = model.predict(x_test)
    df = pd.read_csv(output)
    df['count'] = y_test
    df.to_csv(output, index=False)

#
# model = load_model('save.h5')
# predict(model)