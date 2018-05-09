import os.path
import pandas as pd

from keras.models import load_model
from src.excel_provider import load_x


def predict():
    model = load_model('nn1.h5')
    x_test = load_x(os.path.join('data', 'test.xlsx'), 'in', None)
    y_test = model.predict(x_test)

    submission_path = os.path.join('data', 'submission.xlsx')
    submission = pd.ExcelFile(submission_path).parse('in')
    submission.cnt = y_test

    writer = pd.ExcelWriter(submission_path)
    submission.to_excel(writer, 'in')
    writer.save()


if __name__ == '__main__':
    predict()
