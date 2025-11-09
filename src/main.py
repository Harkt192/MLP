import pandas as pd
from service import SpamModel


pd.options.display.min_rows = 20
pd.options.display.max_rows = 100
pd.options.display.max_columns = None


def main():
    model = SpamModel()
    model.load_data("https://raw.githubusercontent.com/Harkt192/MLP/refs/heads/master/src/dataset/spam.csv")
    model.fit()
    predicted = model.predict()
    model.show_accuracy(predicted)
    model.predict_with_both()


if __name__ == "__main__":
    main()