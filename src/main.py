import pandas as pd
from service import SpamModel


pd.options.display.min_rows = 30
pd.options.display.max_rows = 100
pd.options.display.max_columns = None


def main():
    model = SpamModel()
    model.load_data()
    model.fit()

    test_df = pd.read_csv(
        "./dataset/spam.csv",
        encoding="utf-8",
        encoding_errors="replace"
    )[:1001]
    test_df = test_df[["v1", "v2"]]
    predicted = model.predict(test_df)
    test_df["Predict"] = pd.Series(predicted).map({1: "spam", 0: "ham"})
    print(test_df)


if __name__ == "__main__":
    main()
