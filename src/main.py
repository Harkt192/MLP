import pandas as pd
from service import SpamModel


pd.options.display.min_rows = 10
pd.options.display.max_rows = 100
pd.options.display.max_columns = None


def reformat_text(text: str):
    text = text[9:-1]
    return text


def main():
    model = SpamModel()
    model.load_data()
    model.fit()

    df = pd.read_csv("./dataset/test_spam.csv")
    test_df = pd.DataFrame()
    test_df["v1"] = df["spam"].map({1: "spam", 0: "ham"})
    test_df["v2"] = df["text"].apply(reformat_text)
    a = list(model.predict(test_df))
    print(len(a))


if __name__ == "__main__":
    main()
