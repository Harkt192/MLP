import nltk
import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
import nltk


class NotFitted(Exception):
    pass


nltk.download("stopwords")


class SpamModel:
    def __init__(
            self
    ):
        self.DF: pd.DataFrame = None
        self.pipeline: Pipeline = None
        self.is_fitted = False

    def load_data(self):
        self.DF = pd.read_csv(
            f"./dataset/spam.csv",
            encoding="utf-8",
            encoding_errors="replace"
        )[1002:]
        self.DF = self.DF[["v1", "v2"]]

    @staticmethod
    def preprocessing_text(
            text: str
    ) -> str:
        new_text = text.lower()
        new_text = re.sub(r"[^a-zA-Zа-яА-Я\s]", "", new_text)
        new_text = re.sub(r'\s+', ' ', new_text).strip()
        return new_text

    def prepare_data(
            self,
            df: pd.DataFrame
    ):
        df = df.copy()
        df["v1"] = df["v1"].map({"spam": 1, "ham": 0})
        df["v2"] = df["v2"].apply(self.preprocessing_text)

        return df["v2"], df["v1"]

    def fit(self):
        x, y = self.prepare_data(self.DF)
        self.pipeline = Pipeline(
            [("tfidf", TfidfVectorizer(
                max_features=5000,
                stop_words=stopwords.words("english"),
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8
            )),
             ("clasifier", LinearSVC(
                 C=1.0,
                 class_weight="balanced",
                 random_state=42
             ))
            ]
        )
        self.pipeline.fit(x, y)
        self.is_fitted = True

    def predict(
            self,
            test_df: pd.DataFrame
    ):
        if not self.is_fitted:
            raise NotFitted("Данные не загружены.")

        x, y = self.prepare_data(test_df)
        return self.pipeline.predict(x)

    def __str__(self):
        return str(self.DF)

