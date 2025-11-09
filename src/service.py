import nltk
import pandas as pd
from fontTools.misc.cython import returns
from pandas.core.series import Series
import numpy as np
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns


class SpamModel:
    def __init__(
            self
    ):
        self.DF: pd.DataFrame = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.pipeline: Pipeline = None
        self.is_fitted = False

        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words=stopwords.words("english"),
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        self.LG_algorithm = LogisticRegression(
            C=1.0,
            solver='lbfgs',
            random_state=0,
            max_iter=1000
        )
        self.SVM_algorithm = SVC(
            C=1.0,
            kernel='rbf',
            gamma='scale',
            probability=True,
            random_state=0
        )

    def load_data(self, url: str):
        self.DF = pd.read_csv(
            url,
            encoding="utf-8",
            encoding_errors="replace"
        )
        self.DF = self.DF[["v1", "v2"]]
        X, y = self.DF["v2"], self.DF["v1"]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=0, stratify=y
        )

    @staticmethod
    def preprocessing_text(
            text: str
    ) -> str:
        new_text = text.lower()
        new_text = re.sub(r"[^a-zA-Zа-яА-Я\s]", "", new_text)
        new_text = re.sub(r"\s+", " ", new_text).strip()
        return new_text

    def prepare_data(
            self,
            X_d : Series,
            y_d : Series
    ):
        X = X_d.apply(self.preprocessing_text)
        y = y_d.map({"spam": 1, "ham": 0})
        return X, y

    def fit(self):
        X, y = self.prepare_data(self.X_train, self.y_train)

        self.pipeline = Pipeline(
            [
                ("tfidf", self.vectorizer),
                ("clasifier", self.SVM_algorithm)
            ]
        )
        self.pipeline.fit(X, y)
        self.is_fitted = True

    def predict(self):
        if not self.is_fitted:
            raise NotFitted("Данные не загружены.")

        X, y = self.prepare_data(self.X_test, self.y_test)
        return self.pipeline.predict(X)

    def predict_with_both(self):
        X, y = self.prepare_data(self.X_train, self.y_train)
        data = {
            "Logistic Regression": [],
            "SVM": []
        }
        metrics = ['Accuracy', 'Precision', 'Recall']

        for i, algorithm in enumerate([self.LG_algorithm, self.SVM_algorithm]):
            self.pipeline = Pipeline(
                [
                    ("tfidf", self.vectorizer),
                    ("clasifier", algorithm)
                ]
            )
            self.pipeline.fit(X, y)
            X_test, y_test = self.prepare_data(self.X_test, self.y_test)
            y_pred = self.pipeline.predict(X_test)
            y_predict = Series(y_pred).map({1: "spam", 0: "ham"})
            classif_report = self.get_metrics_report(y_predict)
            data[list(data.keys())[i]] = list(classif_report.values())

        plt.figure(figsize=(10, 8))

        x = np.arange(3)
        width = 0.35
        plt.bar(x - width / 2, data["Logistic Regression"], width, label='Logistic Regression',
                color='darkorange', edgecolor='black', alpha=0.8)
        plt.bar(x + width / 2, data["SVM"], width, label='SVM',
                color='darkgreen', edgecolor='black', alpha=0.8)

        plt.xlabel('Метрики')
        plt.ylabel('Значение')
        plt.title('Сравнение алгоритмов по метрикам качества')
        plt.xticks(x, metrics)
        plt.ylim(0, 1.1)
        plt.legend()
        plt.grid(axis='y', alpha=0.6)

        for i, v in enumerate(data["Logistic Regression"]):
            plt.text(i - width / 2, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        for i, v in enumerate(data["SVM"]):
            plt.text(i + width / 2, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.show()

    def show_confusion_matrix(self, y_predict):
        cm = confusion_matrix(self.y_test, y_predict)
        tn, fp, fn, tp = cm.ravel()
        annotations = [
            [f"TN: {tn}\n(Истинно отрицательные)", f"FP: {fp}\n(Ложно положительные)"],
            [f"FN: {fn}\n(Ложно отрицательные)", f"TP: {tp}\n(Истинно положительные)"]
        ]

        fig, ax = plt.subplots(figsize=(10, 8))

        im = ax.imshow(cm, cmap="Oranges", alpha=0.75)

        for i in range(2):
            for j in range(2):
                text = ax.text(
                    j, i, annotations[i][j],
                    ha="center", va="center",
                    color="black", fontsize=14, weight="bold"
                )

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Предсказан: Не спам", "Предсказан: Спам"])
        ax.set_yticklabels(["Истина: Не спам", "Истина: Спам"])

        plt.title(f"Матрица ошибок", fontsize=20, pad=12)

        plt.tight_layout()
        plt.show()

        return tn, fp, fn, tp

    def get_metrics_report(self, y_predict):
        accuracy = accuracy_score(self.y_test, y_predict)
        classif_report: dict = classification_report(self.y_test, y_predict, output_dict=True)["weighted avg"]
        classif_report.pop("support")
        print(f"Accuracy: {accuracy * 100:.2f}%")
        return classif_report

    def show_metrics_report(self, y_predict):
        classif_report = self.get_metrics_report(y_predict)

        metrics = classif_report.keys()
        values = classif_report.values()
        plt.figure(figsize=(8, 6))
        bars = plt.bar(metrics, values, color=["olive", "coral"])

        for _bar, _value in zip(bars, values):
            plt.text(_bar.get_x() + _bar.get_width() / 2, _bar.get_height() + 0.01,
                     f"{_value:.3f}", ha="center", va="bottom")

        plt.ylim(0, 1.1)
        plt.ylabel("Значение метрики")
        plt.title("Сравнение оценок разных метрик")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.show()

    def show_accuracy(
            self,
            y_predict
    ):
        y_predict = Series(y_predict).map({1: "spam", 0: "ham"})
        self.show_confusion_matrix(y_predict)
        self.show_metrics_report(y_predict)
        return

    def __str__(self):
        return str(self.DF)
