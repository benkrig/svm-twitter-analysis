from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import sklearn.svm as svm
from sklearn.model_selection import train_test_split
from svm_trainer.util import *
import numpy as np

from util.data import plot_classification_report

BUCKET_NAME = "svmclassifier2019-mlengine"


if __name__ == "__main__":
    df = get_data(nrows=1000, path="svm_trainer/data/stanford140.csv")

    # df = clean_pre(df)
    print(df.head)
    vectorized = TfidfVectorizer(
        min_df=2,
        max_df=0.9,
        ngram_range=(1, 2),
        use_idf=True,
        norm="l2",
        sublinear_tf=False,
    )
    features = vectorized.fit_transform(df.text).toarray()

    labels = df.target
    print(features.shape)
    model = svm.LinearSVC(random_state=42)

    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
        features, labels, df.index, test_size=0.3, random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    test_prediction = vectorized.transform(np.array(["I'm not sure if i like this"]))

    test_pred = model.predict(test_prediction)

    print(test_pred)

    print(classification_report(y_test, y_pred))
    plot_classification_report(
        str(classification_report(y_test, y_pred)), title="2-Gram"
    )
    # write_data(model=model)
