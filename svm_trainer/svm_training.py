from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import sklearn.svm as svm
from sklearn.model_selection import train_test_split
from svm_trainer.util import *
import numpy as np
import matplotlib.pyplot as plt
import itertools

BUCKET_NAME = "svmclassifier2019-mlengine"

df = get_data(nrows=1600000)

df = clean_pre(df)
print(df.head)
vectorized = TfidfVectorizer(
    min_df=2, max_df=0.9, ngram_range=(1, 2), use_idf=True, norm='l2', sublinear_tf=False
)
features = vectorized.fit_transform(df.text).toarray()

labels = df.target
print(features.shape)
model = svm.LinearSVC(random_state=42)

X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
    features, labels, df.index, test_size=0.3, random_state=42
)
model.fit(X_train, y_train)

write_data(model=model)
