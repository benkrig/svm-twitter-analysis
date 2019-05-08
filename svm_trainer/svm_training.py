from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import sklearn.svm as svm
from sklearn.model_selection import train_test_split
from svm_trainer.util import *

BUCKET_NAME = 'svmclassifier2019-mlengine'

df = get_data(nrows=1600000)

df = clean_pre(df)
print(df.head)
vectorized = TfidfVectorizer(
    sublinear_tf=True,
    min_df=4,
    max_df=0.90,
    norm="l2",
    ngram_range=(1, 2),
)
features = vectorized.fit_transform(df.text).toarray()
labels = df.target
print(features.shape)
model = svm.LinearSVC(C=1.0, random_state=42, verbose=True, tol=1e-4, loss='hinge')
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
    features, labels, df.index, test_size=0.2, random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
write_data(model=model)
