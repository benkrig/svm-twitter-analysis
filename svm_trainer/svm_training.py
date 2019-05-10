from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import sklearn.svm as svm
from sklearn.model_selection import train_test_split
from svm_trainer.util import *
import numpy as np
import matplotlib.pyplot as plt
import itertools

BUCKET_NAME = "svmclassifier2019-mlengine"


def plot_classification_report(classificationReport,
                               title='Classification report',
                               cmap='RdBu'):

    classificationReport = classificationReport.replace('\n\n', '\n')
    classificationReport = classificationReport.replace(' / ', '/')
    lines = classificationReport.split('\n')

    classes, plotMat, support, class_names = [], [], [], []
    for line in lines[1:3]:  # if you don't want avg/total result, then change [1:] into [1:-1]
        t = line.strip().split()
        if len(t) < 2:
            continue
        classes.append(t[0])
        v = [float(x) for x in t[1: len(t) - 1]]
        support.append(int(t[-1]))
        class_names.append(t[0])
        plotMat.append(v)

    plotMat = np.array(plotMat)
    xticklabels = ['Precision', 'Recall', 'F1-score']
    yticklabels = ['{0}'.format(class_names[idx], sup)
                   for idx, sup in enumerate(support)]

    plt.imshow(plotMat, interpolation='nearest', cmap=cmap, aspect='auto')
    plt.title(title)
    plt.colorbar()
    plt.xticks(np.arange(3), xticklabels, rotation=45)
    plt.yticks(np.arange(len(classes)), yticklabels)

    upper_thresh = plotMat.min() + (plotMat.max() - plotMat.min()) / 10 * 8
    lower_thresh = plotMat.min() + (plotMat.max() - plotMat.min()) / 10 * 2
    for i, j in itertools.product(range(plotMat.shape[0]), range(plotMat.shape[1])):
        plt.text(j, i, format(plotMat[i, j], '.2f'),
                 horizontalalignment="center",
                 color="white" if (plotMat[i, j] > upper_thresh or plotMat[i, j] < lower_thresh) else "black")

    plt.ylabel('Metrics')
    plt.xlabel('Classes')
    plt.tight_layout()


if __name__ == "__main__":
    df = get_data(nrows=10000)

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
    y_pred = model.predict(X_test)

    test_prediction = vectorized.transform(np.array(["I'm not sure if i like this"]))

    test_pred = model.predict(test_prediction)

    print(test_pred)

    print(classification_report(y_test, y_pred))
    plot_classification_report(str(classification_report(y_test, y_pred)), title="2-Gram")
    plt.show()
    #write_data(model=model)
