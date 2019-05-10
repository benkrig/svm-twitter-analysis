import matplotlib.pyplot as plt
import itertools
import numpy as np


def plot_classification_report(
    cls_report, title="Classification report", cmap="RdBu"
):

    cls_report = cls_report.replace("\n\n", "\n")
    cls_report = cls_report.replace(" / ", "/")
    lines = cls_report.split("\n")

    classes, plotMat, support, class_names = [], [], [], []
    for line in lines[
        1:3
    ]:  # if you don't want avg/total result, then change [1:] into [1:-1]
        t = line.strip().split()
        if len(t) < 2:
            continue
        classes.append(t[0])
        v = [float(x) for x in t[1 : len(t) - 1]]
        support.append(int(t[-1]))
        class_names.append(t[0])
        plotMat.append(v)

    plotMat = np.array(plotMat)
    xticklabels = ["Precision", "Recall", "F1-score"]
    yticklabels = [
        "{0}".format(class_names[idx], sup) for idx, sup in enumerate(support)
    ]

    plt.imshow(plotMat, interpolation="nearest", cmap=cmap, aspect="auto")
    plt.title(title)
    plt.colorbar()
    plt.xticks(np.arange(3), xticklabels, rotation=45)
    plt.yticks(np.arange(len(classes)), yticklabels)

    upper_thresh = plotMat.min() + (plotMat.max() - plotMat.min()) / 10 * 8
    lower_thresh = plotMat.min() + (plotMat.max() - plotMat.min()) / 10 * 2
    for i, j in itertools.product(range(plotMat.shape[0]), range(plotMat.shape[1])):
        plt.text(
            j,
            i,
            format(plotMat[i, j], ".2f"),
            horizontalalignment="center",
            color="white"
            if (plotMat[i, j] > upper_thresh or plotMat[i, j] < lower_thresh)
            else "black",
        )

    plt.ylabel("Metrics")
    plt.xlabel("Classes")
    plt.tight_layout()
    plt.show()
