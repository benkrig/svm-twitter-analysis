import pandas as pd
import sklearn
import re
import nltk
from nltk.stem import snowball

sn = snowball.SnowballStemmer("english")


def get_data(**kwargs):
    """Helper for automatically retrieving data.
    Kwargs is present to allow different environments and data sets to be used.

    :param kwargs: passed as local environments may differ
    :return:
    """
    path = "data/sentiment140.csv"  # defaults
    col_names = ["target", "id", "date", "flag", "user", "text"]
    encoding = "ISO-8859-1"
    row_count = 1600000
    offset = 800000 - int(row_count / 2)

    if "path" in kwargs:
        path = kwargs["path"]
    if "col_names" in kwargs:
        col_names = kwargs["col_names"]
    if "encoding" in kwargs:
        encoding = kwargs["encoding"]
    # noinspection SpellCheckingInspection
    if "nrows" in kwargs:
        row_count = kwargs["nrows"]
        offset = 800000 - int(row_count / 2)

    return pd.read_csv(
        path, encoding=encoding, skiprows=offset, nrows=row_count, names=col_names
    )


def write_data(**kwargs):
    """Write a trained model for later use
    :param kwargs:
    :return:
    """
    sklearn.externals.joblib.dump(kwargs["model"], "data/model.pkl", compress=3)


def clean(df):
    """Applies all methods from util.pre
    :param df:
    :return:
    """
    df["text"] = df["text"].apply(pre.remove_user)
    df["text"] = df["text"].apply(pre.remove_url)
    df["text"] = df["text"].apply(pre.remove_quotes)
    df["text"] = df["text"].apply(pre.remove_ws)
    df["text"] = df["text"].apply(pre.lower)
    df["text"] = df["text"].apply(pre.remove_repeat)
    df["text"] = df["text"].apply(pre.add_pos)

    return df


def remove_user(_string):
    """Strips out user mentions from a string (@xyz)
    :param _string: str
    :return: str
    """
    clean = re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", _string)
    return clean


def remove_url(_string):
    """Strips out url mentions from a string (http(s))
    :param _string: str
    :return: str
    """
    clean = re.sub(
        r"[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&=]*)\S*",
        " ",
        _string,
    )
    return clean


def remove_quotes(_string):
    clean = _string.replace('"', " ")
    clean = clean.replace("'", " ")
    clean = clean.replace("amp", "and")
    clean = clean.replace("quot", " ")
    clean = re.sub(r"[^\w\s]", "", clean)
    return clean


def lower(_string):
    clean = str(_string).lower()
    return clean


def remove_repeat(_string):
    # noinspection SpellCheckingInspection
    """Strips out repeated characters from a string (aaaaaabbbbbcccc -> abc)
    :param _string: str
    :return: str
    """
    clean = re.sub(r"(.)\1{2,}", r"\1", _string)
    return clean


def add_pos(_string):
    """Adds part of speech (pos) to a string
    :param _string: str
    :return: str
    """
    text = _string.split()
    text_tagged = nltk.pos_tag(text)  # [("original", "pos"), ...]

    pos = ""
    clean = ""
    for word in text_tagged:
        clean += " " + sn.stem(word[0])

        pos += " " + word[1]
    return clean + pos


def remove_ws(_string):
    return re.sub(r"\s+", " ", _string).strip()
