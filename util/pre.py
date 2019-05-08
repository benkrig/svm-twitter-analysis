import re
import nltk
from nltk.stem import snowball

sn = snowball.SnowballStemmer("english")


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
