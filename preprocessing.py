import pandas as pd
import random
import re
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold
from collections import Counter


def remove_links(comment):
    pattern = re.compile(r"((https?):((//)|(\\\\))+[\w\d:#@%/;$()~_?\+-=\\\.&]*)", re.MULTILINE | re.UNICODE)
    comment = re.sub(pattern, "", comment).strip()
    return comment


def remove_empty_lines(comment):
    pattern = re.compile(r"^\s*$")
    comment = re.sub(pattern, "", comment)
    comment = re.sub(" +", " ", comment)
    return comment


def fix_encoding(comment):
    comment = comment.replace("â€", "\"")
    comment = comment.replace("\"™", "\'")
    comment = comment.replace("œ", "")
    comment = comment.replace("\\", "")
    return comment


def remove_tabs(comment):
    return ' '.join(comment.split(sep=None))


def clean_comment(comment):
    comment = fix_encoding(comment)
    comment = comment.replace("\n\n", " ")
    comment = remove_links(comment)
    comment = remove_empty_lines(comment)
    comment = remove_tabs(comment)
    comment = strip_timestamp(comment)
    return comment


def strip_timestamp(comment):
    pattern = re.compile("\[?\d+\s?\:\s?\d+\:?\d+\]")
    comment = re.sub(pattern, "", comment).strip()
    return comment
