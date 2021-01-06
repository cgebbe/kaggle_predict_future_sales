import functools
import os
import pathlib
import numpy as np
import sys
import pandas as pd
import pathlib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
import nltk
import logging

logger = logging.getLogger(__name__)


# @functools.lru_cache()
def read_csv(filename,
             parent_path='../data/',
             ):
    """
    Wrapper around pd.read_csv(), but with cache!

    :param filename: e.g. sales_train.csv
    :param parent_path: path to parent folder
    :return: DataFrame
    """
    path_csv = pathlib.Path(parent_path) / filename
    assert path_csv.is_file()
    df = pd.read_csv(path_csv)
    return df


def calc_from_text(series,
                   nfeatures,
                   method='svd',
                   ):
    """
    Calculates features from text in three steps:
        1. Stemming
        2. TF-IDF
        3. PCA or TruncatedSVD

    :param series: Series object
    :param nfeatures: number of PCA features
    :param method: either SVD or PCA
    :return:
    """
    # stem
    stemmer = nltk.stem.snowball.SnowballStemmer("russian")
    stemfunc = lambda s: ' '.join([stemmer.stem(w) for w in s.split(' ')])
    series_stemmed = series.apply(stemfunc)

    # TF-IDF
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(series_stemmed)
    unique_words = vectorizer.get_feature_names()

    # Truncated SVD
    if method == 'svd':
        X_reduced = TruncatedSVD(n_components=nfeatures).fit_transform(X)
    elif method == 'pca':
        X_normalized = StandardScaler().fit_transform(X.toarray())
        X_reduced = PCA(n_components=nfeatures).fit_transform(X_normalized)
    else:
        raise ValueError("method={} must be svd or pca".format(method))
    return X_reduced


def setup_logging(path_logfile=None):
    """
    Setup logging to stream and to file

    :param path_logfile:
    :return:
    """

    log = logging.getLogger('')
    log.setLevel(logging.INFO)
    format = logging.Formatter("%(asctime)s @ %(name)s [%(levelname)s] - %(message)s")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(format)
    log.addHandler(sh)

    if path_logfile != None:
        path_logdir = pathlib.Path(path_logfile).parent
        path_logdir = path_logdir.absolute()
        path_logdir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(path_logfile)
        fh.setFormatter(format)
        log.addHandler(fh)
        logger.info("Logging to {}".format(path_logfile))


def encode_category_using_cumsum(df, colname, tarname, allow_nans=False):
    """
    Encodes a column using target mean with expanding mean regularization.
    Warning: Does not work if there are nans in target column, see https://stackoverflow.com/questions/52053091/pandas-cumcount-when-np-nan-exists

    :param df: dataframe
    :param colname: name of column to be replaced
    :param tarname: name of column with target values
    :Param use_nans: The first category member will always be encoded using NaNs. If True, these NaNs are replaced with the global mean
    :return:
    """
    if df[tarname].isna().sum() > 0:
        raise ValueError("Warning: There are nans in target column. Cannot encode this with cumsum!")

    df = df.sample(frac=1)
    cumsum = df.groupby(colname)[tarname].cumsum() - df[tarname]
    cumcnt = df.groupby(colname).cumcount()
    encoded = cumsum / cumcnt

    if not allow_nans:
        globalmean = df[tarname].mean()
        encoded.replace(float('NaN'), globalmean, inplace=True)
    return encoded


def encode_category_using_smoothing(df, colname, tarname):
    """
    Encode a column using target mean with smoothing regularization

    :param df: dataframe
    :param colname: name of column to be replaced
    :param tarname: name of column with target values
    :return:
    """
    # calculate mean per category
    mask = ~df[tarname].isna()
    mean_per_cat = df.loc[mask, :].groupby(colname).mean()[tarname]
    nrows_per_cat = df.loc[mask, :].groupby(colname).count()[tarname]

    # regularize using global mean
    nrows_valid = len(mask)
    nrows_trust = 3  # we "trust" the mean if the category has >=3 rows
    mean_global = df.loc[mask, tarname].mean()
    mean_per_cat = (nrows_per_cat * mean_per_cat + nrows_trust * mean_global) / (nrows_per_cat + nrows_trust)

    # apply mean encoding
    encoded = df[colname].map(mean_per_cat)

    # NaNs can still occur if all targets are NaN for a given category. Then, replace with globalmean
    encoded.replace(float('NaN'), mean_global, inplace=True)
    return encoded


def calc_rmse(true, pred):
    """
    Calculate root mean square error

    :param true:
    :param pred:
    :return:
    """
    se = (true-pred)**2
    mse = np.mean(se)
    rmse = np.sqrt(mse)
    return rmse