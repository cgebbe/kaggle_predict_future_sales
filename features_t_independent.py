import logging
import functools
import pathlib
import pandas as pd
import numpy as np
import nltk
import utils
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

logger = logging.getLogger(__name__)


@functools.lru_cache()
def parse_train():
    """
    Parses the sales_train.csv

    :return: Dataframe with columns [date, dateblocknum, shop_id, item_id, item_cnt_month]
    """
    df = utils.read_csv('sales_train.csv')

    # convert date to month-year
    df['year'] = df['date'].str.slice(start=6).astype(int)
    df['month'] = df['date'].str.slice(start=3, stop=5).astype(int)
    df.drop('date', axis=1, inplace=True)

    # keep remaining cols
    cols = ['date_block_num', 'year', 'month', 'shop_id', 'item_id']  # dropping item_price!
    df = df.groupby(cols)['item_cnt_day'].sum()
    df = df.reset_index()

    # clip values to [0,20]. Otherwise very different metrics compared to leaderboard!
    df['item_cnt_month'] = df['item_cnt_day'].clip(0, 20)  # clip to [0,20]
    df.drop('item_cnt_day', axis=1, inplace=True)
    return df


@functools.lru_cache()
def parse_item_cats():
    logger.info("Calculating features for item_categories.csv")
    df = utils.read_csv('item_categories.csv')

    # calculate features from category description text and drop original category name
    X_pca = _calc_from_text(df['item_category_name'], nfeatures=5)
    ncats, nfeatures = X_pca.shape
    df_add = pd.DataFrame(X_pca, columns=['item_cat_desc{}'.format(i) for i in range(nfeatures)])
    df = df.join(df_add)
    df.drop('item_category_name', axis=1, inplace=True)

    # return
    print(df.columns)
    print(df.shape)
    return df


@functools.lru_cache()
def parse_items():
    logger.info("Calculating features for items.csv")
    df = utils.read_csv('items.csv')

    # calculate features from description text and drop original category name
    X_pca = _calc_from_text(df['item_name'], nfeatures=10)
    ncats, nfeatures = X_pca.shape
    df_add = pd.DataFrame(X_pca, columns=['cat_text{}'.format(i) for i in range(nfeatures)])
    df = df.join(df_add)
    df.drop('item_name', axis=1, inplace=True)

    # return
    print(df.columns)
    print(df.shape)
    return df


@functools.lru_cache()
def parse_shops():
    pass


def _calc_from_text(series,
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
