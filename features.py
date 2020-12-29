import pandas as pd
import final.src.features_t_independent as features_t_independent
import final.src.utils as utils
import logging
import tqdm
import numpy as np

logger = logging.getLogger(__name__)


def calc(train):
    """
    Calculates several features given all available data

    :param train:
    :return:
    """
    nrows = len(train)


    # CALC SALES OF LAST MONTHS
    indexes = [['date_block_num', 'item_id'],
               ['date_block_num', 'shop_id'],
               ['date_block_num', 'item_id', 'shop_id'],
               ]
    for index in tqdm.tqdm(indexes):
        df = pd.pivot_table(train, values='item_cnt_month', aggfunc='sum', index=index)
        df = df.rename(columns={'item_cnt_month': 't'})
        for trel in [-1, -2, -3, -6]:
            index2 = pd.MultiIndex.from_tuples([(idx[0] - trel, *idx[1:]) for idx in df.index])
            df['t{}'.format(trel)] = df['t'].reindex(index=index2).values
        df.drop(columns='t', inplace=True)
        key_prefix = '_'.join(index[1:])
        df.columns = [key_prefix + '_' + k for k in df.columns]
        df = df.reset_index()
        df = df.fillna(value=0) # .astype(np.uint32)
        train = train.merge(df, on=index)

    # omit first 6 months because no lag features there (0,1,2,3,4,5)
    assert len(train) == nrows
    train = train.loc[train['date_block_num'] >= 6, :]
    nrows = len(train)

    # MERGE ON ITEM_ID (information from items.csv)
    df = calc_peritem()
    train = train.merge(df, on='item_id')
    assert len(train) == nrows

    # MERGE ON ITEM_CATEGORY_ID (information from item_categories.csv)
    df = calc_peritemcat()
    train = train.merge(df, on='item_category_id')
    assert len(train) == nrows

    # MERGE ON SHOP_ID (information from shop.csv)
    df = calc_pershop()
    train = train.merge(df, on='shop_id')
    assert len(train) == nrows


    return train


def calc_peritem():
    """
    Calculates features (item_category_id, item_text_0..9)

    :return:
    """
    # Features: item_category_id
    logger.info("Calculating features for items.csv")
    df = utils.read_csv('items.csv')

    # Features: item_text_0...9
    X_pca = features_t_independent._calc_from_text(df['item_name'], nfeatures=10)
    ncats, nfeatures = X_pca.shape
    df_add = pd.DataFrame(X_pca, columns=['item_text_{}'.format(i) for i in range(nfeatures)])
    df = df.join(df_add)
    df = df.drop(columns='item_name')
    return df


def calc_peritemcat():
    """
    Calculates features (itemcat_text_0...9, ...)

    :param train:
    :return:
    """
    logger.info("Calculating features for item_categories.csv")
    df = utils.read_csv('item_categories.csv')

    # calculate features from category description text and drop original category name
    X_pca = features_t_independent._calc_from_text(df['item_category_name'], nfeatures=5)
    ncats, nfeatures = X_pca.shape
    df_add = pd.DataFrame(X_pca, columns=['itemcat_text_{}'.format(i) for i in range(nfeatures)])
    df = df.join(df_add)
    df = df.drop(columns='item_category_name')
    return df


def calc_pershop():
    """
    Calculates features (shop_text_0...9, ...)

    :param train:
    :return:
    """
    logger.info("Calculating features for shops.csv")
    df = utils.read_csv('shops.csv')

    # calculate features from description text and drop original category name
    X_pca = features_t_independent._calc_from_text(df['shop_name'], nfeatures=5)
    ncats, nfeatures = X_pca.shape
    df_add = pd.DataFrame(X_pca, columns=['shop_text{}'.format(i) for i in range(nfeatures)])
    df = df.join(df_add)
    df = df.drop(columns='shop_name')
    return df


def calc_peritemshop(train):
    pass
