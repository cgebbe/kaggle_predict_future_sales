import pandas as pd
import gc
import final.src.features_t_independent as features_t_independent
import final.src.utils as utils
import logging
import tqdm
import numpy as np
import sklearn.model_selection

logger = logging.getLogger(__name__)


def calc(train):
    """
    Calculates several features given all available data

    :param train:
    :return:
    """
    nrows = len(train)

    # fix shop IDs
    train.loc[train.shop_id == 0, 'shop_id'] = 57
    train.loc[train.shop_id == 1, 'shop_id'] = 58
    train.loc[train.shop_id == 10, 'shop_id'] = 11

    # Calc price per item and previous months
    df = calc_price()
    train = train.merge(df, how='left', on=['date_block_num', 'item_id'])
    assert len(train) == nrows

    # Calc sales for previous months
    train = calc_sales(train)

    # omit first 4 months because not possible to calculate previous months here (0,1,2,3)
    assert len(train) == nrows
    train = train.loc[train['date_block_num'] >= 4, :]
    nrows = len(train)

    # items.csv
    df = calc_peritem()
    train = train.merge(df, how='left', on='item_id')
    assert len(train) == nrows

    # item_categories.csv
    df = calc_peritemcat()
    train = train.merge(df, how='left', on='item_category_id')
    assert len(train) == nrows

    # shop.csv
    df = calc_pershop()
    train = train.merge(df, how='left', on='shop_id')
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
    Taken 99% from https://www.kaggle.com/gordotron85/future-sales-xgboost-top-3

    :param train:
    :return:
    """
    logger.info("Calculating features for item_categories.csv")
    cats = utils.read_csv('item_categories.csv')

    # extract type and encode it. Keep only those with at least 5 members
    cats["type"] = cats.item_category_name.apply(lambda x: x.split(" ")[0]).astype(str)
    cats.loc[(cats.type == "Игровые") | (cats.type == "Аксессуары"), "category"] = "Игры"
    n_per_cat = cats.type.value_counts()
    cats.type = cats.type.apply(lambda c: c if n_per_cat[c] >= 5 else "etc")
    # cats.type_code = LabelEncoder().fit_transform(cats.type)

    # extract subtype and encode it
    cats["split"] = cats.item_category_name.apply(lambda x: x.split("-"))
    cats["subtype"] = cats.split.apply(lambda x: x[1].strip() if len(x) > 1 else x[0].strip())
    # cats["subtype_code"] = LabelEncoder().fit_transform(cats["subtype"])

    cats = cats[["item_category_id", "type", "subtype"]]
    return cats


def calc_pershop():
    """
    Calculates features from shop description [shop_city, shop_category]
    Taken 99% from https://www.kaggle.com/gordotron85/future-sales-xgboost-top-3

    :param train:
    :return:
    """
    logger.info("Calculating features for shops.csv")
    df = utils.read_csv('shops.csv').copy()

    # split shop name into city and category
    df.loc[df.shop_name == 'Сергиев Посад ТЦ "7Я"', "shop_name"] = 'СергиевПосад ТЦ "7Я"'
    df["shop_city"] = df.shop_name.str.split(" ").map(lambda x: x[0])
    df["shop_category"] = df.shop_name.str.split(" ").map(lambda x: x[1])
    df.loc[df.shop_city == "!Якутск", "shop_city"] = "Якутск"

    # only keep categories with at least two members
    n_per_cat = df.shop_category.value_counts()
    df.shop_category = df.shop_category.apply(lambda c: c if n_per_cat[c] >= 2 else "other")

    # drop shop_name column
    df = df.loc[:, ['shop_id', 'shop_city', 'shop_category']]
    return df


def calc_sales(train):
    """
    Calculates multiple features based on target

    :param train: DataFrame
    :return: DataFrame with additional columns
    """
    indexes = [['item_id'],
               ['shop_id'],
               ['item_id', 'shop_id'],
               ]

    # CALC MEAN SALES FROM PREVIOUS MONTHS
    for index in tqdm.tqdm(indexes):
        df = pd.pivot_table(train, values='item_cnt_month', aggfunc='mean', index=['date_block_num'] + index)
        df = df.rename(columns={'item_cnt_month': 't'})
        for trel in [1, 2, 3, 4]:
            index2 = pd.MultiIndex.from_tuples([(idx[0] - trel, *idx[1:]) for idx in df.index])
            df['t-{}'.format(trel)] = df['t'].reindex(index=index2).values
        df.drop(columns='t', inplace=True)
        df.columns = ['_'.join(index) + '_' + k for k in df.columns]
        df = df.reset_index()
        df = df.fillna(value=0)  # .astype(np.uint32)
        train = train.merge(df, on=['date_block_num'] + index)
        # del df
        # gc.collect()

    # CALC MEAN SALES OF ALL MONTHS (regularized by CV-loop)
    for index in indexes:
        colname = '_'.join(index) + '_mean'
        kf = sklearn.model_selection.KFold(n_splits=5, shuffle=True)
        for idx_train, idx_val in kf.split(train.values):
            df = pd.pivot_table(train.loc[idx_train, :], values='item_cnt_month', aggfunc='mean', index=index)
            df.columns = [colname]
            df = df.reset_index()
            train.loc[idx_val, colname] = train.loc[idx_val, index].merge(df, on=index, how='left')[colname].values
        print("{} has nans in {}/{} rows".format(colname, train[colname].isna().sum(), len(train)))

    # get first month with sales
    for index in indexes:
        colname = '_'.join(index) + '_firstmonth'
        df = pd.pivot_table(train, values='item_cnt_month', aggfunc='mean', index=index, columns='date_block_num')
        df[colname] = (df > 0).idxmax(axis='columns')  # gets first occurence where sales>0
        train = train.merge(df.loc[:, colname], on=index, how='left')

    return train


def calc_price():
    """
    Calc price per item and month (also relative to other previous months)

    :return:
    """
    df = utils.read_csv('sales_train.csv')
    price_mean = pd.pivot_table(df, values='item_price', aggfunc='mean', index=['item_id']).rename(
        columns={'item_price': 'item_price_mean'})
    # price_median = pd.pivot_table(df, values='item_price', aggfunc='median', index=['item_id']).rename(columns={'item_price':'item_price_median'})
    # price = pd.concat([price_mean, price_median], axis=1)

    # calculate mean price for each item and date_block
    df = pd.pivot_table(df, values='item_price', aggfunc='mean', index=['date_block_num', 'item_id'])
    df.columns = ['t']
    for trel in [1, 2, 3]:
        index2 = pd.MultiIndex.from_tuples([(idx[0] - trel, *idx[1:]) for idx in df.index])
        df['t-{}'.format(trel)] = df['t'].reindex(index=index2).values

    # calculate difference between 1-2 and 2-3
    df.loc[:, 'delta_t12'] = df.loc[:, 't-1'] - df.loc[:, 't-2']
    df.loc[:, 'delta_t23'] = df.loc[:, 't-2'] - df.loc[:, 't-3']
    df = df.drop(columns='t')
    df.columns = ['item_price_{}'.format(k) for k in df.columns]

    # calculate price in percent w.r.t. overall mean (could probably be weighted by revenue, but hey...)
    df = df.reset_index()
    df = df.merge(price_mean, how='left', on='item_id')
    cols = [c for c in df.columns if c not in ['date_block_num', 'item_id', 'item_price_mean']]
    for c in cols:
        df.loc[:, c] /= df.loc[:, 'item_price_mean']

    return df
