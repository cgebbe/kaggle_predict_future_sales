"""
Defines train and valid dataset. Computes all features.
"""
import logging
import pandas as pd
import tqdm
import numpy as np
import utils

logger = logging.getLogger(__name__)


def generate(sparsify_by=1):
    """
    Generates the train, valid and test data including features

    :return: DataFrame with many columns
    """
    logger.info("Defining train,valid data")
    train = define_train()
    if sparsify_by != 1.0:
        train = train.iloc[::sparsify_by, :]
    train = calc_year_month_itemcntmonth(train)
    nrows = len(train)

    # fix shop IDs
    train.loc[train.shop_id == 0, 'shop_id'] = 57
    train.loc[train.shop_id == 1, 'shop_id'] = 58
    train.loc[train.shop_id == 10, 'shop_id'] = 11

    # Calc sales for previous months
    train = calc_sales(train)
    assert len(train) == nrows

    # Calc price per item and previous months
    df = calc_price()
    train = train.merge(df, how='left', on=['date_block_num', 'item_id'])
    assert len(train) == nrows

    # omit first 4 months because not possible to calculate previous months here (0,1,2,3)
    assert len(train) == nrows
    train = train.loc[train['date_block_num'] >= 4, :]
    nrows = len(train)

    # items.csv
    df = parse_items_csv()
    train = train.merge(df, how='left', on='item_id')
    assert len(train) == nrows

    # item_categories.csv
    df = parse_item_categories_csv()
    train = train.merge(df, how='left', on='item_category_id')
    assert len(train) == nrows

    # shop.csv
    df = parse_shop_csv()
    train = train.merge(df, how='left', on='shop_id')
    assert len(train) == nrows

    return train


def define_train():
    """
    Define train dataset, i.e. set of dateblocks, items, shops for training

    :return: DataFrame with columns [date_block_num, item_id, shop_id]
    """
    list_new = []

    # read train and test
    df = utils.read_csv('sales_train.csv')
    dateblocks_unique = df['date_block_num'].unique().tolist()
    dateblocks_unique.sort()
    nrows = 0
    for dt in tqdm.tqdm(dateblocks_unique):
        # find shops and items with at least one entry in this dataframe
        mask = df['date_block_num'] == dt
        shops = df.loc[mask, 'shop_id'].unique().tolist()
        items = df.loc[mask, 'item_id'].unique().tolist()
        nrows += len(shops) * len(items)

        # create new dataframe with combination
        new1 = pd.DataFrame({'j': 0, 'shop_id': sorted(shops)}, dtype=np.uint8)
        new2 = pd.DataFrame({'j': 0, 'item_id': sorted(items)}, dtype=np.uint16)
        new = pd.merge(new1, new2, on='j', how='outer')
        new['date_block_num'] = np.array(dt, dtype=np.uint8)
        new.drop(columns='j', inplace=True)
        list_new.append(new)

    # combine all dataframes
    train = pd.DataFrame().append(list_new)
    assert len(train) == nrows

    # append test dataframe to calculate features for this, too
    test = utils.read_csv('test.csv')
    test['date_block_num'] = 34
    test = test.drop(columns='ID')
    # test.head()
    # test['date'] = '01.11.2015'
    # test['item_price'] = float('NaN')
    # test['year'] = 2015 - 2000
    # test['month'] = 11
    # test['item_cnt_month'] = float('NaN')
    train = train.append(test, ignore_index=True)

    return train


def calc_year_month_itemcntmonth(train):
    """
    Adds columns  [year, month, item_cnt_month]

    :param train: DataFrame
    :return: DataFrame with added columns
    """
    nrows_org = len(train)

    # read
    df = utils.read_csv('sales_train.csv')

    # add month,year to train
    df['year'] = df['date'].str.slice(start=6).astype(int) - 2000
    df['month'] = df['date'].str.slice(start=3, stop=5).astype(int)
    year_per_dateblock = df.groupby('date_block_num')[['year', 'month']].mean()
    year_per_dateblock.loc[34, :] = [15, 11]
    for col in ['month', 'year']:
        year_per_dateblock[col] = year_per_dateblock[col].astype(np.uint8)
        train[col] = train['date_block_num'].map(year_per_dateblock[col])

    # add sales per month
    sales_per_month = df.groupby(['date_block_num', 'shop_id', 'item_id'])['item_cnt_day'].sum()
    sales_per_month = sales_per_month.reset_index()
    train = train.merge(sales_per_month, on=['date_block_num', 'shop_id', 'item_id'], how='left')
    train = train.rename(columns={'item_cnt_day': 'item_cnt_month'})
    train['item_cnt_month'] = train['item_cnt_month'].fillna(0).clip(lower=0, upper=20)
    train['item_cnt_month'] = train['item_cnt_month']  # .astype(np.uint8)
    assert train.isna().sum().sum() == 0
    assert len(train) == nrows_org

    return train


def parse_items_csv():
    """
    Calculates features (item_category_id, item_text_0..9)

    :return:
    """
    # Features: item_category_id
    logger.info("Parsing items.csv")
    df = utils.read_csv('items.csv')

    # Features: item_text_0...9
    X_pca = utils.calc_from_text(df['item_name'], nfeatures=10)
    ncats, nfeatures = X_pca.shape
    df_add = pd.DataFrame(X_pca, columns=['item_text_{}'.format(i) for i in range(nfeatures)])
    df = df.join(df_add)
    df = df.drop(columns='item_name')
    return df


def parse_item_categories_csv():
    """
    Calculates features (itemcat_text_0...9, ...)
    Taken 99% from https://www.kaggle.com/gordotron85/future-sales-xgboost-top-3

    :param train:
    :return:
    """
    logger.info("Parsing item_categories.csv")
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


def parse_shop_csv():
    """
    Calculates features from shop description [shop_city, shop_category]
    Taken 99% from https://www.kaggle.com/gordotron85/future-sales-xgboost-top-3

    :param train:
    :return:
    """
    logger.info("Parsing shops.csv")
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
    logger.info("Calculating features from number of sales")
    indexes = [['item_id'],
               ['shop_id'],
               ['item_id', 'shop_id'],
               ]

    # GET (MEAN) SALES FROM PREVIOUS MONTHS
    for index in tqdm.tqdm(indexes):
        df = pd.pivot_table(train, values='item_cnt_month', aggfunc='mean', index=['date_block_num'] + index)
        df = df.rename(columns={'item_cnt_month': 't'})
        for trel in [1, 2, 3, 4]:
            index2 = pd.MultiIndex.from_tuples([(idx[0] - trel, *idx[1:]) for idx in df.index])
            df['t-{}'.format(trel)] = df['t'].reindex(index=index2).values
        df.drop(columns='t', inplace=True)

        # calculate relative difference in sales between 1-2 and 2-3
        eps = 1e-5
        df.loc[:, 'd_t12'] = 0.5 * (df.loc[:, 't-1'] - df.loc[:, 't-2']) / (df.loc[:, 't-1'] + df.loc[:, 't-2'] + eps)
        df.loc[:, 'd_t13'] = 0.5 * (df.loc[:, 't-1'] - df.loc[:, 't-3']) / (df.loc[:, 't-1'] + df.loc[:, 't-3'] + eps)
        df.loc[:, 'd_t23'] = 0.5 * (df.loc[:, 't-2'] - df.loc[:, 't-3']) / (df.loc[:, 't-2'] + df.loc[:, 't-3'] + eps)
        df.columns = ['_'.join(index) + '_' + k for k in df.columns]
        df = df.reset_index()
        df = df.fillna(value=0)  # .astype(np.uint32)
        train = train.merge(df, how='left', on=['date_block_num'] + index)

    # CALC MEAN SALES OF ALL PREVIOUS MONTHS (regularized by expanding mean over time)
    for index in indexes:
        df = pd.pivot_table(train, values='item_cnt_month', aggfunc='sum', index=index, columns='date_block_num')
        df = df.fillna(value=0)
        df = df.cumsum(axis=1) - df  # = cumulative sales of all previous months (NOT including current month!)
        for col in df.columns:
            if col == 0:
                continue  # sales of first month are zero anyhow
            else:
                df.loc[:, col] /= col  # divide cumulative sales with number of previous months ~ cumcount
        df = df.stack()  # reshape mxn into (m*n)x1
        df = pd.DataFrame(df, columns=['_'.join(index) + '_meansales'])

        # check that cumulative mean for month34 is equal to total sum divided by number of month -> True
        # df_check = pd.pivot_table(train, values='item_cnt_month', aggfunc='sum', index=index)
        # assert df.loc[(22167,34),:].values[0] == (df_check.loc[22167,:].values[0] / 34)

        # merge with train
        train = train.merge(df, how='left', on=['date_block_num'] + index)

    # count number of month (so far!) with sales
    for index in indexes:
        df = pd.pivot_table(train, values='item_cnt_month', aggfunc='sum', index=index, columns='date_block_num')
        df = df.fillna(value=0)
        df = df > 0
        df = df.cumsum(axis=1) - df
        df = df.stack()
        df = pd.DataFrame(df, columns=['_'.join(index) + '_nmonth_sales'])
        train = train.merge(df, how='left', on=['date_block_num'] + index)

    return train


def calc_price():
    """
    Calc price per item and month (also relative to other previous months)

    :return:
    """
    logger.info("Calculating features from price")
    df = utils.read_csv('sales_train.csv')

    # calculate mean price for each item and date_block
    df = pd.pivot_table(df, values='item_price', aggfunc='mean', index=['date_block_num', 'item_id'])
    df.columns = ['t']
    for trel in [1, 2, 3]:
        index2 = pd.MultiIndex.from_tuples([(idx[0] - trel, *idx[1:]) for idx in df.index])
        df['t-{}'.format(trel)] = df['t'].reindex(index=index2).values
    df.drop(columns='t', inplace=True)

    # calculate difference between 1-2 and 2-3
    eps = 1e-5
    df.loc[:, 'd_t12'] = 0.5 * (df.loc[:, 't-1'] - df.loc[:, 't-2']) / (df.loc[:, 't-1'] + df.loc[:, 't-2'] + eps)
    df.loc[:, 'd_t13'] = 0.5 * (df.loc[:, 't-1'] - df.loc[:, 't-3']) / (df.loc[:, 't-1'] + df.loc[:, 't-3'] + eps)
    df.loc[:, 'd_t23'] = 0.5 * (df.loc[:, 't-2'] - df.loc[:, 't-3']) / (df.loc[:, 't-2'] + df.loc[:, 't-3'] + eps)
    df.columns = ['item_price_{}'.format(k) for k in df.columns]

    return df
