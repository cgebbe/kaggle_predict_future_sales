import pandas as pd
import pathlib
import yaml
import tqdm
import numpy as np
import pprint
import catboost
import functools
import logging
import final.src.features as features
import final.src.utils as utils
import pickle
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main():
    # get "raw" training data (precalculated from pickle)
    if False:
        train = define_train()
        train = fill_train(train)
        with open('train.pickle', 'wb') as f:
            pickle.dump(train, f)
    else:
        with open('train.pickle', 'rb') as f:
            train = pickle.load(f)
    print(train.head())

    # calc features
    train = features.calc(train)

    # split train into [train,valid] and get X,y
    Xtrain, ytrain = _getXy(train.loc[train['date_block_num'] < 33, :])
    Xvalid, yvalid = _getXy(train.loc[train['date_block_num'] == 33, :])

    # train model
    logger.info("Setup model")
    model = catboost.CatBoostRegressor(random_seed=42,
                                       iterations=100,
                                       loss_function='RMSE',  # MSE not supported?!
                                       train_dir='catboost',
                                       task_type='GPU',
                                       )
    cat_features = np.where(Xtrain.dtypes != float)[0]

    # fit model
    logger.info("Fit model")
    sparsiy_factor=1
    model.fit(Xtrain[::sparsiy_factor], ytrain[::sparsiy_factor],
              eval_set=(Xvalid, yvalid),
              cat_features=cat_features,
              # silent=True,
              metric_period=50,
              # use_best_model=False # mainly to check
              )

    # eval model
    logger.info("Eval model")
    eval = pd.DataFrame({'name': model.feature_names_,
                         'importance': model.feature_importances_})
    eval.sort_values(by='importance', ascending=True, inplace=True)
    print(eval)


def define_train():
    """
    Define train dataset, i.e. set of dateblocks, items, shops for training

    :return: DataFrame with columns [date_block_num, item_id, shop_id]
    """
    list_new = []
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
    train = train.iloc[:, [2, 0, 1]]  # change column order
    assert len(train) == nrows
    return train


def fill_train(train):
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
    for col in ['month', 'year']:
        year_per_dateblock[col] = year_per_dateblock[col].astype(np.uint8)
        train[col] = train['date_block_num'].map(year_per_dateblock[col])

    # add sales per month
    sales_per_month = df.groupby(['date_block_num', 'shop_id', 'item_id'])['item_cnt_day'].sum()
    sales_per_month = sales_per_month.reset_index()
    train = train.merge(sales_per_month, on=['date_block_num', 'shop_id', 'item_id'], how='left')
    train = train.rename(columns={'item_cnt_day': 'item_cnt_month'})
    train['item_cnt_month'] = train['item_cnt_month'].fillna(0).clip(lower=0, upper=20)
    train['item_cnt_month'] = train['item_cnt_month'].astype(np.uint8)
    assert train.isna().sum().sum() == 0

    assert len(train) == nrows_org
    return train


def _getXy(df):
    y = df['item_cnt_month']
    X = df.drop('item_cnt_month', axis=1)
    return X, y


if __name__ == '__main__':
    main()
