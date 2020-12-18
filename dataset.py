import pandas as pd
import numpy as np
import functools
import pathlib
import itertools
import logging
import final.src.features
import final.src.features_t_independent

logger = logging.getLogger(__name__)


class Dataset:
    def __init__(self, use_testcsv, dateblock_target=None):
        # init dataframe
        if use_testcsv:
            self.df = self.define_targets_from_test()
            self.dateblock_target = 34
        else:
            self.df = self.define_targets_from_train(dateblock_target)
            self.dateblock_target = dateblock_target

        # make sure that columns are the same!
        cols = ['date_block_num', 'year', 'month', 'shop_id', 'item_id']  # dropping item_price!
        if not use_testcsv:
            cols.append('item_cnt_month')
        for c in cols:
            assert c in self.df.columns, "c={} does not exist".format(c)
        self.df = self.df.loc[:, cols]

    def define_targets_from_test(self):
        """

        :return:
        """
        path_csv_test = pathlib.Path('/mnt/sda1/projects/git/courses/coursera_win_kaggle/final/data/test.csv')
        assert path_csv_test.is_file()
        df = pd.read_csv(path_csv_test)
        df['date_block_num'] = 34
        df['year'] = 2015
        df['month'] = 11

        # check
        nshops = df['shop_id'].nunique()
        nitems = df['item_id'].nunique()
        assert len(df) == nshops * nitems, "all combinations are tested"
        return df

    def define_targets_from_train(self, dateblock_target):
        """

        :param dateblock_target: which
        :return:
        """
        path_csv_train = pathlib.Path('/mnt/sda1/projects/git/courses/coursera_win_kaggle/final/data/sales_train.csv')
        assert path_csv_train.is_file()
        df = final.src.features_t_independent.parse_train()
        mask_target = df['date_block_num'] == dateblock_target

        # create new dataframe with all possible shopID and itemID combinations
        shops = df.loc[mask_target, 'shop_id'].unique().tolist()
        items = df.loc[mask_target, 'item_id'].unique().tolist()
        df_new = pd.DataFrame(itertools.product(shops, items), columns=['shop_id', 'item_id'])
        df_new = df_new.merge(df.loc[mask_target, :], how='left', on=['shop_id', 'item_id'])
        assert len(df_new) == len(shops) * len(items)

        # fill Nans
        idx = df.index[mask_target][0]
        for col in ['date_block_num', 'year', 'month']:
            value = df.loc[idx, col]
            df_new[col].fillna(value, inplace=True)
        df_new['item_cnt_month'].fillna(0, inplace=True)
        assert df_new.isna().sum().sum() == 0

        return df_new

    def get_Xy(self):
        """
        Returns the features X and targets y as pandas DataFrame / Series

        :return: (X,y) tuple
        """
        # get X
        features = self.df.columns.tolist()
        if 'item_cnt_month' in features:
            features.remove('item_cnt_month')
        X = self.df.loc[:, features]

        # get y
        if 'item_cnt_month' in self.df.columns:
            y = self.df.loc[:, 'item_cnt_month']
        else:
            y = None
        return X, y

    def calc_features(self):
        logger.info("Calculating features")

        # get target dateblock
        # columns = ['date_block_num', 'year', 'month', 'shop_id', 'item_id', 'item_cnt_month']
        nrows, nfeatures = self.df.shape

        # MERGE ON ITEM_ID (items.csv -> sales_train.csv)
        df_items = final.src.features.parse_items(self.dateblock_target)
        self.df = self.df.merge(df_items, on='item_id')
        assert len(self.df) == nrows

        # MERGE ON ITEM_CAT_ID (item_categories.csv -> items.csv)
        df_item_cats = final.src.features.parse_item_cats(self.dateblock_target)
        self.df = self.df.merge(df_item_cats, on='item_category_id')
        assert len(self.df) == nrows

        # MERGE ON SHOP_ID
        df_shops = final.src.features.parse_shops(self.dateblock_target)
        self.df = self.df.merge(df_shops, on='shop_id')
        assert len(self.df) == nrows

        # MERGE ON ITEM_ID + SHOP_ID ?!?
        # ...

        # DROP ORIGINAL ITEMS?
        # self.df.drop('date_block_num', axis=1, inplace=True)
        # self.df.drop('year', axis=1, inplace=True)
        # self.df.drop('month', axis=1, inplace=True)
        # self.df.drop('item_id', axis=1, inplace=True)
        # self.df.drop('shop_id', axis=1, inplace=True)
