import pandas as pd
import numpy as np
import functools
import pathlib


@functools.lru_cache(maxsize=None)
def parse_train():
    """
    Parses the sales_train.csv

    :return: Dataframe with columns [date, dateblocknum, shop_id, item_id, item_cnt_month]
    """
    path_csv = pathlib.Path('/mnt/sda1/projects/git/courses/coursera_win_kaggle/final/data/sales_train.csv')
    assert path_csv.is_file()
    df = pd.read_csv(path_csv)

    cols = ['date', 'date_block_num', 'shop_id', 'item_id']  # dropping item_price!
    df = df.groupby(cols)['item_cnt_day'].sum()
    df = df.reset_index()

    # add new columns
    df['year'] = df['date'].str.slice(start=6).astype(int)
    df['month'] = df['date'].str.slice(start=3, stop=5).astype(int)

    # clip values to [0,20]. Otherwise very different metrics compared to leaderboard!
    df['item_cnt_month'] = np.clip(df['item_cnt_day'].values, 0, 20)  # clip to [0,20]
    df.drop('item_cnt_day', axis=1, inplace=True)
    return df


class Dataset:
    def __init__(self, use_testcsv, dateblock_target=None):
        # init dataframe
        if use_testcsv:
            self.df = self.define_targets_from_test()
        else:
            self.df = self.define_targets_from_train(dateblock_target)

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
        return df

    def define_targets_from_train(self, dateblock_target):
        """

        :param dateblock_target: which
        :return:
        """
        path_csv_train = pathlib.Path('/mnt/sda1/projects/git/courses/coursera_win_kaggle/final/data/sales_train.csv')
        assert path_csv_train.is_file()
        df = parse_train()
        mask_target = df['date_block_num'] == dateblock_target
        df = df.loc[mask_target, :]
        return df

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
        # get target dateblock
        dateblock_target = self.df['date_block_num'].unique().tolist()
        assert len(dateblock_target) == 1, "somehow several target dateblocks?!"
        dateblock_target = dateblock_target[0]

        # load all features BEFORE target month
        dftrain = parse_train()
        mask_before = dftrain['date_block_num'] < dateblock_target
        dftrain = dftrain.loc[mask_before, :]

        # for each item_ID, get sales in last 1,3,6,12,24 months
        dftrain['date_block_num'] -= dateblock_target
