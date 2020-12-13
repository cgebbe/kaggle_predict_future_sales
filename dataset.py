import pandas as pd
import numpy as np


class Dataset:
    def __init__(self, path_csv, is_train, sparsify_by=1):
        """
        Inits a dataset (effectively a standardized pandas DataFrame)

        :param path_csv: path to csv with values
        :param is_train: If True, assumes that CSV represents train.csv, which has more columns
        :param sparsify_by: If >1, sparsifies rows by this factor
        """
        assert path_csv.is_file()
        df = pd.read_csv(path_csv)
        df = df.iloc[::sparsify_by, :]

        # modify df
        if is_train:
            df = df.groupby(['date', 'shop_id', 'item_id'])['item_cnt_day'].sum()
            df = df.reset_index()
            df['year'] = df['date'].str.slice(start=6).astype(int)
            df['month'] = df['date'].str.slice(start=3, stop=5).astype(int)
            df.drop('date', axis=1, inplace=True)

            # clip values to [0,20]. Otherwise very different metrics compared to leaderboard!
            df['item_cnt_day'] = np.clip(df['item_cnt_day'].values, 0, 20)  # clip to [0,20]
            self.df = df
        else:
            df['year'] = 2015
            df['month'] = 11

        # Now, train and test should have same features  !!!
        features = ['shop_id', 'item_id', 'year', 'month']
        if is_train:
            features.append('item_cnt_day')
        df = df.loc[:, features]
        self.df = df

    def get_Xy(self):
        """
        Returns the features X and targets y as pandas DataFrame / Series

        :return: (X,y) tuple
        """
        # get X
        features = self.df.columns.tolist()
        if 'item_cnt_day' in features:
            features.remove('item_cnt_day')
        X = self.df.loc[:, features]

        # get y
        if 'item_cnt_day' in self.df.columns:
            y = self.df.loc[:, 'item_cnt_day']
        else:
            y = None
        return X, y

    def calc_features(self):
        """
        Adds more columns / features to the existing dataframe

        :return:
        """
        # calculate mean sales per item_ID in last 1,3,6,12 months
