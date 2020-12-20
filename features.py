import logging
import functools
import final.src.features_t_independent

logger = logging.getLogger(__name__)


@functools.lru_cache()
def parse_train_for_dateblock(dateblock_target):
    dftrain = final.src.features_t_independent.parse_train()
    mask_before = dftrain['date_block_num'] < dateblock_target
    dftrain = dftrain.loc[mask_before, :]
    dftrain['date_block_num'] -= dateblock_target  # convert dateblocknum into relative time
    return dftrain


def parse_items(dateblock_target):
    df = final.src.features_t_independent.parse_items()

    # sales in last 1,3,6,12,24 months
    dftrain = parse_train_for_dateblock(dateblock_target)
    masks = dict()
    masks['-1a'] = dftrain['date_block_num'] == -12
    for n in [1 , 3, 6]:
        masks['last{}'.format(n)] = dftrain['date_block_num'] >= -n
    for key, mask_time in masks.items():
        n_perid = dftrain.loc[mask_time, :].groupby('item_id')['item_cnt_month'].mean()
        df['n_peritem_{}'.format(key)] = df['item_id'].map(n_perid).astype(float)
        df['n_peritem_{}'.format(key)].fillna(value=-1, inplace=True)

    # price
    path_csv_sales = '/mnt/sda1/projects/git/courses/coursera_win_kaggle/final/data/sales_train.csv'
    sales = final.src.features_t_independent.read_csv(path_csv_sales)
    price_perid = sales.groupby('item_id')['item_price'].mean()
    df['price_peritem'] = df['item_id'].map(price_perid)

    return df


def parse_item_cats(dateblock_target):
    df = final.src.features_t_independent.parse_item_cats()

    # add sales per item category in last 1,3,6,12,24 months
    # dftrain['item_cat'] = dftrain['item_id'].map(dfitems['item_category_id'])
    # for n in [1, 2,3, 6,12]:
    #     mask_time = dftrain['date_block_num'] >= -n
    #     tmp = dftrain.loc[mask_time, :].groupby('item_cat')['item_cnt_month'].mean()
    #     self.df['nper_itemcat_{}months'.format(n)] = self.df['item_cat'].map(tmp)

    return df


def parse_shops(dateblock_target):
    df = final.src.features_t_independent.parse_shops()

    # add sales per shop_id
    dftrain = parse_train_for_dateblock(dateblock_target)
    masks = dict()
    masks['-1a'] = dftrain['date_block_num'] == -12
    for n in [1, 3, 6]:
        masks['last{}'.format(n)] = dftrain['date_block_num'] >= -n
    for key, mask_time in masks.items():
        tot = dftrain.loc[mask_time, :]['item_cnt_month'].mean()
        n_perid = dftrain.loc[mask_time, :].groupby('shop_id')['item_cnt_month'].mean() / tot
        df['n_pershop_{}'.format(key)] = df['shop_id'].map(n_perid)

    return df
