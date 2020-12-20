import pandas as pd
import final.src.utils as utils
import logging
logger = logging.getLogger(__name__)

def calc(train):
    """

    :param train:
    :return:
    """
    nrows = len(train)

    # MERGE ON ITEM_ID (items.csv -> sales_train.csv)
    df = calc_peritem()
    train = train.merge(df, on='item_id')
    assert len(train) == nrows

    return train


def calc_peritem():
    logger.info("Calculating features for items.csv")
    df = utils.read_csv('items.csv')

    # calculate features from description text and drop original category name
    # X_pca = _calc_from_text(df['item_name'], nfeatures=10)
    # ncats, nfeatures = X_pca.shape
    # df_add = pd.DataFrame(X_pca, columns=['cat_text{}'.format(i) for i in range(nfeatures)])
    # df = df.join(df_add)
    df.drop('item_name', axis=1, inplace=True)

    # return
    print(df.columns)
    print(df.shape)
    return df




def calc_peritemcat(train):
    pass


def calc_pershop(train):
    pass


def calc_peritemshop(train):
    pass
