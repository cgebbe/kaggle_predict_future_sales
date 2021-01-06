import datetime
import pandas as pd
import pathlib
import yaml
import numpy as np
import pprint
import functools
import logging
import pickle
import hyperopt
import pprint
import sklearn.preprocessing
import tqdm

import data
import models
import utils

logger = logging.getLogger(__name__)
COLS_FINAL = ['date_block_num',
              'item_category_id',
              'item_id',
              'item_id_d_t12',
              'item_id_d_t13',
              'item_id_d_t23',
              'item_id_meansales',
              'item_id_nmonth_sales',
              'item_id_shop_id_d_t12',
              'item_id_shop_id_d_t13',
              'item_id_shop_id_d_t23',
              'item_id_shop_id_meansales',
              'item_id_shop_id_nmonth_sales',
              'item_id_shop_id_t-1',
              'item_id_shop_id_t-2',
              'item_id_shop_id_t-3',
              'item_id_shop_id_t-4',
              'item_id_t-1',
              'item_id_t-2',
              'item_id_t-3',
              'item_id_t-4',
              'item_price_d_t12',
              'item_price_d_t13',
              'item_price_d_t23',
              'item_price_t-1',
              'item_price_t-2',
              'item_price_t-3',
              'item_text_0',
              'item_text_1',
              'item_text_2',
              'item_text_3',
              'item_text_4',
              'item_text_5',
              'item_text_6',
              'item_text_7',
              'item_text_8',
              'item_text_9',
              'month',
              'shop_category',
              'shop_city',
              'shop_id',
              'shop_id_d_t12',
              'shop_id_d_t13',
              'shop_id_d_t23',
              'shop_id_meansales',
              'shop_id_nmonth_sales',
              'shop_id_t-1',
              'shop_id_t-2',
              'shop_id_t-3',
              'shop_id_t-4',
              'subtype',
              'type',
              'year']


def main():
    # find_best_features()
    run_with_selected_features()


def run_with_selected_features():
    df = pd.read_csv('df_eval_v3.csv', index_col="Unnamed: 0")
    first = df.loc[0, :].to_dict()
    all = {c: 1 for c in COLS_FINAL}

    # drop duplicate rows
    nrows = len(df)
    df = df.drop_duplicates()

    # get top performing feature combinations
    df = df.sort_values(by='loss', ascending=True)
    top1 = df.iloc[0, :].to_dict()

    # top10/20/50 yields same median values as top1...
    top10 = df.iloc[0:10, :].median(axis=0).to_dict()
    for k in top1.keys():
        try:
            if top1[k] != top10[k]:
                print("{}: top1={}, top10={}".format(k, top1[k], top10[k]))
        except Exception as err:
            print(err)

    run(top1)


def find_best_features():
    assert 'item_cnt_month' not in COLS_FINAL
    space = {c: hyperopt.hp.choice(c, [0, 1]) for c in COLS_FINAL}
    dct = hyperopt.pyll.stochastic.sample(space)

    trials = hyperopt.Trials()
    best = hyperopt.fmin(
        fn=run,
        space=space,
        algo=hyperopt.tpe.suggest,
        max_evals=600,
        trials=trials,
    )
    df_eval = pd.concat([pd.DataFrame(trials.results), pd.DataFrame(trials.vals)], axis=1)
    print(df_eval)
    df_eval.to_csv('df_eval.csv')
    pprint.pprint(best)


def run(space=None):
    assert type(space) == dict
    print(space)

    # PARAMS
    USE_ONLY_TEST_IDS = True
    DO_SUBMIT = True
    LOG_DIR = pathlib.Path(__file__).parent.parent.absolute() / 'logs' / datetime.datetime.now().strftime(
        "%Y%m%d_%H%M%S")
    utils.setup_logging(LOG_DIR / "log.txt")
    # pd.options.display.max_columns = 50

    # init training data with features
    df_data = data.generate(sparsify_by=1.0, use_cache=True)
    if USE_ONLY_TEST_IDS:
        df_test = utils.read_csv('test.csv')
        mask = df_data['item_id'].isin(df_test['item_id'])
        logger.info("Using {}/{} rows".format(sum(mask), len(mask)))
        df_data = df_data.loc[mask, :]
    # print(df_data.head())

    # init model
    model = models.Catboost()
    # model = models.KNN() # takes too long to train
    # model = models.NN()
    # model = models.Linear()

    # Train and validate multiple times (similar to cross-validation)
    dcts = []
    for idx_valid in [34]:  # tqdm.trange(30, 34 + 1):  # last training on 34 for later submit
        logger.info("Using idx={} for validation".format(idx_valid))

        # split train into [train,valid] and get X,y
        df_train = df_data.loc[df_data['date_block_num'] == (idx_valid if idx_valid < 34 else 33), :].copy()
        df_valid = df_data.loc[df_data['date_block_num'] < idx_valid, :].copy()
        Xvalid, yvalid = _getXy(df_train)
        Xtrain, ytrain = _getXy(df_valid)

        # pick only selected features from X. Cant do this before, because may also remove date_block_num
        if space is not None:
            cols = [c for c in Xtrain.columns if space[c] == True]
            Xtrain = Xtrain.loc[:, cols]
            Xvalid = Xvalid.loc[:, cols]

        # perform mean encoding if necessary
        if isinstance(model, models.SklearnInterface):
            Xtrain = model.encode_features(Xtrain, ytrain, drop_org=True)
            Xvalid = model.encode_features(Xvalid, yvalid, drop_org=True)
            cols = Xtrain.columns.tolist()  # afterwards Xtrain is not a DataFrame anymore
            scaler = sklearn.preprocessing.StandardScaler().fit(Xtrain)
            Xtrain = scaler.transform(Xtrain)
            Xvalid = scaler.transform(Xvalid)

        # train
        dct = model.fit(Xtrain, ytrain, Xvalid, yvalid)
        dct['idx_valid'] = idx_valid
        dcts.append(dct)

        # save predictions for validation dataset
        yvalid_pred = model.predict(Xvalid)
        df_valid = pd.DataFrame({'true': yvalid, 'pred': yvalid_pred})
        path_csv = LOG_DIR / 'yvalid_{}.csv'.format(idx_valid)
        df_valid.to_csv(path_csv, index=False)

    # save total validation score
    df_eval = pd.DataFrame(dcts)
    df_eval.to_csv(LOG_DIR / 'df_eval.csv')
    logger.info(df_eval)

    # submit
    if DO_SUBMIT:
        assert idx_valid == 34, "last training was not on idx_valid=34!"
        logger.info("Starting submission")

        # load test
        df_test = utils.read_csv('test.csv')
        df_test['date_block_num'] = 34
        df_test.loc[df_test.shop_id == 0, 'shop_id'] = 57
        df_test.loc[df_test.shop_id == 1, 'shop_id'] = 58
        df_test.loc[df_test.shop_id == 10, 'shop_id'] = 11
        Xtest, _ = _getXy(df_test)
        nrows_test = len(Xtest)

        # apply features from df_data (date_block_num=34)
        if isinstance(model, models.SklearnInterface):
            df_data.loc[df_data.date_block_num == 34, 'item_cnt_month'] = float('NaN')  # to not influence mean encoding
            Xdata, ydata = _getXy(df_data)
            Xdata = model.encode_features(Xdata, ydata, drop_org=False)
        else:
            Xdata, ydata = _getXy(df_data)
        Xtest = Xtest.merge(Xdata, how='inner', on=['date_block_num', 'shop_id', 'item_id'])
        Xtest = Xtest.loc[:, cols]
        if isinstance(model, models.SklearnInterface):
            Xtest = scaler.transform(Xtest)

        # predict
        ytest = model.predict(Xtest)

        # postprocess
        ytest = np.clip(ytest, 0, 20)  # clip values to [0,20], same clipping as target values
        if False:
            ytest *= 0.2839 / np.mean(ytest)  # known mean
            ytest = np.clip(ytest, 0, 20)

        df_test = pd.DataFrame({'ID': df_test.index,
                                'item_cnt_month': ytest,
                                })

        # save submission
        path_csv = LOG_DIR / 'submission.csv'
        df_test.to_csv(path_csv, index=False)
        logger.info("Wrote submission to {}".format(path_csv))
        if nrows_test != len(ytest):
            logger.warning("ytest has {} != {} rows".format(len(ytest), nrows_test))

    # calculate final KPI without idx_eval==34
    mask = df_eval['idx_valid'] < 34
    kpi_final = df_eval.loc[mask, 'final'].mean()
    return kpi_final


def _getXy(df):
    if 'item_cnt_month' in df.columns:
        X = df.drop(columns='item_cnt_month')
        y = df['item_cnt_month']
    else:
        X = df
        y = None
    return X, y


if __name__ == '__main__':
    main()
