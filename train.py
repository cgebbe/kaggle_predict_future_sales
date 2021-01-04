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
    # eval_feature_combination()
    run_once()


def run_once():
    df = pd.read_csv('df_eval.csv', index_col="Unnamed: 0")
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

    pipeline(all)


def eval_feature_combination():
    assert 'item_cnt_month' not in COLS_FINAL
    space = {c: hyperopt.hp.choice(c, [0, 1]) for c in COLS_FINAL}
    dct = hyperopt.pyll.stochastic.sample(space)

    trials = hyperopt.Trials()
    best = hyperopt.fmin(
        fn=pipeline,
        space=space,
        algo=hyperopt.tpe.suggest,
        max_evals=600,
        trials=trials,
    )
    df_eval = pd.concat([pd.DataFrame(trials.results), pd.DataFrame(trials.vals)], axis=1)
    print(df_eval)
    df_eval.to_csv('df_eval.csv')
    pprint.pprint(best)


def pipeline(space=None):
    assert type(space) == dict
    print(space)

    # PARAMS
    RECALC_TRAIN = False
    USE_ONLY_TEST_IDS = True
    SPARSIFY_BY = 1000
    DO_SUBMIT = True
    LOG_DIR = pathlib.Path(__file__).parent.parent / 'logs' / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    utils.setup_logging(LOG_DIR / "log.txt")
    # pd.options.display.max_columns = 50

    # init training data with features
    if RECALC_TRAIN:
        df_data = data.generate(sparsify_by=SPARSIFY_BY)
        with open('train.pickle', 'wb') as f:
            pickle.dump(df_data, f)
    else:
        with open('train.pickle', 'rb') as f:
            df_data = pickle.load(f)
    if USE_ONLY_TEST_IDS:
        df_test = utils.read_csv('test.csv')
        mask = df_data['item_id'].isin(df_test['item_id'])
        logger.info("Using {}/{} rows".format(sum(mask), len(mask)))
        df_data = df_data.loc[mask, :]
    # print(df_data.head())

    # init model
    # model = models.Catboost()
    model = models.KNN()
    # model = models.NN()
    # model = models.Linear()

    # Train and validate multiple times (similar to cross-validation)
    dcts = []
    idxs_valid = [34] if DO_SUBMIT else [33, 32, 31]
    for idx_valid in idxs_valid:
        logger.info("Using idx={} for validation".format(idx_valid))

        # split train into [train,valid] and get X,y
        df_train = df_data.loc[df_data['date_block_num'] == (33 if DO_SUBMIT else idx_valid), :].copy()
        df_valid = df_data.loc[df_data['date_block_num'] < idx_valid, :].copy()

        # perform mean encoding if necessary
        if isinstance(model, models.SklearnInterface):
            df_train, _ = model.encode_features(df_train, tarcol='item_cnt_month', drop_org=True)
            df_valid, _ = model.encode_features(df_valid, tarcol='item_cnt_month', drop_org=True)
        Xvalid, yvalid = _getXy(df_train)
        Xtrain, ytrain = _getXy(df_valid)

        # pick only selected features from X. Cant do this before, because may also remove date_block_num
        if space is not None:
            cols = []
            for c in Xtrain.columns:
                if c in space and space[c] == True:
                    cols.append(c)
                elif c in space and space[c + 'encoded'] == True:
                    cols.append(c)
            Xtrain = Xtrain.loc[:, cols]
            Xvalid = Xvalid.loc[:, cols]

        # train
        dct = model.fit(Xtrain, ytrain, Xvalid, yvalid)
        dct['idx_valid'] = idx_valid
        dcts.append(dct)

        # save predictions for validation dataset
        if not DO_SUBMIT:
            yvalid_pred = model.predict(Xvalid)
            df_valid = pd.DataFrame({'true': yvalid, 'pred': yvalid_pred})
            path_csv = LOG_DIR / 'yvalid_{}.csv'.format(idx_valid)
            df_valid.to_csv(path_csv, index=False)

    # save total validation score
    df_eval = pd.DataFrame(dcts)
    df_eval.to_csv(LOG_DIR / 'df_eval.csv')

    # submit
    if DO_SUBMIT:
        logger.info("Starting submission")

        # load test
        df_test = utils.read_csv('test.csv')
        df_test['date_block_num'] = 34
        df_test.loc[df_test.shop_id == 0, 'shop_id'] = 57
        df_test.loc[df_test.shop_id == 1, 'shop_id'] = 58
        df_test.loc[df_test.shop_id == 10, 'shop_id'] = 11
        nrows_test = len(df_test)

        # apply features from df_data (date_block_num=34)
        if isinstance(model, models.SklearnInterface):
            df_data.loc[df_data.date_block_num == 34, 'item_cnt_month'] = float('NaN')  # to not influence mean encoding
            df_data, cols_to_encode = model.encode_features(df_data, tarcol='item_cnt_month', drop_org=False)
            # unencoded columns will be filtered out later automatically
        df_test = df_test.merge(df_data, how='inner', on=['date_block_num', 'shop_id', 'item_id'])
        Xtest, _ = _getXy(df_test)
        Xtest = Xtest.loc[:, Xtrain.columns.tolist()]

        # predict
        ytest = model.predict(Xtest)
        ytest = np.clip(ytest, 0, 20)  # clip values to [0,20], same clipping as target values
        df_test = pd.DataFrame({'ID': df_test.index,
                                'item_cnt_month': ytest,
                                })

        # save submission
        path_csv = LOG_DIR / 'submission.csv'
        df_test.to_csv(path_csv, index=False)
        logger.info("Wrote submission to {}".format(path_csv))
        if nrows_test != len(ytest):
            logger.warning("ytest has {} != {} rows".format(len(ytest), nrows_test))

    return df_eval['final'].mean()


def _getXy(df):
    X = df.drop(columns='item_cnt_month')
    y = df['item_cnt_month']
    return X, y


if __name__ == '__main__':
    main()
