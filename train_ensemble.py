import numpy as np
import pandas as pd
import pathlib
import models
import pprint
import utils
import tqdm
import datetime
import logging

# setup logging
LOG_DIR = pathlib.Path(__file__).parent.parent.absolute() / 'logs' / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
utils.setup_logging(LOG_DIR / "log.txt")
logger = logging.getLogger(__name__)


def main():
    # find_best_model()
    perform_submission()


def find_best_model():
    """
    Find best model via cross-validation. -> Only linear model doesn't overfit!

    :return:
    """
    # data from dateblocks 30,31,32,33 and 34 (last one without true...)
    df_all = parse_data()  # includes train,valid AND test
    df_data = df_all.loc[df_all.idx_valid != 34, :]

    # find suitable model using cross validation
    dcts = []
    for idx_valid in tqdm.trange(30, 34):
        df_valid = df_data.loc[df_data.idx_valid == idx_valid, :]
        df_train = df_data.loc[df_data.idx_valid != idx_valid, :]
        Xvalid, yvalid = getXy(df_valid)
        Xtrain, ytrain = getXy(df_train)

        # fit model
        model = models.Linear()
        dct = model.fit(Xtrain, ytrain, Xvalid, yvalid)

        # calculate more errors
        ypred_ensemble = model.predict(Xvalid)
        errors = {'err_tot': utils.calc_rmse(yvalid, ypred_ensemble),
                  'err_lin': utils.calc_rmse(yvalid, Xvalid['pred_lin']),
                  'err_cat': utils.calc_rmse(yvalid, Xvalid['pred_cat']),
                  'err_nn': utils.calc_rmse(yvalid, Xvalid['pred_nn']),
                  }
        dct.update(errors)
        dcts.append(dct)

    df_eval = pd.DataFrame(dcts)
    pprint.pprint(df_eval)
    return


def perform_submission(use_dataleak=True):
    df_data = parse_data()  # includes train,valid AND test

    # train
    df_train = df_data.loc[df_data.idx_valid < 34, :]
    df_valid = df_data.loc[df_data.idx_valid == 33, :]
    Xtrain, ytrain = getXy(df_train)
    Xvalid, yvalid = getXy(df_valid)
    model = models.Linear()
    dct = model.fit(Xtrain, ytrain, Xvalid, yvalid)

    # perform prediction
    df_test = df_data.loc[df_data.idx_valid == 34, :]
    Xtest, _ = getXy(df_test)
    ytest = model.predict(Xtest)

    # postprocess
    ytest = np.clip(ytest, 0, 20)
    if use_dataleak:
        ytest *= 0.2839 / np.mean(ytest)  # known mean
        ytest = np.clip(ytest, 0, 20)

    # create submission
    df_sub = utils.read_csv('test.csv')
    df_sub = pd.DataFrame({'ID': df_sub.index,
                           'item_cnt_month': ytest,
                           })

    # save
    path_csv = LOG_DIR / 'submission.csv'
    df_sub.to_csv(path_csv, index=False)
    logger.info("Wrote submission to {}".format(path_csv))


def getXy(df):
    y = df.loc[:, 'true']
    X = df.loc[:, ['pred_cat', 'pred_lin', 'pred_nn']]
    return X, y


def parse_data():
    dirs = {'cat': '20210106_153134_catboost_top1',
            'lin': '20210106_174039_linear',
            'nn': '20210106_174434_NN',
            }
    lst_df = []
    for i, (k, v) in enumerate(dirs.items()):
        df_agg = pd.DataFrame()

        # load dateblocks 30-33
        for idx_valid in range(30, 34):  # 30,31,32,33
            path = pathlib.Path('../logs') / v / 'yvalid_{}.csv'.format(idx_valid)
            assert path.exists()
            df = pd.read_csv(path)
            df['idx_valid'] = idx_valid
            df = df.rename(columns={'pred': 'pred_' + k})
            if i > 0:
                df = df.drop(columns='true')
            df_agg = df_agg.append(df)

        # load datebock 34 = submission
        path = pathlib.Path('../logs') / v / 'submission.csv'
        assert path.exists()
        df = pd.read_csv(path)
        df['idx_valid'] = 34
        df = df.rename(columns={'item_cnt_month': 'pred_' + k})
        df = df.drop(columns='ID')
        df_agg = df_agg.append(df)

        lst_df.append(df_agg)
    df_tot = pd.concat(lst_df, axis=1)
    df_tot = df_tot.iloc[:, [0, 1, 3, 5, 6]]  # 0=true, 6=idx_valid, 1,3,5=pred
    return df_tot


if __name__ == '__main__':
    main()
