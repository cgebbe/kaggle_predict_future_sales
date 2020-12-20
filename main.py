import pandas as pd
import plotly.express as px
import plotly.io as pio
import catboost
import os
import pathlib
import numpy as np
import datetime
from final.src.dataset import Dataset
import hydra
import logging
from omegaconf import DictConfig, OmegaConf

pio.renderers.default = 'browser'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info('setup logger')


@hydra.main(config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    if cfg.do_submit:
        ds_test = Dataset(use_testcsv=True)  # dateblock_target=34
        ds_train = Dataset(use_testcsv=False, dateblock_target=33)
        model = train(ds_train, ds_train)
        submit(model, ds_test)
    else:
        times = sorted(list(range(33, 30, -1)))
        ds = dict()
        for i, t in enumerate(times):
            ds[t] = Dataset(use_testcsv=False, dateblock_target=t)
            ds[t].calc_features()
            if i == 0:
                print(ds[t].df.columns)

        for t in times[1:]:
            print("=== Evaluating on dateblock {}".format(t))
            ds_valid = ds[t]
            ds_train = ds[t - 1]

            # Simple baseline: What happens if we simply use last month sales? > RMSE of ~2
            # y_pred = ds_valid.df['n_peritem_last1']
            # y_true = ds_valid.df['item_cnt_month']
            # rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

            model = train(ds_train, ds_valid)
            eval_predictions(model, ds_valid, ds_train)
            # eval_model(model)
            # model.save_model('catboost/model.cbm')


def eval_model(model):
    """
    Evaluate model, most useful features, etc...

    :param model: CatBoostRegressor
    :return:
    """
    df = pd.DataFrame({'feature_names': model.feature_names_,
                       'feature_importance': model.feature_importances_})
    df.sort_values(by='feature_importance', inplace=True)
    print(df)


def eval_predictions(model, ds_valid, ds_train=None):
    """
    Evaluate mode on dataset. Includes plots and KPIs

    :param model: CatBoostRegressor
    :param ds_valid: dataset
    :return:
    """
    X, y_true = ds_valid.get_Xy()
    y_pred = model.predict(X)
    y_pred = np.clip(y_pred, 0, 20)  # clip values to [0,20], same clipping as target values

    #
    df = ds_valid.df.copy()
    df['y_true'] = y_true
    df['y_pred'] = y_pred

    # calc errors
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    print("RMSE={:.3f}".format(rmse))
    df['abs_err'] = np.abs(y_true - y_pred)
    df['rel_abs_err'] = np.abs(y_true - y_pred) / y_true

    # ANALYSIS (might be better in jupyter notebook for readers, but primarily for me)
    if ds_train:
        # add information: Which valid items exist also in train?
        items_in_train = set(ds_train.df['item_id'].unique().tolist())
        items_in_valid = set(ds_valid.df['item_id'].unique().tolist())
        df['item_in_train'] = ds_valid.df['item_id'].isin(items_in_train)
        print("{}/{} items in valid exist in train"
              .format(len(items_in_valid & items_in_train), len(items_in_valid)))

        # > 76% of error come from items, which do exist in train
        px.histogram(df, y='abs_err', x='item_in_train', histnorm='percent').show()

    # > Error-sum comes 50% from y_true=0, 20% from y_true=1, 5% from y_true=20
    px.histogram(df, y='abs_err', x='y_true', histnorm='percent').show()

    # > error-sum comes 66% from y_pred < 1.0
    px.histogram(df, y='abs_err', x='y_pred', histnorm='percent', cumulative=True).show()

    # > Each item contributes at most 1% to total error.
    #   However, there seems to be correlation with close-by itemIDs! (E.g. 4719+4721 and 6497+6503+6507)
    #   Check correlation between similar itemIDs
    px.histogram(df, y='abs_err', x='item_id', histnorm='percent',
                 nbins=int(df['item_id'].max())).show()

    # > Shops also contribute similiarly to total error.
    #   Several shops NO sales at all?! --> Double check in EDA
    #   Only 5 shops with sumerror>3%: 25,27,28,31,42 - Out of 59 shops
    px.histogram(df, y='abs_err', x='shop_id', histnorm='percent').show()

    # > Median error for y_true=0 is 0.08, seems okay?!
    px.box(df, y='abs_err', x='y_true').show()


def train(ds_train, ds_valid):
    """
    Train model on training dataset and directly evaluate on valid

    :param ds_train: train dataset
    :param ds_valid: valid dataset
    :return: CatBoostRegressor model
    """
    X_valid, y_valid = ds_valid.get_Xy()
    X_train, y_train = ds_train.get_Xy()
    cat_features = np.where(X_valid.dtypes != float)[0]
    # print(X_valid.dtypes)

    # train model
    model = catboost.CatBoostRegressor(random_seed=42,
                                       iterations=100,
                                       loss_function='RMSE',  # MSE not supported?!
                                       train_dir='catboost'
                                       )
    model.fit(X_train, y_train,
              eval_set=(X_valid, y_valid),
              cat_features=cat_features,
              # silent=True,
              metric_period=50,
              )
    return model


def submit(model, ds_test):
    """

    :param model:
    :param ds_test:
    :return:
    """
    ds_test.calc_features()
    X_test, _ = ds_test.get_Xy()
    y_test = model.predict(X_test)
    y_test = np.clip(y_test, 0, 20)  # clip values to [0,20], same clipping as target values
    df_test = pd.DataFrame({'ID': ds_test.df.index.values,
                            'item_cnt_month': y_test,
                            })

    # save submission
    filename = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S") + '.csv'
    path_sub = pathlib.Path(__file__).parent.parent / 'submissions' / filename
    path_sub.parent.mkdir(parents=True, exist_ok=True)
    df_test.to_csv(path_sub, index=False)
    d = 0


if __name__ == '__main__':
    main()
