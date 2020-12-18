import pandas as pd
import catboost
import os
import pathlib
import numpy as np
import datetime
from final.src.dataset import Dataset
import hydra
import logging
from omegaconf import DictConfig, OmegaConf


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
        times = sorted(list(range(33, 29, -1)))
        ds = dict()
        for i,t in enumerate(times):
            ds[t] = Dataset(use_testcsv=False, dateblock_target=t)
            ds[t].calc_features()
            if i==0:
                print(ds[t].df.columns)

        for t in times[1:]:
            print("=== Evaluating on dateblock {}".format(t))
            ds_valid = ds[t]
            ds_train = ds[t - 1]
            model = train(ds_train, ds_valid)


def train(ds_train, ds_valid):
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
    # model.save_model('catboost/model.cbm')
    return model


def submit(model, ds_test):
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
