import pandas as pd
import catboost
import os
import pathlib
import numpy as np
import datetime
from final.src.dataset import Dataset
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # load Data
    if cfg.is_train:
        path_csv_train = pathlib.Path(__file__).parent.parent / 'data' / 'sales_train_train.csv'
        path_csv_valid = pathlib.Path(__file__).parent.parent / 'data' / 'sales_train_valid.csv'
        path_csv_test = None  # pathlib.Path(__file__).parent.parent / 'data' / 'test.csv'
    else:
        path_csv_train = pathlib.Path(__file__).parent.parent / 'data' / 'sales_train.csv'
        path_csv_valid = pathlib.Path(__file__).parent.parent / 'data' / 'sales_train_valid.csv'  # dummy valid
        path_csv_test = pathlib.Path(__file__).parent.parent / 'data' / 'test.csv'
    ds_train = Dataset(path_csv_train, is_train=True)
    ds_valid = Dataset(path_csv_valid, is_train=True)

    # calculate features
    ds_train.calc_features()
    ds_valid.calc_features()

    # setup model
    model = catboost.CatBoostRegressor(random_seed=42,
                                       iterations=100,
                                       loss_function='RMSE',  # MSE not supported?!
                                       )

    # train model
    X_train, y_train = ds_train.get_Xy()
    X_valid, y_valid = ds_valid.get_Xy()
    idx_col_categorical = [1 if t == 'object' else 0 for t in X_train.dtypes]
    model.fit(X_train, y_train,
              eval_set=(X_valid, y_valid),
              cat_features=idx_col_categorical,
              )

    # infer for test
    if cfg.is_train:
        ds_test = Dataset(path_csv_test, is_train=False)
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
