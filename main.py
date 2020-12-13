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
    ds_test = Dataset(use_testcsv=True) # dateblock_target=34
    ds_valid = Dataset(use_testcsv=False, dateblock_target=33)
    ds_train = Dataset(use_testcsv=False, dateblock_target=32 if cfg.do_eval_internally else 33)

    # calc features
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
    if not cfg.do_eval_internally:
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
