import abc
import catboost
import numpy as np
import logging
import pandas as pd
import sklearn.preprocessing
import sklearn.neighbors
import sklearn.neural_network
import sklearn.linear_model

import utils

logger = logging.getLogger(__name__)


class Interface(abc.ABC):
    def fit(self, Xtrain, ytrain, Xvalid, yvalid):
        pass

    def predict(self, Xtest):
        pass


class Catboost(Interface):
    def __init__(self):
        self.model = catboost.CatBoostRegressor(random_seed=42,
                                                iterations=125,
                                                loss_function='RMSE',  # MSE not supported?!
                                                train_dir='catboost',
                                                task_type='GPU',
                                                )

    def fit(self, Xtrain, ytrain, Xvalid, yvalid):
        cat_features = np.where(Xtrain.dtypes != float)[0]
        self.model.fit(Xtrain, ytrain,
                       eval_set=(Xvalid, yvalid),
                       cat_features=cat_features,
                       # silent=True,
                       metric_period=25,
                       # use_best_model=False # mainly to check
                       )
        # eval model
        logger.info("Eval model")
        eval = pd.DataFrame({'name': self.model.feature_names_,
                             'importance': self.model.feature_importances_})
        eval.sort_values(by='importance', ascending=True, inplace=True)
        print(eval)

        # get scores
        res = self.model.get_best_score()
        dct = {'train': res['learn']['RMSE'],
               'val': res['validation']['RMSE'],
               'niter': self.model.get_best_iteration(),
               'final': res['validation']['RMSE'],
               }
        return dct

    def predict(self, Xtest):
        ytest = self.model.predict(Xtest)
        return ytest


class SklearnInterface(Interface):
    def __init__(self):
        self.clf = None

    def fit(self, Xtrain, ytrain, Xvalid, yvalid):
        # train
        self.clf.fit(Xtrain, ytrain)

        # evaluate performance on validation set
        yvalid_pred = self.predict(Xvalid)
        err = np.abs(yvalid - yvalid_pred).mean()
        dct = {'MSE': err,
               'final': err,
               }
        return dct

    def predict(self, Xtest):
        ytest = self.clf.predict(Xtest)
        return ytest

    def encode_features(self, df, tarcol, drop_org=True, cols_to_ignore=None):
        # specify columns to encode
        cols_to_encode = df.dtypes.index[df.dtypes != float].tolist()
        if cols_to_ignore:
            for ci in cols_to_ignore:
                cols_to_encode.remove(ci)
        if tarcol in cols_to_encode:
            cols_to_encode.remove(tarcol)

        # perform encoding
        for catcol in cols_to_encode:
            df.loc[:, catcol + '_encoded'] = utils.encode_category_using_smoothing(df, catcol, tarcol)
            # df[:,catcol] = utils.encode_category_using_cumsum(df, catcol, tarcol)
            if drop_org:
                df.drop(columns=catcol, inplace=True)

        # replace remaining nans
        cols_with_nans = df.columns[df.isna().sum(axis=0).astype(bool)]
        logger.info("Replacing Nans in columns {}".format(cols_with_nans))
        df.fillna(value=-1, inplace=True)
        return df, cols_to_encode


class Linear(SklearnInterface):
    def __init__(self):
        """
        Warning, to use SVR for >10.000 samples, see https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
        Therefore, use one of the following:
            sklearn.svm.LinearSVR
            sklearn.linear_model.SGDRegressor
            sklearn.linear_model.LinearRegression
        Comparison https://sdsawtelle.github.io/blog/output/week2-andrew-ng-machine-learning-with-python.html
        """
        super(Linear, self).__init__()
        # self.clf = sklearn.svm.SVR()
        self.clf = sklearn.linear_model.LinearRegression()


class KNN(SklearnInterface):
    def __init__(self):
        super(KNN, self).__init__()
        self.clf = sklearn.neighbors.KNeighborsRegressor()


class NN(SklearnInterface):
    def __init__(self):
        super(NN, self).__init__()
        self.clf = sklearn.neural_network.MLPRegressor([500, 50, 5])
