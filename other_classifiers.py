"""
SVM
KNN
ANN
"""
import sklearn.svm
import numpy as np
import pandas as pd
import seaborn as sns
import numpy
import sklearn.preprocessing
import sklearn.neighbors
import sklearn.neural_network
import sklearn.linear_model

def main():
    df = sns.load_dataset('exercise')
    df.drop(columns=['Unnamed: 0', 'id'], inplace=True)

    # encode all categories
    tarcol = 'pulse'
    cat_columns = df.dtypes.index[df.dtypes != float].tolist()
    for catcol in cat_columns:
        if catcol == tarcol:
            continue
        df[catcol + '_encoded'] = encode_category(df, catcol, tarcol, use_nans=False)
        df.drop(columns=catcol, inplace=True)
    df.head()

    # split into train, test
    def get_Xy(df, tarcol):
        X = df.drop(columns=tarcol)
        y = df[tarcol]
        return X, y

    Xtrain, ytrain = get_Xy(df.iloc[:70, :], tarcol)
    Xvalid, yvalid = get_Xy(df.iloc[70:, :], tarcol)

    # scale X
    scaler = sklearn.preprocessing.StandardScaler().fit(Xtrain)
    Xtrain = scaler.transform(Xtrain)
    Xvalid = scaler.transform(Xvalid)

    # predict
    y_SVM = predict_using_lin(Xtrain, ytrain, Xvalid, yvalid)
    y_KNN = predict_using_KNN(Xtrain, ytrain, Xvalid, yvalid)
    y_ANN = predict_using_ANN(Xtrain, ytrain, Xvalid, yvalid)

    # ensemble using mean (?)
    y_all = np.stack([y_SVM, y_KNN, y_ANN]).mean(axis=0)
    err = np.abs(yvalid - y_all).mean()
    print("error={}".format(err))


def predict_using_lin(Xtrain, ytrain, Xvalid, yvalid_true):
    """
    Warning, to use SVR for >10.000 samples, see https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
    Therefore, use one of the following:
        sklearn.svm.LinearSVR
        sklearn.linear_model.SGDRegressor
        sklearn.linear_model.LinearRegression
    Comparison https://sdsawtelle.github.io/blog/output/week2-andrew-ng-machine-learning-with-python.html
    """
    # clf = sklearn.svm.SVR()
    clf = sklearn.linear_model.LinearRegression()
    clf.fit(Xtrain, ytrain)
    yvalid_pred = clf.predict(Xvalid)
    err = np.abs(yvalid_true - yvalid_pred).mean()
    print("error={}".format(err))
    return yvalid_pred


def predict_using_KNN(Xtrain, ytrain, Xvalid, yvalid_true):
    clf = sklearn.neighbors.KNeighborsRegressor()
    clf.fit(Xtrain, ytrain)
    yvalid_pred = clf.predict(Xvalid)
    err = np.abs(yvalid_true - yvalid_pred).mean()
    print("error={}".format(err))
    return yvalid_pred


def predict_using_ANN(Xtrain, ytrain, Xvalid, yvalid_true):
    clf = sklearn.neural_network.MLPRegressor([500, 50, 5])
    clf.fit(Xtrain, ytrain)
    yvalid_pred = clf.predict(Xvalid)
    err = np.abs(yvalid_true - yvalid_pred).mean()
    print("error={}".format(err))
    return yvalid_pred


def encode_category(df, colname, tarname, use_nans=True):
    """
    Encodes a column using target mean with expanding mean regularization

    :param df: dataframe
    :param colname: name of column to be replaced
    :param tarname: name of column with target values
    :Param use_nans: The first category member will always be encoded using NaNs. If True, these NaNs are replaced with the global mean
    :return:
    """
    cumsum = df.groupby(colname)[tarname].cumsum() - df[tarname]
    cumcnt = df.groupby(colname).cumcount()
    encoded = cumsum / cumcnt

    if not use_nans:
        globalmean = df[tarname].mean()
        encoded.replace(float('NaN'), globalmean, inplace=True)
    return encoded


if __name__ == '__main__':
    main()
