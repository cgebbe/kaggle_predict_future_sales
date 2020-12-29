import functools
import pandas as pd
import pathlib

@functools.lru_cache()
def read_csv(filename,
             parent_path='../data/',
             ):
    """
    Wrapper around pd.read_csv(), but with cache!

    :param filename: e.g. sales_train.csv
    :param parent_path: path to parent folder
    :return: DataFrame
    """
    path_csv = pathlib.Path(parent_path) / filename
    assert path_csv.is_file()
    df = pd.read_csv(path_csv)
    return df

