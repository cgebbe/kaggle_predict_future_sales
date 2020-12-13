import pathlib
import pandas as pd

def main():
    # define paths
    path_in = pathlib.Path(__file__).parent.parent / 'data' / 'sales_train.csv'
    path_train = pathlib.Path(__file__).parent.parent / 'data' / 'sales_train_train.csv'
    path_valid = pathlib.Path(__file__).parent.parent / 'data' / 'sales_train_valid.csv'


    # split
    df = pd.read_csv(path_in)
    mask_valid = df['date_block_num'] == 33
    df_valid = df.loc[mask_valid, :]
    df_train = df.loc[~mask_valid, :]

    # export
    df_valid.to_csv(path_valid, index=False)
    df_train.to_csv(path_train, index=False)

if __name__ == '__main__':
    main()