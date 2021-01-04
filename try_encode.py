import pandas as pd


def main():
    df = pd.DataFrame({
        'category': [0, 0, 0, 1, 1, 1, 1],
        'target': [5, float('NaN'), 7, 18, 19, float('NaN'), 21],
    })
    df['category'] = df['category'].astype(str)
    print(df)


    colname = 'category'
    tarname  = 'target'

    # calculate mean per category
    mask = ~df[tarname].isna()
    mean_per_cat = df.loc[mask,:].groupby(colname).mean()[tarname]
    nrows_per_cat = df.loc[mask,:].groupby(colname).count()[tarname]

    # regularize using global mean
    nrows_valid = len(mask)
    nrows_trust = 5 # starting
    mean_global = df.loc[mask, tarname].mean()
    mean_per_cat = (nrows_per_cat * mean_per_cat + nrows_trust * mean_global) / (nrows_per_cat + nrows_trust)

    # apply
    encoded = df[colname].map(mean_per_cat)


if __name__ == '__main__':
    main()
