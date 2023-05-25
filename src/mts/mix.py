from pathlib import Path

import pandas as pd

if __name__ == '__main__':
    dfs = []
    for file in Path('data/mixer').rglob('*.csv'):
        df = pd.read_csv(file, index_col=None)
        dfs.append(df)
    dfs = pd.concat(dfs, axis=0)
    ages = dfs.groupby('user_id')['age'].apply(lambda x: x.value_counts().index[0])
    males = dfs.groupby('user_id')['is_male'].agg(pd.Series.mean)
    ages.to_frame().join(males).reset_index().to_csv('data/submission_mix.csv', index=None)